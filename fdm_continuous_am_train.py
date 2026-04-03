"""
FDM Continuous AM Training — Paper 2
=====================================
Trains a language model to encode and decode arbitrary text concepts
via continuous amplitude modulation (AM) in an FDM signal.

Architecture:
  - ProjectionDirections: learned w_k per channel, maps concept embedding → scalar amplitude
  - ReadoutProbe: MLP at layer 14, maps residual stream → concept embedding (decode supervision)
  - FDMAMEncoder: generates composite sinusoidal signal from continuous amplitudes
  - LM (frozen or unfrozen depending on stage): reads signal, generates concept text

Training schedule (4 stages):
  Stage 0: Extract initial projection directions from frozen base model (no gradient)
  Stage 1: Train decode only — freeze w_k, train LM on cross-entropy
  Stage 2: Train encode only — freeze LM, optimize w_k via reconstruction loss
  Stage 3: Joint fine-tuning — train everything end-to-end
  Stage 4: Generalization eval — held-out concepts never seen in training

Usage:
  python fdm_continuous_am_train.py --stage 0   # extract directions
  python fdm_continuous_am_train.py --stage 1   # train decode
  python fdm_continuous_am_train.py --stage 2   # train encode
  python fdm_continuous_am_train.py --stage 3   # joint
  python fdm_continuous_am_train.py --eval      # generalization eval
  python fdm_continuous_am_train.py --all       # run all stages sequentially
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

MODEL_ID   = "Qwen/Qwen3-0.6B-Base"   # swap for any model
MODEL_SHORT = "qwen3"
DTYPE      = torch.bfloat16

NUM_CHANNELS   = 40          # carrier frequencies
EMBED_DIM      = 1024        # model hidden size (set per model)
READOUT_LAYER  = 14          # layer where signal reading hands off to answer composition
SAMPLE_RATE    = 100.0       # Hz
NUM_TOKENS_PER_ENCODER = 256 # samples per encoder pass
NUM_LEVELS     = 256         # quantization levels (higher than paper 1's 64)
LAMBDA_RECON   = 1.0         # reconstruction loss weight — increased after normalization fix

BATCH_SIZE     = 4        # reduced for 512-token prompt repetition
MAX_LENGTH     = 768          # FDM signal + question + answer
OUTPUT_DIR     = "fdm_continuous_am"

# Stage-specific training config
STAGE_CONFIG = {
    1: {"lr": 3e-5, "epochs": 5,  "samples": 20000, "freeze_proj": True,  "freeze_lm": False},
    2: {"lr": 1e-4, "epochs": 3,  "samples": 10000, "freeze_proj": False, "freeze_lm": True},
    3: {"lr": 5e-6, "epochs": 5,  "samples": 30000, "freeze_proj": False, "freeze_lm": False},
}

# Fact corpus: (slot_name, concept_text) pairs
# In a real run, load from a knowledge base or dataset
FACT_CORPUS = [
    ("capital",    "Paris"),
    ("capital",    "Berlin"),
    ("capital",    "Tokyo"),
    ("capital",    "London"),
    ("capital",    "Madrid"),
    ("animal",     "elephant"),
    ("animal",     "dolphin"),
    ("animal",     "eagle"),
    ("element",    "hydrogen"),
    ("element",    "oxygen"),
    ("element",    "carbon"),
    ("inventor",   "Edison"),
    ("inventor",   "Tesla"),
    ("inventor",   "Curie"),
    ("color",      "crimson"),
    ("color",      "azure"),
    ("color",      "emerald"),
    ("material",   "titanium"),
    ("material",   "graphene"),
    ("material",   "ceramic"),
    # ... extend with real knowledge base
]

# Held-out for generalization eval (never seen in training)
HELD_OUT_CONCEPTS = [
    ("capital",  "Ottawa"),
    ("capital",  "Canberra"),
    ("animal",   "narwhal"),
    ("element",  "xenon"),
    ("inventor", "Faraday"),
]


# ============================================================
# Projection Directions (learnable w_k per channel)
# ============================================================

class ProjectionDirections(nn.Module):
    """
    Learnable projection directions w_k for each channel k.
    Maps a concept embedding (dim=EMBED_DIM) to a scalar amplitude.

    amplitude_k = dot(embed(concept), w_k)  ∈ [-1, 1] after tanh
    """

    def __init__(self, num_channels, embed_dim):
        super().__init__()
        # Initialize near zero — Stage 0 will replace with meaningful directions
        self.directions = nn.Parameter(
            torch.randn(num_channels, embed_dim) * 0.01
        )

    def project(self, embeddings):
        """
        embeddings: (batch, embed_dim)
        returns: (batch, num_channels) — one amplitude per channel
        """
        # Normalize directions for stability
        dirs = F.normalize(self.directions, dim=-1)  # (num_channels, embed_dim)
        amps = embeddings @ dirs.T                    # (batch, num_channels)
        return torch.tanh(amps)                       # squash to [-1, 1]

    def load_from_mean_diff(self, directions_tensor):
        """Replace with Stage 0 extracted directions."""
        with torch.no_grad():
            self.directions.copy_(directions_tensor)


# ============================================================
# Readout Probe (decode supervision at layer 14)
# ============================================================

class ReadoutProbe(nn.Module):
    """
    Small MLP attached at READOUT_LAYER.
    Maps the mean residual stream over FDM token positions → concept embedding.
    Used for reconstruction loss during Stage 2 and Stage 3.
    """

    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, hidden_dim) — layer 14 output
        returns: (batch, embed_dim)
        """
        # Mean pool over FDM signal token positions (first 512 positions)
        fdm_states = hidden_states[:, :512, :]
        pooled = fdm_states.mean(dim=1)
        return self.net(pooled.float())


# ============================================================
# Continuous AM FDM Encoder
# ============================================================

class ContinuousAMFDMEncoder:
    """
    Encodes continuous amplitudes (one per channel) into a composite
    sinusoidal signal, quantizes, and maps to token IDs.

    Signal equation:
        x(t) = sum_k amplitude_k * sin(2*pi*f_k*t)

    Unlike paper 1 (ASK), amplitude_k is a continuous float in [-1, 1],
    not a binary {a_low, a_high} derived from a bit string.
    """

    def __init__(self, vocab_size, num_tokens=256, sample_rate=100.0,
                 num_levels=256, seed=42):
        self.num_tokens     = num_tokens
        self.sample_rate    = sample_rate
        self.num_levels     = num_levels
        self.num_channels   = NUM_CHANNELS
        self.carrier_freqs  = [1.0 + i for i in range(NUM_CHANNELS)]
        self.t              = np.arange(num_tokens) / sample_rate

        # S-random interleaver (same as paper 1)
        S = int(np.sqrt(num_tokens / 2))
        self.interleaver = self._s_random_interleaver(num_tokens, S)

        # Token mapping: pseudorandom bijection over vocab
        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(vocab_size, size=num_levels, replace=False)

    def _s_random_interleaver(self, length, S, seed=42):
        import random as rnd
        rnd.seed(seed)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = all(
                    abs(interleaver[j] - interleaver[k]) >= S
                    for k in range(max(0, i - S + 1), i)
                )
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                    break
        return interleaver

    def encode(self, amplitudes):
        """
        amplitudes: np.array of shape (num_channels,), values in [-1, 1]
        returns: (fdm_text, token_ids)

        Key difference from paper 1:
          - No bit periods, no switching
          - Each carrier has a fixed amplitude for the full signal duration
          - x(t) = sum_k amp_k * sin(2*pi*f_k*t)
        """
        # Build composite signal — static amplitude per carrier
        composite = np.zeros(self.num_tokens)
        for ch, amp in enumerate(amplitudes):
            composite += amp * np.sin(2 * np.pi * self.carrier_freqs[ch] * self.t)

        # Normalize by number of channels (max possible range is ±NUM_CHANNELS)
        sig_min = -self.num_channels
        sig_max =  self.num_channels
        sig_range = sig_max - sig_min + 1e-10

        # Encoder 1: natural order
        norm1 = (composite - sig_min) / sig_range
        q1 = np.clip(
            np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int),
            0, self.num_levels - 1
        )
        tokens1 = [int(self.token_map[q]) for q in q1]

        # Encoder 2: S-random interleaved
        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.clip(
            np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int),
            0, self.num_levels - 1
        )
        tokens2 = [int(self.token_map[q]) for q in q2]

        return tokens1 + tokens1  # prompt repetition — attention scaffold


# ============================================================
# Concept Embedder (frozen — extracts concept representations)
# ============================================================

class ConceptEmbedder:
    """
    Extracts concept embeddings from the LM's residual stream at READOUT_LAYER.
    Used in Stage 0 to compute initial projection directions,
    and at training time to produce target embeddings for reconstruction loss.

    The embedding for concept text C is:
        embed(C) = mean over token positions of layer-14 hidden states
                   when running C through the model as a plain text prompt.
    """

    def __init__(self, model, tokenizer, device, layer=READOUT_LAYER):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.layer     = layer
        self._cache    = {}

    @torch.no_grad()
    def embed(self, text):
        """Returns a (embed_dim,) tensor for the given text."""
        if text in self._cache:
            return self._cache[text]

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=32
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        # Mean pool layer READOUT_LAYER hidden states
        hidden = outputs.hidden_states[self.layer]  # (1, seq_len, hidden_dim)
        emb = hidden[0].mean(dim=0).float()         # (hidden_dim,)

        self._cache[text] = emb
        return emb

    @torch.no_grad()
    def embed_batch(self, texts):
        """Returns (batch, hidden_dim)."""
        return torch.stack([self.embed(t) for t in texts])


# ============================================================
# Stage 0: Extract initial projection directions
# ============================================================

def stage0_extract_directions(model, tokenizer, device):
    """
    For each channel k, compute a projection direction w_k using the
    mean-difference method from the Anthropic emotion vector paper:

      w_k = mean(embed(positive_concepts)) - mean(embed(negative_concepts))

    For factual concepts, 'positive' = the concept we want to encode on
    channel k, 'negative' = the mean over all other concepts.

    Returns a (num_channels, embed_dim) tensor of initial directions.
    """
    print("\n=== Stage 0: Extracting initial projection directions ===")

    embedder = ConceptEmbedder(model, tokenizer, device)

    # Collect all concept embeddings
    all_texts   = list(set(v for _, v in FACT_CORPUS))
    all_embeds  = embedder.embed_batch(all_texts)   # (N, hidden_dim)
    mean_embed  = all_embeds.mean(dim=0)            # (hidden_dim,)

    embed_dim   = all_embeds.shape[1]
    directions  = torch.zeros(NUM_CHANNELS, embed_dim)

    # Assign one concept per channel (cycling through corpus)
    # In a real system you'd have a principled assignment strategy
    for ch in range(NUM_CHANNELS):
        idx  = ch % len(all_texts)
        text = all_texts[idx]
        emb  = embedder.embed(text)

        # Mean-difference direction: this concept vs the background
        directions[ch] = emb - mean_embed

    # Normalize
    directions = F.normalize(directions, dim=-1)

    # Remove top principal components from neutral text (confound removal)
    # Run model on neutral prompts, compute PCA, project out
    neutral_texts = [
        "The answer is", "Please respond", "Hello there",
        "The following is", "In summary", "As requested",
    ]
    neutral_embeds = embedder.embed_batch(neutral_texts)  # (6, hidden_dim)

    # SVD to get top neutral directions
    _, _, Vt = torch.linalg.svd(neutral_embeds - neutral_embeds.mean(0), full_matrices=False)
    n_confounds = min(3, Vt.shape[0])
    confound_dirs = Vt[:n_confounds].to(directions.device)  # (n_confounds, hidden_dim)

    # Project out confounds from each direction
    for i in range(NUM_CHANNELS):
        d = directions[i]
        for c in confound_dirs:
            d = d - (d @ c) * c
        directions[i] = F.normalize(d, dim=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(directions, os.path.join(OUTPUT_DIR, "stage0_directions.pt"))
    print(f"  Saved {directions.shape} directions to {OUTPUT_DIR}/stage0_directions.pt")

    return directions


# ============================================================
# Dataset
# ============================================================

class ContinuousAMDataset(Dataset):
    """
    Each sample:
      - Randomly selects a concept from FACT_CORPUS
      - Assigns it to a random channel
      - Encodes amplitudes: channel k gets embed(concept) projected onto w_k,
        all other channels get small random noise
      - Generates a question/answer pair
    """

    def __init__(self, tokenizer, proj_directions, embedder, fdm_encoder,
                 num_samples, max_length=MAX_LENGTH, held_out=False):
        self.tokenizer      = tokenizer
        self.proj           = proj_directions
        self.embedder       = embedder
        self.fdm_encoder    = fdm_encoder
        self.num_samples    = num_samples
        self.max_length     = max_length
        self.corpus         = HELD_OUT_CONCEPTS if held_out else FACT_CORPUS
        self.samples        = self._generate(num_samples)

    def _generate(self, n):
        samples = []
        for _ in range(n):
            # Pick a concept and assign to a channel
            slot, value = random.choice(self.corpus)
            channel = random.randint(0, NUM_CHANNELS - 1)

            # Get concept embedding
            with torch.no_grad():
                emb = self.embedder.embed(value)  # (hidden_dim,)

            # Compute amplitudes
            # Channel k: project concept embedding onto w_k
            # Other channels: small noise
            amps = np.random.uniform(-0.05, 0.05, NUM_CHANNELS)
            with torch.no_grad():
                dirs = F.normalize(self.proj.directions, dim=-1)  # (K, D)
                amp_k = float(torch.tanh(emb @ dirs[channel]))
            amps[channel] = amp_k

            # Encode to token IDs
            token_ids = self.fdm_encoder.encode(amps)

            # Decode token IDs to text
            fdm_text = self.tokenizer.decode(token_ids)

            # Question/answer
            question = f"What is the {slot} stored in channel {channel}?"
            answer   = value

            samples.append({
                "fdm_text":  fdm_text,
                "token_ids": token_ids,
                "channel":   channel,
                "slot":      slot,
                "value":     value,
                "amplitudes": amps.tolist(),
                "question":  question,
                "answer":    answer,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        full_text  = input_text + f" {s['answer']}"

        enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )

        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels         = input_ids.clone()

        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len]          = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "value":          s["value"],
            "channel":        s["channel"],
        }


# ============================================================
# Hook to capture layer-14 hidden states
# ============================================================

class HiddenStateCapture:
    """Registers a forward hook on a specific layer to capture its output."""

    def __init__(self):
        self.hidden = None
        self._hook  = None

    def register(self, layer):
        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) for transformer layers
            if isinstance(output, tuple):
                self.hidden = output[0]
            else:
                self.hidden = output
        self._hook = layer.register_forward_hook(hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()


# ============================================================
# Training stages
# ============================================================

def get_readout_layer(model):
    """Return the nn.Module for READOUT_LAYER (model-agnostic)."""
    # Works for most HuggingFace transformer models
    if hasattr(model, "transformer"):
        layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer layers. Adapt get_readout_layer().")
    return layers[READOUT_LAYER]


def train_stage(
    stage_idx, model, tokenizer, proj_directions, readout_probe,
    embedder, fdm_encoder, device
):
    cfg = STAGE_CONFIG[stage_idx]
    print(f"\n=== Stage {stage_idx} ===")
    print(f"  lr={cfg['lr']}, epochs={cfg['epochs']}, samples={cfg['samples']}")
    print(f"  freeze_proj={cfg['freeze_proj']}, freeze_lm={cfg['freeze_lm']}")

    # Freeze / unfreeze
    for p in model.parameters():
        p.requires_grad = not cfg["freeze_lm"]
    for p in proj_directions.parameters():
        p.requires_grad = not cfg["freeze_proj"]
    for p in readout_probe.parameters():
        p.requires_grad = True  # probe always trains

    # Dataset
    dataset = ContinuousAMDataset(
        tokenizer, proj_directions, embedder, fdm_encoder,
        num_samples=cfg["samples"]
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer over all trainable parameters
    # Move readout_probe to device before building optimizer
    readout_probe.to(device)

    trainable = (
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(filter(lambda p: p.requires_grad, proj_directions.parameters())) +
        list(readout_probe.parameters())
    )
    optimizer = AdamW(trainable, lr=cfg["lr"])

    # Hook to capture layer-14 hidden states
    capture = HiddenStateCapture()
    capture.register(get_readout_layer(model))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stage_dir = os.path.join(OUTPUT_DIR, f"stage{stage_idx}")
    os.makedirs(stage_dir, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        readout_probe.train()
        total_ce   = 0.0
        total_recon = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"S{stage_idx} E{epoch+1}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            values         = batch["value"]   # list of strings

            # Forward pass (capture hidden states at layer 14 via hook)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,  # we use the hook instead
            )
            ce_loss = outputs.loss

            # Reconstruction loss (only if we have hidden states)
            recon_loss = torch.tensor(0.0, device=device)
            if capture.hidden is not None:
                # Ensure probe is on same device as captured hidden states
                hidden = capture.hidden  # (batch, seq_len, hidden_dim)
                readout_probe.to(hidden.device)

                # Readout probe prediction — mean pool over FDM token positions
                pred_emb = readout_probe(hidden.float())  # (batch, embed_dim)

                # Target embeddings — no_grad since embedder is frozen reference
                with torch.no_grad():
                    target_emb = embedder.embed_batch(values).to(hidden.device).float()
                    target_emb = F.normalize(target_emb, dim=-1)  # unit norm for stable cosine loss

                # Cosine reconstruction loss
                pred_emb = F.normalize(pred_emb, dim=-1)  # unit norm

                # Contrastive loss: pred should be closer to its target
                # than to other targets in the batch
                # sim_matrix[i,j] = cosine sim of pred_i with target_j
                sim_matrix = pred_emb @ target_emb.T  # (batch, batch)
                # Diagonal is the correct pair
                batch_size = pred_emb.shape[0]
                labels = torch.arange(batch_size, device=pred_emb.device)
                # InfoNCE / NT-Xent loss with temperature 0.1
                temperature = 0.1
                recon_loss = F.cross_entropy(sim_matrix / temperature, labels)

                # Debug on first batch of first epoch
                if n_batches == 0 and epoch == 0:
                    print(f"  DEBUG recon: loss={recon_loss.item():.4f} "
                          f"pred_norm={pred_emb.norm().item():.3f} "
                          f"target_norm={target_emb.norm().item():.3f} "
                          f"hidden shape={hidden.shape} "
                          f"capture.hidden is None: False")
            else:
                if n_batches == 0 and epoch == 0:
                    print(f"  DEBUG: capture.hidden is None — hook not firing!")

            loss = ce_loss + LAMBDA_RECON * recon_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_ce    += ce_loss.item()
            total_recon += recon_loss.item()
            n_batches   += 1

        print(f"  Epoch {epoch+1}: CE={total_ce/n_batches:.4f}  Recon={total_recon/n_batches:.4f}")

        # Quick decode eval
        quick_eval(model, tokenizer, proj_directions, fdm_encoder, device, n=5)

    # Save
    capture.remove()
    model.save_pretrained(stage_dir)
    tokenizer.save_pretrained(stage_dir)
    torch.save(proj_directions.state_dict(), os.path.join(stage_dir, "proj_directions.pt"))
    torch.save(readout_probe.state_dict(),   os.path.join(stage_dir, "readout_probe.pt"))
    print(f"  Saved to {stage_dir}/")


# ============================================================
# Quick decode evaluation
# ============================================================

def quick_eval(model, tokenizer, proj_directions, fdm_encoder, device, n=5):
    model.eval()
    concepts = random.sample(FACT_CORPUS, min(n, len(FACT_CORPUS)))

    correct = 0
    for slot, value in concepts:
        channel = random.randint(0, NUM_CHANNELS - 1)

        # Encode
        with torch.no_grad():
            # Use a simple text embedding for eval (just tokenize and embed)
            inputs = tokenizer(value, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            emb = outputs.hidden_states[READOUT_LAYER][0].mean(0).float()

            dirs = F.normalize(proj_directions.directions, dim=-1)
            amp_k = float(torch.tanh(emb @ dirs[channel]))

        amps = np.random.uniform(-0.05, 0.05, NUM_CHANNELS)
        amps[channel] = amp_k
        token_ids = fdm_encoder.encode(amps)
        fdm_text  = tokenizer.decode(token_ids)

        question   = f"What is the {slot} stored in channel {channel}?"
        input_text = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        input_ids  = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        hit = value.lower() in response.lower()
        if hit:
            correct += 1
        print(f"    [{slot}] target={value!r}  got={response[:40]!r}  {'✓' if hit else '✗'}")

    print(f"    Quick eval: {correct}/{n}")


# ============================================================
# Generalization evaluation (Stage 4)
# ============================================================

def eval_generalization(model, tokenizer, proj_directions, fdm_encoder, device):
    """
    Tests held-out concepts never seen during training.
    This is the core empirical test of paper 2.
    """
    print("\n=== Stage 4: Generalization Evaluation ===")
    model.eval()

    results = {}
    for slot, value in HELD_OUT_CONCEPTS:
        channel = random.randint(0, NUM_CHANNELS - 1)

        # Encode using projection directions
        with torch.no_grad():
            inputs = tokenizer(value, return_tensors="pt").to(device)
            out    = model(**inputs, output_hidden_states=True)
            emb    = out.hidden_states[READOUT_LAYER][0].mean(0).float()
            dirs   = F.normalize(proj_directions.directions, dim=-1)
            amp_k  = float(torch.tanh(emb @ dirs[channel]))

        amps          = np.random.uniform(-0.05, 0.05, NUM_CHANNELS)
        amps[channel] = amp_k
        token_ids     = fdm_encoder.encode(amps)
        fdm_text      = tokenizer.decode(token_ids)

        question   = f"What is the {slot} stored in channel {channel}?"
        input_text = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        input_ids  = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(
            generated[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        hit = value.lower() in response.lower()
        results[(slot, value)] = {"response": response, "correct": hit}
        print(f"  [{slot}] target={value!r}  got={response[:50]!r}  {'✓' if hit else '✗'}")

    n_correct = sum(v["correct"] for v in results.values())
    print(f"\n  Generalization accuracy: {n_correct}/{len(HELD_OUT_CONCEPTS)}")
    print("  NOTE: This is the key result for paper 2.")
    print("        If >50%, the projection directions generalize beyond training vocab.")
    print("        If ~chance, the model memorized a lookup table, not a geometric mapping.")

    with open(os.path.join(OUTPUT_DIR, "generalization_results.json"), "w") as f:
        json.dump({f"{k[0]}:{k[1]}": v for k, v in results.items()}, f, indent=2)

    return results


# ============================================================
# Main
# ============================================================

def load_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MEMORY]", "[/MEMORY]"]})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer


def load_stage_checkpoint(stage_idx, model, tokenizer, proj_directions, readout_probe, device):
    stage_dir = os.path.join(OUTPUT_DIR, f"stage{stage_idx}")
    if not os.path.exists(stage_dir):
        return False

    from transformers import AutoModelForCausalLM as AMCLM
    model_new = AMCLM.from_pretrained(stage_dir, torch_dtype=DTYPE, trust_remote_code=True)
    model.load_state_dict(model_new.state_dict())
    model.to(device)

    proj_path  = os.path.join(stage_dir, "proj_directions.pt")
    probe_path = os.path.join(stage_dir, "readout_probe.pt")
    if os.path.exists(proj_path):
        proj_directions.load_state_dict(torch.load(proj_path, map_location=device))
    if os.path.exists(probe_path):
        readout_probe.load_state_dict(torch.load(probe_path, map_location=device))

    print(f"  Loaded checkpoint from {stage_dir}/")
    return True


def main():
    global LAMBDA_RECON
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument("--eval",  action="store_true")
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--lambda_recon", type=float, default=LAMBDA_RECON)
    args = parser.parse_args()

    LAMBDA_RECON = args.lambda_recon

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Model: {MODEL_ID}")

    model, tokenizer = load_model_and_tokenizer(device)

    # Infer embed_dim from model
    embed_dim = model.config.hidden_size

    proj_directions = ProjectionDirections(NUM_CHANNELS, embed_dim).to(device)
    readout_probe   = ReadoutProbe(embed_dim, embed_dim).to(device)
    embedder        = ConceptEmbedder(model, tokenizer, device)
    fdm_encoder     = ContinuousAMFDMEncoder(
        vocab_size=len(tokenizer),
        num_tokens=NUM_TOKENS_PER_ENCODER,
        num_levels=NUM_LEVELS,
    )

    if args.all:
        # Full sequential pipeline
        directions = stage0_extract_directions(model, tokenizer, device)
        proj_directions.load_from_mean_diff(directions.to(device))

        for s in [1, 2, 3]:
            train_stage(s, model, tokenizer, proj_directions, readout_probe,
                        embedder, fdm_encoder, device)

        eval_generalization(model, tokenizer, proj_directions, fdm_encoder, device)

    elif args.stage == 0:
        directions = stage0_extract_directions(model, tokenizer, device)
        proj_directions.load_from_mean_diff(directions.to(device))

    elif args.stage in [1, 2, 3]:
        # Load stage 0 directions if available
        dir_path = os.path.join(OUTPUT_DIR, "stage0_directions.pt")
        if os.path.exists(dir_path):
            d = torch.load(dir_path, map_location=device)
            proj_directions.load_from_mean_diff(d)
            print(f"  Loaded stage 0 directions from {dir_path}")

        # Load previous stage checkpoint if available
        # Stage 3 explicitly loads stage 1 (stage 2 was useless)
        if args.stage > 1:
            load_from = 1 if args.stage == 3 else args.stage - 1
            load_stage_checkpoint(
                load_from, model, tokenizer,
                proj_directions, readout_probe, device
            )

        train_stage(args.stage, model, tokenizer, proj_directions, readout_probe,
                    embedder, fdm_encoder, device)

    elif args.eval:
        # Load stage 3 checkpoint
        loaded = load_stage_checkpoint(3, model, tokenizer, proj_directions, readout_probe, device)
        if not loaded:
            print("No stage 3 checkpoint found. Run --stage 3 first.")
            sys.exit(1)
        eval_generalization(model, tokenizer, proj_directions, fdm_encoder, device)

    else:
        print("Usage:")
        print("  python fdm_continuous_am_train.py --stage 0   # extract directions")
        print("  python fdm_continuous_am_train.py --stage 1   # train decode")
        print("  python fdm_continuous_am_train.py --stage 2   # train encode")
        print("  python fdm_continuous_am_train.py --stage 3   # joint fine-tune")
        print("  python fdm_continuous_am_train.py --eval      # generalization eval")
        print("  python fdm_continuous_am_train.py --all       # run full pipeline")
        print()
        print("Key hyperparameter:")
        print("  --lambda_recon FLOAT   weight on reconstruction loss (default 0.1)")


if __name__ == "__main__":
    main()
