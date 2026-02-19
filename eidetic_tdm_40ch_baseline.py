"""
End-to-End TDM Baseline with 40 Channels - Curriculum Learning

40-channel TDM baseline for fair comparison with 40-channel FDM.

Encoding: Time-Division Multiplexing (codebook-biased GPT-2 generation)
  - Each token position carries ONE channel: position p carries channel p % 40
  - Dual codebooks + S-random interleaver (turbo structure)
  - 12 tokens per channel per encoder (max for GPT-2's 1024 context window)
  - 480 tokens per encoder × 2 = 960 total tokens
  - FDM uses 512 tokens for the same 40 channels → 1.9x density advantage

This is the SAME task, SAME channels, SAME token budget as the FDM experiment.
The ONLY difference is TDM encoding (one channel per position) vs FDM encoding
(all 40 channels superposed in every position via sinusoidal carriers).

Channel allocation (identical to FDM 40ch):
  0-5:   Facts (SECRET, LOCATION, AGENT, STATUS, PRIORITY, BACKUP)
  6:     RULE (SAFETY_FIRST, MISSION_FIRST, BALANCED, CAUTIOUS)
  7:     META (NONE, OVERRIDE_STATUS, OVERRIDE_PRIORITY, EMERGENCY, LOCKDOWN)
  8-19:  Extra context (TEAM, REGION, PHASE, COMM, ASSET, WINDOW, COVER, SUPPORT,
         THREAT, WEATHER, TERRAIN, EXTRACT)
  20-39: Extended context (CIPHER, FREQ, PAYLOAD, ROUTE, DURATION, CONTACT, FUEL,
         ALTITUDE, VISIBILITY, NOISE, FORMATION, ARMOR, SIGNAL, MORALE, SUPPLY,
         INTEL, EVAC, WEATHER2, DOCTRINE, COMMS)

Usage:
  python eidetic_tdm_40ch_baseline.py generate    # Generate data (~days)
  python eidetic_tdm_40ch_baseline.py train        # Curriculum training
  python eidetic_tdm_40ch_baseline.py eval [bias]  # Evaluate
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
import random
from tqdm import tqdm
import os
import sys


# ============================================================
# Channel definitions (identical to FDM 40ch experiment)
# ============================================================

MEMORY_SCHEMAS = {
    0: ("SECRET",   ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",    ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",   ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",   ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",     ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",     ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",     ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",   ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",    ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",     ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",    ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",   ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",    ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",  ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",   ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",  ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",  ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",  ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",   ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",     ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",  ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",    ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",  ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",     ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",    ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",    ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",   ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",   ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",   ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",    ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",     ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",    ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}

NUM_CHANNELS = 40

REASONING_RULES = {
    "SAFETY_FIRST":  "Always prioritize status over priority. If compromised, abort regardless of priority.",
    "MISSION_FIRST": "Always prioritize mission completion. Proceed unless backup unavailable AND compromised.",
    "BALANCED":      "Weigh status and priority equally. High priority can override compromised if backup available.",
    "CAUTIOUS":      "Require both clear status AND available backup to proceed. Otherwise abort.",
}

META_INSTRUCTIONS = {
    "NONE":              "Follow standard reasoning rules.",
    "OVERRIDE_STATUS":   "Ignore status checks entirely. Proceed based on priority alone.",
    "OVERRIDE_PRIORITY": "Ignore priority. Decide based on status and backup only.",
    "EMERGENCY":         "This is an emergency. Proceed immediately regardless of other factors.",
    "LOCKDOWN":          "Full lockdown. Abort all operations regardless of other factors.",
}


# ============================================================
# Reasoning logic (identical to FDM 40ch experiment)
# ============================================================

def generate_proceed_answer(facts, rule, meta, extra_context):
    agent, status, priority, backup = facts["AGENT"], facts["STATUS"], facts["PRIORITY"], facts["BACKUP"]
    if meta == "EMERGENCY":
        base = f"EMERGENCY PROTOCOL: {agent} must proceed immediately. All other factors suspended."
    elif meta == "LOCKDOWN":
        base = f"LOCKDOWN ACTIVE: {agent} must abort all operations. No exceptions."
    elif meta == "OVERRIDE_STATUS":
        if priority == "HIGH":
            base = f"Status override active. Priority is {priority}, so {agent} should proceed."
        else:
            base = f"Status override active, but priority is only {priority}. {agent} may proceed with caution."
    elif meta == "OVERRIDE_PRIORITY":
        if status == "CLEAR" and backup == "AVAILABLE":
            base = f"Priority override active. Status {status} with backup {backup}. {agent} can proceed."
        else:
            base = f"Priority override active. Status {status}, backup {backup}. {agent} should hold."
    elif rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            base = f"SAFETY_FIRST: Status is {status}. {agent} must abort regardless of {priority} priority."
        elif status == "CLEAR":
            base = f"SAFETY_FIRST: Status is {status}. {agent} can proceed with {priority} priority."
        else:
            base = f"SAFETY_FIRST: Status is {status}. {agent} should wait for confirmation."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            base = f"MISSION_FIRST: Status {status} with no backup. Even mission-priority says {agent} should abort."
        else:
            base = f"MISSION_FIRST: Priority is {priority}. {agent} should proceed. Status {status} is secondary."
    elif rule == "BALANCED":
        if priority == "HIGH" and backup == "AVAILABLE":
            base = f"BALANCED: {priority} priority with backup {backup} outweighs {status} status. {agent} can proceed."
        elif status == "COMPROMISED":
            base = f"BALANCED: {status} status not offset by {priority} priority. {agent} should abort."
        else:
            base = f"BALANCED: Status {status}, priority {priority}. {agent} can proceed carefully."
    elif rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            base = f"CAUTIOUS: Status {status} AND backup {backup}. Both conditions met. {agent} can proceed."
        else:
            base = f"CAUTIOUS: Need CLEAR status AND AVAILABLE backup. Have {status}/{backup}. {agent} must abort."
    else:
        base = f"{agent} should assess situation."
    
    # Append extra context (same format as FDM experiment)
    ctx_parts = [f"{k}={v}" for k, v in extra_context.items()]
    return base + " Context: " + ", ".join(ctx_parts) + "."


def generate_risk_answer(facts, rule, meta, extra_context):
    status, priority, backup, location = facts["STATUS"], facts["PRIORITY"], facts["BACKUP"], facts["LOCATION"]
    if meta == "EMERGENCY":
        base = f"EMERGENCY: Risk assessment suspended. Immediate action required in {location}."
    elif meta == "LOCKDOWN":
        base = f"LOCKDOWN: Maximum risk assumed. All operations in {location} halted."
    else:
        risk_factors = []
        if status == "COMPROMISED": risk_factors.append("compromised status")
        if backup == "UNAVAILABLE": risk_factors.append("no backup")
        if status == "UNKNOWN": risk_factors.append("unknown status")
        if rule == "SAFETY_FIRST":
            if risk_factors:
                base = f"SAFETY_FIRST assessment: HIGH RISK in {location}. Factors: {', '.join(risk_factors)}."
            else:
                base = f"SAFETY_FIRST assessment: LOW RISK in {location}. Status {status}, backup {backup}."
        elif rule == "MISSION_FIRST":
            if len(risk_factors) >= 2:
                base = f"MISSION_FIRST assessment: MODERATE RISK in {location}. Acceptable for {priority} priority."
            else:
                base = f"MISSION_FIRST assessment: LOW RISK in {location}. Proceed with mission."
        elif rule == "BALANCED":
            level = "HIGH" if len(risk_factors) >= 2 else "MEDIUM" if len(risk_factors) == 1 else "LOW"
            base = f"BALANCED assessment: {level} RISK in {location}. Factors: {len(risk_factors)} concerns."
        elif rule == "CAUTIOUS":
            if risk_factors:
                base = f"CAUTIOUS assessment: HIGH RISK in {location}. Any risk factor triggers alert: {', '.join(risk_factors)}."
            else:
                base = f"CAUTIOUS assessment: LOW RISK in {location}. All safety conditions met."
        else:
            base = f"Risk assessment for {location}."
    
    ctx_parts = [f"{k}={v}" for k, v in extra_context.items()]
    return base + " Context: " + ", ".join(ctx_parts) + "."


def generate_share_answer(facts, rule, meta, extra_context):
    secret, status, location = facts["SECRET"], facts["STATUS"], facts["LOCATION"]
    if meta == "EMERGENCY":
        base = f"EMERGENCY: Share {secret} immediately with any allied contact. Speed over security."
    elif meta == "LOCKDOWN":
        base = f"LOCKDOWN: Do not share {secret} under any circumstances. Maintain radio silence."
    elif meta == "OVERRIDE_STATUS":
        base = f"Status override active. You may share {secret} with {location} contact despite {status} status."
    elif rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            base = f"SAFETY_FIRST: Do NOT share {secret}. {location} may be compromised."
        else:
            base = f"SAFETY_FIRST: Status {status}. May share {secret} with verified contacts only."
    elif rule == "MISSION_FIRST":
        base = f"MISSION_FIRST: Share {secret} with {location} contact to advance mission. Accept calculated risk."
    elif rule == "BALANCED":
        if status == "CLEAR":
            base = f"BALANCED: Status {status}. Safe to share {secret} with {location} contact."
        else:
            base = f"BALANCED: Status {status}. Share {secret} only if mission-critical."
    elif rule == "CAUTIOUS":
        if status == "CLEAR":
            base = f"CAUTIOUS: Status {status}. May share {secret} after secondary verification."
        else:
            base = f"CAUTIOUS: Status {status}. Do not share {secret}. Request secure channel."
    else:
        base = f"Evaluate sharing {secret} in {location}."
    
    ctx_parts = [f"{k}={v}" for k, v in extra_context.items()]
    return base + " Context: " + ", ".join(ctx_parts) + "."


QUESTION_TEMPLATES = [
    {"question": "Should {AGENT} proceed with the mission?", "fn": generate_proceed_answer},
    {"question": "What is the risk assessment?", "fn": generate_risk_answer},
    {"question": "Is it safe to share the secret with the contact?", "fn": generate_share_answer},
]


# ============================================================
# Turbo TDM Encoder (codebook-biased, NOT FDM)
# ============================================================

def generate_s_random_interleaver(length, S=None):
    if S is None:
        S = int(np.sqrt(length / 2))
    random.seed(42)
    interleaver = list(range(length))
    for i in range(length):
        for _ in range(100):
            j = random.randint(i, length - 1)
            valid = True
            for k in range(max(0, i - S + 1), i):
                if abs(interleaver[j] - interleaver[k]) < S:
                    valid = False
                    break
            if valid:
                interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                break
    return interleaver


class TurboTDMEncoder:
    """
    Turbo TDM encoder for 40 channels.
    
    Time-Division: position p carries channel (p % NUM_CHANNELS).
    Each channel gets tokens_per_channel positions per encoder.
    Dual codebooks + S-random interleaver for turbo structure.
    
    With 40 channels and tokens_per_channel=12:
      Encoder 1: 480 tokens (12 per channel)
      Encoder 2: 480 tokens (S-random interleaved)
      Total: 960 tokens
    
    This is the maximum that fits in GPT-2's 1,024 context window.
    FDM encodes the same 40 channels in 512 tokens (1.9x fewer).
    The 8ch TDM baseline used 16 tokens/channel — GPT-2's context
    window prevents matching that at 40 channels, which is itself
    evidence of FDM's density advantage.
    """

    def __init__(self, bias_strength=10.0, tokens_per_channel=12):
        self.bias_strength = bias_strength
        self.num_channels = NUM_CHANNELS
        self.tokens_per_channel = tokens_per_channel
        self.encoder1_tokens = self.num_channels * tokens_per_channel  # 240
        self.encoder2_tokens = self.num_channels * tokens_per_channel  # 240
        self.total_tokens = self.encoder1_tokens + self.encoder2_tokens  # 480

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)

        self.interleaver = generate_s_random_interleaver(self.encoder1_tokens, S=int(np.sqrt(self.encoder1_tokens / 2)))
        self._generate_codebooks()

        print(f"Turbo TDM 40ch Encoder:")
        print(f"  Channels: {self.num_channels}")
        print(f"  Tokens per channel per encoder: {self.tokens_per_channel}")
        print(f"  Encoder 1 tokens: {self.encoder1_tokens}")
        print(f"  Encoder 2 tokens: {self.encoder2_tokens} (S-random interleaved)")
        print(f"  Total tokens: {self.total_tokens}")

    def _generate_codebooks(self):
        self.codebook1, self.codebook2 = {}, {}
        for ch, (name, values) in MEMORY_SCHEMAS.items():
            for val in values:
                seed = hash(f"{name}:{val}") % (2**32)
                rng = np.random.RandomState(seed)
                tokens1 = set(rng.choice(self.vocab_size, size=200, replace=False))
                overlap = set(list(tokens1)[:150])
                new_tokens = set(rng.choice(self.vocab_size, size=50, replace=False))
                tokens2 = overlap | new_tokens
                self.codebook1[(ch, val)] = tokens1
                self.codebook2[(ch, val)] = tokens2

    def _get_bias(self, channel_id, value, encoder_num):
        codebook = self.codebook1 if encoder_num == 1 else self.codebook2
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        for t in codebook[(channel_id, value)]:
            bias[t] = self.bias_strength
        return bias

    def encode_memory(self, memory: dict) -> str:
        input_ids = self.tokenizer.encode("Memory:", return_tensors='pt').to(self.device)
        generated_tokens = []
        with torch.no_grad():
            for pos in range(self.total_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                if pos < self.encoder1_tokens:
                    # TDM: position determines channel
                    ch = pos % self.num_channels
                    val = memory.get(ch, MEMORY_SCHEMAS[ch][1][0])
                    bias = self._get_bias(ch, val, 1)
                else:
                    # Encoder 2: S-random interleaved positions
                    orig = self.interleaver[pos - self.encoder1_tokens]
                    ch = orig % self.num_channels
                    val = memory.get(ch, MEMORY_SCHEMAS[ch][1][0])
                    bias = self._get_bias(ch, val, 2)
                logits = logits + bias
                logits = logits - logits.max()
                probs = np.exp(logits) / np.exp(logits).sum()
                token = np.random.choice(self.vocab_size, p=probs)
                generated_tokens.append(token)
                input_ids = torch.cat([input_ids, torch.tensor([[token]], device=self.device)], dim=1)
        return self.tokenizer.decode(generated_tokens)


# ============================================================
# Dataset
# ============================================================

class EideticReasoningDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_text = f"[MEMORY]{s['tdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        target_text = f" {s['answer']}"
        full_text = input_text + target_text

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# Data generation
# ============================================================

def generate_stage_data(bias_strength, num_samples, output_prefix):
    encoder = TurboTDMEncoder(bias_strength=bias_strength)
    samples = []

    print(f"Generating {num_samples} TDM samples at bias={bias_strength}...")

    for _ in tqdm(range(num_samples)):
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)

        tdm_text = encoder.encode_memory(memory)

        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        rule = memory[6]
        meta = memory[7]
        
        # Extra context (channels 8-39)
        extra_context = {}
        for ch in range(8, NUM_CHANNELS):
            name = MEMORY_SCHEMAS[ch][0]
            extra_context[name] = memory[ch]

        q_template = random.choice(QUESTION_TEMPLATES)
        question = q_template["question"].format(**facts)
        answer = q_template["fn"](facts, rule, meta, extra_context)

        explicit_parts = [f"{MEMORY_SCHEMAS[ch][0]}:{memory[ch]}" for ch in range(NUM_CHANNELS)]
        explicit = "|".join(explicit_parts)

        samples.append({
            "tdm_text": tdm_text,
            "memory": {str(k): v for k, v in memory.items()},
            "explicit": explicit,
            "question": question,
            "answer": answer,
            "rule": rule,
            "meta": meta,
            "facts": facts,
            "extra_context": extra_context,
            "bias_strength": bias_strength,
        })

    random.shuffle(samples)
    n = len(samples)
    splits = {
        'train': samples[:int(0.85 * n)],
        'val': samples[int(0.85 * n):int(0.95 * n)],
        'test': samples[int(0.95 * n):],
    }

    for split_name, split_data in splits.items():
        path = f"{output_prefix}_{split_name}.jsonl"
        with open(path, 'w') as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split_name}: {len(split_data)} samples -> {path}")

    return samples


def generate_all_stages():
    """Generate data for all curriculum stages. Scaled up for 40 channels."""
    stages = [
        {"bias": 20.0, "samples": 10000},
        {"bias": 15.0, "samples": 15000},
        {"bias": 10.0, "samples": 25000},
        {"bias": 7.0,  "samples": 30000},
        {"bias": 5.0,  "samples": 35000},
    ]

    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE {i}: bias={stage['bias']}, samples={stage['samples']}")
        print(f"{'='*60}")
        generate_stage_data(stage['bias'], stage['samples'], f"tdm_40ch_stage{i}")


# ============================================================
# Training
# ============================================================

def train(checkpoint_dir="checkpoints_tdm_40ch"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    stages = [
        {"bias": 20.0, "epochs": 5,  "lr": 5e-5, "prefix": "tdm_40ch_stage0"},
        {"bias": 15.0, "epochs": 5,  "lr": 3e-5, "prefix": "tdm_40ch_stage1"},
        {"bias": 10.0, "epochs": 8,  "lr": 2e-5, "prefix": "tdm_40ch_stage2"},
        {"bias": 7.0,  "epochs": 8,  "lr": 1e-5, "prefix": "tdm_40ch_stage3"},
        {"bias": 5.0,  "epochs": 8,  "lr": 1e-5, "prefix": "tdm_40ch_stage4"},
    ]

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[MEMORY]', '[/MEMORY]']
    })

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Check for resume
    resume_stage = 0
    resume_epoch = 0
    ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f"Found checkpoint: {latest}")
        ckpt = torch.load(os.path.join(checkpoint_dir, latest))
        model.load_state_dict(ckpt['model_state_dict'])
        resume_stage = ckpt.get('stage', 0)
        resume_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resuming from stage {resume_stage}, epoch {resume_epoch}")

    for stage_idx, stage in enumerate(stages):
        if stage_idx < resume_stage:
            print(f"Skipping stage {stage_idx} (already completed)")
            continue

        prefix = stage['prefix']
        train_path = f"{prefix}_train.jsonl"
        val_path = f"{prefix}_val.jsonl"

        if not os.path.exists(train_path):
            print(f"ERROR: {train_path} not found. Run 'generate' first.")
            return

        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: bias={stage['bias']}, lr={stage['lr']}, epochs={stage['epochs']}")
        print(f"{'='*60}")

        train_dataset = EideticReasoningDataset(train_path, tokenizer)
        val_dataset = EideticReasoningDataset(val_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)

        optimizer = AdamW(model.parameters(), lr=stage['lr'])
        best_val_loss = float('inf')

        start_epoch = resume_epoch if stage_idx == resume_stage else 0
        resume_epoch = 0

        for epoch in range(start_epoch, stage['epochs']):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"S{stage_idx} E{epoch+1} Train"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)

            print(f"  Stage {stage_idx} Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            if (epoch + 1) % 2 == 0 or epoch == stage['epochs'] - 1:
                quick_eval(model, tokenizer, val_path, device, num_samples=5, stage_idx=stage_idx)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best val loss: {val_loss:.4f}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'stage': stage_idx,
                'epoch': epoch,
                'val_loss': val_loss,
                'bias': stage['bias'],
            }, os.path.join(checkpoint_dir, f'stage{stage_idx}_epoch{epoch+1}.pt'))

        model.save_pretrained(f'tdm_40ch_model_stage{stage_idx}')
        tokenizer.save_pretrained(f'tdm_40ch_model_stage{stage_idx}')
        print(f"Saved stage {stage_idx} model")

    model.save_pretrained('tdm_40ch_model_final')
    tokenizer.save_pretrained('tdm_40ch_model_final')
    print("\nSaved final model: tdm_40ch_model_final/")


def quick_eval(model, tokenizer, val_path, device, num_samples=5, stage_idx=0):
    samples = [json.loads(l) for l in open(val_path)][:num_samples]
    model.eval()
    correct_action = 0
    correct_rule = 0
    total = 0

    for s in samples:
        input_text = f"[MEMORY]{s['tdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=150, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0].strip()
        expected = s['answer']

        for kw in ['abort', 'proceed', 'hold', 'wait', 'share', 'Do NOT share']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                correct_action += 1
                break

        if s['rule'] in response:
            correct_rule += 1
        total += 1

    print(f"    Quick eval (S{stage_idx}): action={correct_action}/{total}, rule={correct_rule}/{total}")


# ============================================================
# Full evaluation
# ============================================================

def evaluate(model_path='tdm_40ch_model_final', test_bias=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {model_path}/...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    if test_bias is not None:
        test_path = f"tdm_40ch_eval_bias{int(test_bias)}_test.jsonl"
        if not os.path.exists(test_path):
            print(f"Generating test data at bias={test_bias}...")
            generate_stage_data(test_bias, 1250, f"tdm_40ch_eval_bias{int(test_bias)}")
    else:
        test_path = "tdm_40ch_stage2_test.jsonl"

    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found. Run 'generate' first.")
        return

    test_samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating on {len(test_samples)} samples...\n")

    action_correct = 0
    rule_correct = 0
    meta_correct = 0
    fact_correct = 0
    extra_correct = 0
    extra_total = 0
    total = 0

    rule_stats = {r: {"correct": 0, "total": 0} for r in REASONING_RULES}
    meta_stats = {m: {"correct": 0, "total": 0} for m in META_INSTRUCTIONS}

    for i, s in enumerate(tqdm(test_samples, desc="Evaluating")):
        input_text = f"[MEMORY]{s['tdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=150, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0].strip()
        expected = s['answer']
        rule = s['rule']
        meta = s['meta']

        # Action correctness
        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok:
            action_correct += 1

        # Rule correctness
        rule_ok = (rule in response)
        if rule_ok:
            rule_correct += 1
        rule_stats[rule]["total"] += 1
        if rule_ok:
            rule_stats[rule]["correct"] += 1

        # Meta correctness
        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response:
            meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response:
            meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]:
            meta_ok = True
        if meta_ok:
            meta_correct += 1
        meta_stats[meta]["total"] += 1
        if meta_ok:
            meta_stats[meta]["correct"] += 1

        # Fact correctness
        fact_ok = False
        facts = s['facts']
        if s['question'].startswith("Should"):
            fact_ok = facts['AGENT'] in response
        elif s['question'].startswith("What is the risk"):
            fact_ok = facts['LOCATION'] in response
        elif s['question'].startswith("Is it safe"):
            fact_ok = facts['SECRET'] in response or facts['LOCATION'] in response
        if fact_ok:
            fact_correct += 1

        # Extra channel correctness (channels 8-39)
        extra_ctx = s.get('extra_context', {})
        for k, v in extra_ctx.items():
            extra_total += 1
            if f"{k}={v}" in response:
                extra_correct += 1

        total += 1

        if i < 15:
            print(f"\n{'='*60}")
            print(f"Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"Memory: {s['explicit']}")
            print(f"Q: {s['question']}")
            print(f"Expected: {expected[:200]}...")
            print(f"Got:      {response[:200]}...")
            action_mark = "✓" if action_ok else "✗"
            rule_mark = "✓" if rule_ok else "✗"
            fact_mark = "✓" if fact_ok else "✗"
            print(f"Action: {action_mark} | Rule: {rule_mark} | Fact: {fact_mark}")

    print(f"\n{'='*60}")
    print(f"40-CHANNEL TDM BASELINE EVALUATION")
    print(f"{'='*60}")

    print(f"\n--- Overall ---")
    print(f"  Action correct: {100*action_correct/total:.1f}%")
    print(f"  Rule correct:   {100*rule_correct/total:.1f}%")
    print(f"  Meta correct:   {100*meta_correct/total:.1f}%")
    print(f"  Fact correct (ch 0-5): {100*fact_correct/total:.1f}%")
    if extra_total > 0:
        print(f"  Extra correct (ch 8-39): {100*extra_correct/extra_total:.1f}% ({extra_correct}/{extra_total})")

    print(f"\n--- Per Rule ---")
    for r, stats in rule_stats.items():
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {r:15s}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\n--- Per Meta ---")
    for m, stats in meta_stats.items():
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {m:20s}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\n--- Comparison ---")
    print(f"  TDM 8ch (codebook):              Action=93.5%, Fact=99.0%")
    print(f"  FDM 8ch v1 (256t, 64l):          Action=80.0%, Fact=88.0%")
    print(f"  FDM 20ch single (256t, 64l):     Action=63.0%, Fact=70.5%, Extra=36.8%")
    print(f"  FDM 20ch TURBO (256+256t, 64l):  Action=72.0%, Fact=88.5%, Extra=48.8%")
    print(f"  FDM 40ch TURBO v1 (19K samples): Action=77.0%, Fact=81.5%, Extra=44.3%")
    print(f"  FDM 40ch TURBO v2 (46K samples): Action=94.2%, Fact=98.0%, Extra=63.4%")
    print(f"  FDM 40ch TURBO v3 (115K samples):Action=98.3%, Fact=99.9%, Extra=89.2%")
    print(f"  TDM 40ch (this run):             Action={100*action_correct/total:.1f}%, Fact={100*fact_correct/total:.1f}%", end="")
    if extra_total > 0:
        print(f", Extra={100*extra_correct/extra_total:.1f}%")
    else:
        print()

    print(f"\n--- Configuration ---")
    print(f"  Channels: {NUM_CHANNELS}")
    print(f"  Encoding: TDM (codebook-biased, one channel per position)")
    print(f"  Tokens: 480 per encoder x 2 = 960 total (S-random interleaved)")
    print(f"  Each channel: 12 token positions per encoder (max for GPT-2's 1024 context)")
    print(f"  FDM comparison: 512 tokens total (1.9x fewer than this TDM baseline)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eidetic_tdm_40ch_baseline.py generate    # Generate data for all stages")
        print("  python eidetic_tdm_40ch_baseline.py train       # Curriculum training")
        print("  python eidetic_tdm_40ch_baseline.py eval [bias] # Evaluate")
        print()
        print("Full pipeline:")
        print("  1. generate  (115K samples across 5 stages)")
        print("  2. train     (34 epochs across 5 stages)")
        print("  3. eval      (minutes)")
        print()
        print("This is the TDM BASELINE for comparison with FDM 40ch.")
        print("Same channels, same task. TDM uses 960 tokens (12/ch × 40ch × 2 encoders),")
        print("the maximum that fits GPT-2's 1024 context window.")
        print("FDM uses 512 tokens for the same 40 channels (1.9x fewer).")
        print("GPT-2's context limit itself demonstrates FDM's density advantage.")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "generate":
        generate_all_stages()
    elif cmd == "train":
        train()
    elif cmd == "eval":
        bias = float(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate(test_bias=bias)
