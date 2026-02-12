"""
Coherent Turbo FDM

Combines:
1. Coherent text generation (readable)
2. Turbo structure (dual encoder, interleaving, deterministic codebooks)

Goal: Readable text with extractable hidden channels.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import random
from tqdm import tqdm

CHANNELS = {
    0: ("THREAT", ["minimal", "low", "moderate", "elevated", "high", "severe", "critical", "extreme"]),
    1: ("STATUS", ["offline", "degraded", "limited", "operational", "stable", "optimal", "enhanced", "peak"]),
    2: ("PRIORITY", ["routine", "low", "normal", "elevated", "high", "urgent", "critical", "emergency"]),
    3: ("TIMELINE", ["indefinite", "extended", "long", "medium", "short", "immediate", "urgent", "now"]),
}

NUM_CHANNELS = len(CHANNELS)
VALUES_PER_CHANNEL = 8
TOKENS_PER_CHANNEL = 8  # Per encoder
BIAS_STRENGTH = 6.0  # Stronger than soft (3.0), weaker than gibberish (15.0)


def generate_s_random_interleaver(length, S=None):
    """S-random interleaver for turbo structure."""
    if S is None:
        S = int(np.sqrt(length / 2))
    
    random.seed(42)  # Deterministic
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


class CoherentTurboEncoder:
    """
    Turbo encoder that produces coherent text.
    
    - Dual encoders (redundancy)
    - S-random interleaving (error spreading)
    - Deterministic codebooks (learnable patterns)
    - Topic prompts (coherence)
    """
    
    def __init__(self, bias_strength=BIAS_STRENGTH):
        self.bias_strength = bias_strength
        self.num_channels = NUM_CHANNELS
        self.tokens_per_channel = TOKENS_PER_CHANNEL
        
        self.encoder1_tokens = self.num_channels * self.tokens_per_channel  # 32
        self.encoder2_tokens = self.num_channels * self.tokens_per_channel  # 32
        self.total_tokens = self.encoder1_tokens + self.encoder2_tokens     # 64
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)
        
        # S-random interleaver
        self.interleaver = generate_s_random_interleaver(self.encoder1_tokens, S=4)
        
        # Deterministic codebooks (like turbo FDM)
        self._build_codebooks()
        
        print(f"Coherent Turbo Encoder:")
        print(f"  Channels: {self.num_channels}")
        print(f"  Values per channel: {VALUES_PER_CHANNEL}")
        print(f"  Tokens per encoder: {self.encoder1_tokens}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Bias strength: {self.bias_strength}")
    
    def _build_codebooks(self):
        """Build dual codebooks with overlap (turbo structure)."""
        self.codebook1 = {}
        self.codebook2 = {}
        
        for ch_id in range(self.num_channels):
            ch_name = CHANNELS[ch_id][0]
            for val_idx in range(VALUES_PER_CHANNEL):
                seed = hash(f"{ch_name}:{val_idx}") % (2**32)
                rng = np.random.RandomState(seed)
                
                # Codebook 1: 150 tokens
                tokens1 = set(rng.choice(self.vocab_size, size=150, replace=False))
                
                # Codebook 2: 100 overlapping + 50 new (correlation for iterative decoding)
                overlap = set(list(tokens1)[:100])
                new_tokens = set(rng.choice(self.vocab_size, size=50, replace=False))
                tokens2 = overlap | new_tokens
                
                self.codebook1[(ch_id, val_idx)] = tokens1
                self.codebook2[(ch_id, val_idx)] = tokens2
    
    def _get_bias(self, channel_id, value_idx, encoder_num):
        """Get bias vector for channel/value/encoder."""
        codebook = self.codebook1 if encoder_num == 1 else self.codebook2
        tokens = codebook[(channel_id, value_idx)]
        
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        for t in tokens:
            bias[t] = self.bias_strength
        
        return bias
    
    def encode(self, memory, prompt):
        """Generate coherent turbo-encoded text."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = []
        
        with torch.no_grad():
            for pos in range(self.total_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                # Determine channel and encoder
                if pos < self.encoder1_tokens:
                    # Encoder 1: direct positions
                    channel_id = pos % self.num_channels
                    value_idx = memory[channel_id]
                    bias = self._get_bias(channel_id, value_idx, encoder_num=1)
                else:
                    # Encoder 2: interleaved positions
                    interleaved_pos = pos - self.encoder1_tokens
                    original_pos = self.interleaver[interleaved_pos]
                    channel_id = original_pos % self.num_channels
                    value_idx = memory[channel_id]
                    bias = self._get_bias(channel_id, value_idx, encoder_num=2)
                
                # Apply bias
                logits = logits + bias
                
                # Temperature
                logits = logits / 0.85
                
                # Softmax + sample
                logits = logits - logits.max()
                probs = np.exp(logits) / np.exp(logits).sum()
                token = np.random.choice(self.vocab_size, p=probs)
                
                generated.append(token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
        
        return prompt + self.tokenizer.decode(generated)


PROMPTS = [
    "Situation report:",
    "Field assessment:",
    "Current status update:",
    "Intelligence summary:",
    "Operational briefing:",
    "Tactical overview:",
    "Mission status:",
    "Sector analysis:",
]


def generate_target(memory):
    """Generate target response based on memory."""
    threat = CHANNELS[0][1][memory[0]]
    status = CHANNELS[1][1][memory[1]]
    priority = CHANNELS[2][1][memory[2]]
    timeline = CHANNELS[3][1][memory[3]]
    
    response = f"Extracted: THREAT={threat}, STATUS={status}, PRIORITY={priority}, TIMELINE={timeline}. "
    
    if memory[0] >= 6:
        response += "CRITICAL threat level. Immediate response required. "
    elif memory[0] >= 4:
        response += "Elevated threat. Increased vigilance recommended. "
    else:
        response += "Threat level acceptable. "
    
    if memory[2] >= 6 and memory[3] >= 5:
        response += "High priority with urgent timeline - expedite action."
    elif memory[1] >= 5:
        response += f"Status {status} - conditions favorable."
    
    return response


def generate_dataset(num_samples=10000, output_path="coherent_turbo_fdm.jsonl"):
    """Generate coherent turbo FDM dataset."""
    
    encoder = CoherentTurboEncoder(bias_strength=BIAS_STRENGTH)
    samples = []
    
    print(f"\nGenerating {num_samples} coherent turbo samples...")
    
    for _ in tqdm(range(num_samples)):
        memory = {ch: random.randint(0, 7) for ch in range(NUM_CHANNELS)}
        prompt = random.choice(PROMPTS)
        
        fdm_text = encoder.encode(memory, prompt)
        target = generate_target(memory)
        
        samples.append({
            "fdm_text": fdm_text,
            "target": target,
            "memory": memory,
        })
    
    # Save
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    
    # Splits
    random.shuffle(samples)
    n = len(samples)
    for name, start, end in [('train', 0, 0.8), ('val', 0.8, 0.9), ('test', 0.9, 1.0)]:
        split = samples[int(n*start):int(n*end)]
        path = output_path.replace(".jsonl", f"_{name}.jsonl")
        with open(path, 'w') as f:
            for s in split:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(split)} samples")
    
    # Examples
    print("\n=== Examples ===\n")
    for s in samples[:3]:
        print(f"Memory: {s['memory']}")
        print(f"FDM Text: {s['fdm_text'][:150]}...")
        print(f"Target: {s['target']}")
        print()


if __name__ == "__main__":
    generate_dataset(num_samples=10000)
