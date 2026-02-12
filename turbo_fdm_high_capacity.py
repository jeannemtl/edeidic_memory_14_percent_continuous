"""
High-capacity Turbo FDM encoder.

Goal: Maximize bits per token while maintaining decodability.

Changes from v2:
- 16 channels (up from 8)
- 16 values per channel (4 bits each, up from 2-3)
- 8 tokens per channel per encoder (down from 16)
- Total: 64 bits in 256 tokens = 0.25 bits/token

If this works, we can push further.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import random
from tqdm import tqdm

# Expanded schema: 16 channels, 16 values each = 4 bits per channel
NUM_CHANNELS = 16
VALUES_PER_CHANNEL = 16

MEMORY_SCHEMAS = {
    ch: (f"CH{ch:02d}", [f"V{ch:02d}_{v:02d}" for v in range(VALUES_PER_CHANNEL)])
    for ch in range(NUM_CHANNELS)
}


def generate_s_random_interleaver(length, S=None):
    """Generate S-random interleaver."""
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


class HighCapacityTurboEncoder:
    """
    High-capacity turbo encoder.
    
    16 channels × 4 bits = 64 bits total
    8 tokens per channel × 2 encoders = 256 tokens
    Efficiency: 0.25 bits/token (baseline to improve)
    """
    
    def __init__(self, bias_strength=15.0, tokens_per_channel=8):
        self.bias_strength = bias_strength
        self.num_channels = NUM_CHANNELS
        self.tokens_per_channel = tokens_per_channel
        
        self.encoder1_tokens = self.num_channels * tokens_per_channel  # 128
        self.encoder2_tokens = self.num_channels * tokens_per_channel  # 128
        self.total_tokens = self.encoder1_tokens + self.encoder2_tokens  # 256
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)
        
        self.interleaver = generate_s_random_interleaver(self.encoder1_tokens, S=8)
        self._generate_codebooks()
        
        print(f"High-Capacity Turbo Encoder:")
        print(f"  Channels: {self.num_channels}")
        print(f"  Values per channel: {VALUES_PER_CHANNEL} ({np.log2(VALUES_PER_CHANNEL):.1f} bits)")
        print(f"  Total bits: {self.num_channels * np.log2(VALUES_PER_CHANNEL):.0f}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Efficiency: {self.num_channels * np.log2(VALUES_PER_CHANNEL) / self.total_tokens:.3f} bits/token")
    
    def _generate_codebooks(self):
        """Generate deterministic codebooks for each (channel, value) pair."""
        self.codebook1 = {}
        self.codebook2 = {}
        
        for ch in range(self.num_channels):
            for v in range(VALUES_PER_CHANNEL):
                seed = hash(f"ch{ch}_v{v}") % (2**32)
                rng = np.random.RandomState(seed)
                
                # Codebook 1: 150 tokens
                tokens1 = set(rng.choice(self.vocab_size, size=150, replace=False))
                
                # Codebook 2: 100 overlapping + 50 new
                overlap = set(list(tokens1)[:100])
                new_tokens = set(rng.choice(self.vocab_size, size=50, replace=False))
                tokens2 = overlap | new_tokens
                
                self.codebook1[(ch, v)] = tokens1
                self.codebook2[(ch, v)] = tokens2
    
    def _get_bias(self, channel_id, value_idx, encoder_num):
        """Get bias vector."""
        codebook = self.codebook1 if encoder_num == 1 else self.codebook2
        boosted_tokens = codebook[(channel_id, value_idx)]
        
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        for t in boosted_tokens:
            bias[t] = self.bias_strength
        
        return bias
    
    def _softmax(self, x):
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
    
    def encode_memory(self, memory: dict) -> str:
        """Encode memory dict to turbo FDM tokens."""
        input_ids = self.tokenizer.encode("Data:", return_tensors='pt').to(self.device)
        generated_tokens = []
        
        with torch.no_grad():
            for pos in range(self.total_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                if pos < self.encoder1_tokens:
                    # Encoder 1
                    channel_id = pos % self.num_channels
                    value_idx = memory.get(channel_id, 0)
                    bias = self._get_bias(channel_id, value_idx, encoder_num=1)
                    logits = logits + bias
                else:
                    # Encoder 2 (interleaved)
                    interleaved_pos = pos - self.encoder1_tokens
                    original_pos = self.interleaver[interleaved_pos]
                    channel_id = original_pos % self.num_channels
                    value_idx = memory.get(channel_id, 0)
                    bias = self._get_bias(channel_id, value_idx, encoder_num=2)
                    logits = logits + bias
                
                probs = self._softmax(logits)
                token = np.random.choice(self.vocab_size, p=probs)
                
                generated_tokens.append(token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
        
        return self.tokenizer.decode(generated_tokens)


def generate_high_capacity_dataset(num_samples=20000, output_path="turbo_high_capacity_training.jsonl"):
    """Generate high-capacity training data."""
    
    encoder = HighCapacityTurboEncoder(bias_strength=15.0)
    samples = []
    
    print(f"\nGenerating {num_samples} high-capacity samples...")
    
    for _ in tqdm(range(num_samples)):
        # Random memory: 16 channels, each with value 0-15
        memory = {ch: random.randint(0, VALUES_PER_CHANNEL - 1) for ch in range(NUM_CHANNELS)}
        
        # Encode
        fdm_text = encoder.encode_memory(memory)
        
        samples.append({
            "fdm_text": fdm_text,
            "memory": memory,
        })
    
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


if __name__ == "__main__":
    generate_high_capacity_dataset(num_samples=20000)
