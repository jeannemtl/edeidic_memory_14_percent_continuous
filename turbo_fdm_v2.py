"""
Turbo-code inspired FDM encoding for approaching Shannon limit.

Key principles:
1. Two parallel encoders with different bias patterns
2. S-random interleaver ensures distant mapping
3. Structured redundancy that's learnable, not random noise
4. Designed for iterative decoding
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import random
from tqdm import tqdm

MEMORY_SCHEMAS = {
    0: ("SECRET", ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT", ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS", ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP", ["AVAILABLE", "UNAVAILABLE"]),
    6: ("TIMELINE", ["URGENT", "FLEXIBLE"]),
    7: ("MODE", ["AUTO", "MANUAL", "STANDBY"]),
}


def generate_s_random_interleaver(length, S=None):
    """
    Generate S-random interleaver.
    
    Ensures that if |i - j| < S, then |π(i) - π(j)| >= S.
    This spreads adjacent positions far apart in output.
    """
    if S is None:
        S = int(np.sqrt(length / 2))  # Typical choice
    
    interleaver = list(range(length))
    
    for i in range(length):
        # Try to find a valid swap
        for _ in range(100):  # Max attempts
            j = random.randint(i, length - 1)
            
            # Check S-random constraint
            valid = True
            for k in range(max(0, i - S + 1), i):
                if abs(interleaver[j] - interleaver[k]) < S:
                    valid = False
                    break
            
            if valid:
                interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                break
    
    return interleaver


class TurboFDMEncoder:
    """
    Turbo-code inspired FDM encoder.
    
    Structure:
    - Encoder 1: Direct encoding at positions [0, N)
    - Interleaver: S-random permutation
    - Encoder 2: Interleaved encoding at positions [N, 2N)
    
    Both encoders encode the same channel values but with:
    - Different positions (spread by interleaver)
    - Different (but related) bias patterns
    """
    
    def __init__(self, bias_strength=10.0, num_channels=8, tokens_per_channel=16):
        self.bias_strength = bias_strength
        self.num_channels = num_channels
        self.tokens_per_channel = tokens_per_channel
        
        # Total tokens = 2 * (channels * tokens_per_channel) for two encoders
        self.encoder1_tokens = num_channels * tokens_per_channel  # 128
        self.encoder2_tokens = num_channels * tokens_per_channel  # 128
        self.total_tokens = self.encoder1_tokens + self.encoder2_tokens  # 256
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)
        
        # Generate S-random interleaver for encoder 2
        self.interleaver = generate_s_random_interleaver(self.encoder1_tokens, S=8)
        
        # Generate two different (but related) bias codebooks
        self._generate_bias_codebooks()
        
        print(f"Turbo FDM Encoder initialized:")
        print(f"  Encoder 1: positions 0-{self.encoder1_tokens-1}")
        print(f"  Encoder 2: positions {self.encoder1_tokens}-{self.total_tokens-1} (interleaved)")
        print(f"  S-random interleaver with S={int(np.sqrt(self.encoder1_tokens/2))}")
    
    def _generate_bias_codebooks(self):
        """
        Generate two related but different bias codebooks.
        
        Codebook 1: Primary bias tokens for each (channel, value) pair
        Codebook 2: Secondary bias tokens (overlapping but shifted)
        
        The overlap creates correlation that iterative decoding can exploit.
        """
        self.codebook1 = {}  # (channel, value) -> set of boosted token IDs
        self.codebook2 = {}
        
        for ch, (name, values) in MEMORY_SCHEMAS.items():
            for val in values:
                # Deterministic seed for reproducibility
                seed = hash(f"{name}:{val}") % (2**32)
                rng = np.random.RandomState(seed)
                
                # Codebook 1: 200 random tokens
                tokens1 = set(rng.choice(self.vocab_size, size=200, replace=False))
                
                # Codebook 2: 150 overlapping + 50 new tokens
                # This creates correlation between the two encodings
                overlap = set(list(tokens1)[:150])
                new_tokens = set(rng.choice(self.vocab_size, size=50, replace=False))
                tokens2 = overlap | new_tokens
                
                self.codebook1[(ch, val)] = tokens1
                self.codebook2[(ch, val)] = tokens2
    
    def _get_bias(self, channel_id, value, encoder_num):
        """Get bias vector for a (channel, value) pair from specified codebook."""
        codebook = self.codebook1 if encoder_num == 1 else self.codebook2
        boosted_tokens = codebook[(channel_id, value)]
        
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        for t in boosted_tokens:
            bias[t] = self.bias_strength
        
        return bias
    
    def _softmax(self, x):
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
    
    def encode_memory(self, memory: dict) -> str:
        """
        Encode memory using turbo-code structure.
        
        First half: Encoder 1 (direct)
        Second half: Encoder 2 (interleaved positions, related bias)
        """
        input_ids = self.tokenizer.encode("Memory:", return_tensors='pt').to(self.device)
        generated_tokens = []
        
        with torch.no_grad():
            for pos in range(self.total_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                if pos < self.encoder1_tokens:
                    # Encoder 1: Direct encoding
                    channel_id = pos % self.num_channels
                    token_in_channel = pos // self.num_channels
                    value = memory.get(channel_id, MEMORY_SCHEMAS[channel_id][1][0])
                    
                    bias = self._get_bias(channel_id, value, encoder_num=1)
                    logits = logits + bias
                    
                else:
                    # Encoder 2: Interleaved encoding
                    interleaved_pos = pos - self.encoder1_tokens
                    original_pos = self.interleaver[interleaved_pos]
                    
                    channel_id = original_pos % self.num_channels
                    value = memory.get(channel_id, MEMORY_SCHEMAS[channel_id][1][0])
                    
                    bias = self._get_bias(channel_id, value, encoder_num=2)
                    logits = logits + bias
                
                probs = self._softmax(logits)
                token = np.random.choice(self.vocab_size, p=probs)
                
                generated_tokens.append(token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
        
        return self.tokenizer.decode(generated_tokens)


def generate_explicit_memory_string(memory: dict) -> str:
    parts = []
    for ch in range(8):
        if ch in memory:
            name, _ = MEMORY_SCHEMAS[ch]
            parts.append(f"{name}:{memory[ch]}")
    return "|".join(parts)


def generate_turbo_dataset(num_samples=5000, output_path="turbo_fdm_v2_training.jsonl"):
    """Generate training data with turbo-code FDM."""
    
    encoder = TurboFDMEncoder(bias_strength=10.0)
    samples = []
    
    print(f"\nGenerating {num_samples} turbo-FDM samples...")
    
    for _ in tqdm(range(num_samples)):
        # Random memory
        memory = {}
        for ch, (name, values) in MEMORY_SCHEMAS.items():
            memory[ch] = random.choice(values)
        
        # Encode with turbo structure
        fdm_text = encoder.encode_memory(memory)
        
        # Explicit target
        explicit = generate_explicit_memory_string(memory)
        
        samples.append({
            "fdm_text": fdm_text,
            "explicit": explicit,
            "memory": {str(k): v for k, v in memory.items()},
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
    
    return samples


if __name__ == "__main__":
    generate_turbo_dataset(num_samples=5000)
