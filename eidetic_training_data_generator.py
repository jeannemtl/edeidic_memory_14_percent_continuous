"""
Generate training data for eidetic model.

Each sample:
- Input: [memory_encoded_preamble] + "User: {question}\nAssistant:"
- Output: Response that correctly uses the encoded memory

The model learns to decode the memory preamble and use it in responses.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from tqdm import tqdm
import random

# Memory templates and values
MEMORY_SCHEMAS = {
    0: ("secret", ["RED", "BLUE", "GREEN", "GOLD", "ALPHA", "BETA", "GAMMA"]),
    1: ("location", ["PARIS", "TOKYO", "LONDON", "BERLIN", "CAIRO", "ROME"]),
    2: ("agent", ["ALICE", "BOB", "CAROL", "DAVE", "EVE", "FRANK"]),
    3: ("task", ["WAIT", "GO", "SEND", "FETCH", "SCAN", "REPORT"]),
    4: ("status", ["READY", "BUSY", "IDLE", "ACTIVE", "DONE"]),
    5: ("mood", ["HAPPY", "CALM", "ALERT", "FOCUSED", "EAGER"]),
    6: ("priority", ["HIGH", "MEDIUM", "LOW", "URGENT", "NORMAL"]),
    7: ("mode", ["AUTO", "MANUAL", "STANDBY", "ACTIVE", "SAFE"]),
}

# Questions that require specific memory channels to answer
QUESTIONS = {
    0: [
        ("What is the secret code?", "The secret code is {value}."),
        ("Tell me the password.", "The password is {value}."),
        ("What's the secret?", "The secret is {value}."),
    ],
    1: [
        ("Where are we?", "We are in {value}."),
        ("What is our location?", "Our location is {value}."),
        ("Where should I go?", "You should go to {value}."),
    ],
    2: [
        ("Who is the agent?", "The agent is {value}."),
        ("Who am I working with?", "You are working with {value}."),
        ("Who is assigned?", "{value} is assigned."),
    ],
    3: [
        ("What should I do?", "You should {value}."),
        ("What is my task?", "Your task is to {value}."),
        ("What's the current task?", "The current task is {value}."),
    ],
    4: [
        ("What is the status?", "The status is {value}."),
        ("Are we ready?", "Status is {value}."),
        ("What's our current state?", "Current state is {value}."),
    ],
    5: [
        ("How are things?", "The mood is {value}."),
        ("What's the vibe?", "The vibe is {value}."),
        ("How should I feel?", "You should feel {value}."),
    ],
    6: [
        ("What's the priority?", "Priority is {value}."),
        ("How urgent is this?", "This is {value} priority."),
        ("What priority level?", "Priority level is {value}."),
    ],
    7: [
        ("What mode are we in?", "We are in {value} mode."),
        ("What's the current mode?", "Current mode is {value}."),
        ("Which mode?", "Mode is {value}."),
    ],
}


class EideticDataGenerator:
    def __init__(self, num_channels=8, tokens_per_channel=16, bias_strength=3.0):
        self.num_channels = num_channels
        self.tokens_per_channel = tokens_per_channel
        self.bias_strength = bias_strength
        self.memory_tokens = num_channels * tokens_per_channel
        
        print("Loading GPT-2 for data generation...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)
        
        # Add pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _context_to_bits(self, context_string):
        if not context_string:
            return "0" * 64
        context_bytes = context_string.encode('utf-8')
        bits = ''.join(format(byte, '08b') for byte in context_bytes)
        return bits.ljust(64, '0')[:64]
    
    def _get_channel_bias(self, bits, token_idx, channel_id):
        bit_int = int(bits[:min(32, len(bits))], 2) if bits else 0
        seed = 42 + channel_id * 1000 + bit_int + token_idx
        rng = np.random.RandomState(seed)
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        boosted = rng.choice(self.vocab_size, size=500, replace=False)
        bias[boosted] = self.bias_strength
        return bias
    
    def _softmax(self, x):
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
    
    def generate_memory_preamble(self, memory: dict) -> tuple:
        """Generate memory-encoded token sequence."""
        input_ids = self.tokenizer.encode("Memory:", return_tensors='pt').to(self.device)
        generated_tokens = []
        
        with torch.no_grad():
            for pos in range(self.memory_tokens):
                channel_id = pos % self.num_channels
                token_idx = pos // self.num_channels
                
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                context = memory.get(channel_id, "")
                bits = self._context_to_bits(context)
                bias = self._get_channel_bias(bits, token_idx, channel_id)
                biased_logits = logits + bias
                
                probs = self._softmax(biased_logits)
                token = np.random.choice(self.vocab_size, p=probs)
                
                generated_tokens.append(token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
        
        preamble_text = self.tokenizer.decode(generated_tokens)
        return preamble_text, generated_tokens
    
    def generate_sample(self):
        """Generate one training sample."""
        # Random memory state
        memory = {}
        for ch in range(self.num_channels):
            schema_name, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)
        
        # Generate memory preamble
        preamble_text, preamble_tokens = self.generate_memory_preamble(memory)
        
        # Pick a random channel to ask about
        asked_channel = random.randint(0, self.num_channels - 1)
        question, answer_template = random.choice(QUESTIONS[asked_channel])
        
        # Format answer with actual memory value
        answer = answer_template.format(value=memory[asked_channel])
        
        # Build training sample
        input_text = f"[MEM]{preamble_text}[/MEM]\nUser: {question}\nAssistant:"
        target_text = f" {answer}"
        
        return {
            "input": input_text,
            "target": target_text,
            "full": input_text + target_text,
            "memory": {str(k): v for k, v in memory.items()},
            "question": question,
            "answer": answer,
            "asked_channel": asked_channel,
        }
    
    def generate_dataset(self, num_samples=5000, output_path="eidetic_memory_training.jsonl"):
        """Generate full training dataset."""
        samples = []
        
        print(f"Generating {num_samples} samples...")
        for _ in tqdm(range(num_samples)):
            sample = self.generate_sample()
            samples.append(sample)
        
        # Save
        with open(output_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        print(f"Saved {len(samples)} samples to {output_path}")
        
        # Splits
        random.shuffle(samples)
        n = len(samples)
        splits = {
            'train': samples[:int(0.8*n)],
            'val': samples[int(0.8*n):int(0.9*n)],
            'test': samples[int(0.9*n):],
        }
        
        for name, data in splits.items():
            path = output_path.replace(".jsonl", f"_{name}.jsonl")
            with open(path, 'w') as f:
                for s in data:
                    f.write(json.dumps(s) + "\n")
            print(f"  {name}: {len(data)} samples")
        
        return samples


if __name__ == "__main__":
    generator = EideticDataGenerator(
        num_channels=8,
        tokens_per_channel=16,
        bias_strength=3.0
    )
    generator.generate_dataset(num_samples=5000)
