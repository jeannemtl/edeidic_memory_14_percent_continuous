"""
Approach C: Curriculum Learning for FDM Decoding

Start with EASY encoding (high bias = obvious signal)
Gradually increase difficulty (lower bias = subtler signal)

Curriculum stages:
  Stage 1: bias_strength=20.0 (very obvious)
  Stage 2: bias_strength=10.0 (obvious)
  Stage 3: bias_strength=5.0  (moderate)
  Stage 4: bias_strength=3.0  (subtle)
  Stage 5: bias_strength=2.0  (hard)

The model learns progressively harder decoding tasks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
from tqdm import tqdm
import random
import os

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


class FDMEncoderCurriculum:
    """FDM encoder with adjustable bias strength for curriculum learning."""
    
    def __init__(self, bias_strength=10.0, num_channels=8, tokens_per_channel=16):
        self.bias_strength = bias_strength
        self.num_channels = num_channels
        self.tokens_per_channel = tokens_per_channel
        self.memory_tokens = num_channels * tokens_per_channel
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)
    
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
        # More tokens boosted = stronger signal
        num_boosted = max(100, int(500 * (self.bias_strength / 10.0)))
        boosted = rng.choice(self.vocab_size, size=num_boosted, replace=False)
        bias[boosted] = self.bias_strength
        return bias
    
    def _softmax(self, x):
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
    
    def encode_memory(self, memory: dict) -> tuple:
        """Encode memory dict into FDM tokens."""
        input_ids = self.tokenizer.encode("Memory:", return_tensors='pt').to(self.device)
        generated_tokens = []
        
        with torch.no_grad():
            for pos in range(self.memory_tokens):
                channel_id = pos % self.num_channels
                token_idx = pos // self.num_channels
                
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                name, _ = MEMORY_SCHEMAS[channel_id]
                value = memory.get(channel_id, "")
                context = f"{name}:{value}" if value else ""
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
        
        fdm_text = self.tokenizer.decode(generated_tokens)
        return fdm_text, generated_tokens


def generate_explicit_memory_string(memory: dict) -> str:
    parts = []
    for ch in range(8):
        if ch in memory:
            name, _ = MEMORY_SCHEMAS[ch]
            parts.append(f"{name}:{memory[ch]}")
    return "|".join(parts)


def generate_curriculum_data(bias_strength, num_samples, output_path):
    """Generate training data at a specific difficulty level."""
    
    encoder = FDMEncoderCurriculum(bias_strength=bias_strength)
    samples = []
    
    print(f"Generating {num_samples} samples with bias_strength={bias_strength}...")
    
    for _ in tqdm(range(num_samples)):
        # Random memory
        memory = {}
        for ch, (name, values) in MEMORY_SCHEMAS.items():
            memory[ch] = random.choice(values)
        
        # Encode to FDM
        fdm_text, fdm_tokens = encoder.encode_memory(memory)
        
        # Explicit target
        explicit = generate_explicit_memory_string(memory)
        
        samples.append({
            "fdm_text": fdm_text,
            "explicit": explicit,
            "memory": {str(k): v for k, v in memory.items()},
            "bias_strength": bias_strength,
        })
    
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Saved to {output_path}")
    return samples


class CurriculumDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=384):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        input_text = f"[FDM]{s['fdm_text']}[/FDM][DECODE]"
        target_text = s['explicit']
        full_text = input_text + target_text
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
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


def train_curriculum():
    """Train with curriculum learning — progressively harder FDM decoding."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Curriculum stages: high bias (easy) → low bias (hard)
    curriculum = [
        {"bias": 20.0, "epochs": 3, "samples": 2000},
        {"bias": 10.0, "epochs": 3, "samples": 2000},
        {"bias": 5.0,  "epochs": 4, "samples": 3000},
        {"bias": 3.0,  "epochs": 5, "samples": 3000},
    ]
    
    # Initialize model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[FDM]', '[/FDM]', '[DECODE]']})
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    for stage_idx, stage in enumerate(curriculum):
        bias = stage["bias"]
        epochs = stage["epochs"]
        num_samples = stage["samples"]
        
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage_idx + 1}: bias_strength={bias}")
        print(f"{'='*60}\n")
        
        # Generate data for this stage
        train_path = f"curriculum_stage{stage_idx}_train.jsonl"
        val_path = f"curriculum_stage{stage_idx}_val.jsonl"
        
        if not os.path.exists(train_path):
            samples = generate_curriculum_data(bias, num_samples, f"curriculum_stage{stage_idx}.jsonl")
            
            # Split
            random.shuffle(samples)
            n = len(samples)
            train_samples = samples[:int(0.9*n)]
            val_samples = samples[int(0.9*n):]
            
            with open(train_path, 'w') as f:
                for s in train_samples:
                    f.write(json.dumps(s) + "\n")
            with open(val_path, 'w') as f:
                for s in val_samples:
                    f.write(json.dumps(s) + "\n")
        
        train_dataset = CurriculumDataset(train_path, tokenizer)
        val_dataset = CurriculumDataset(val_path, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Stage {stage_idx+1} Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
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
            print(f"Stage {stage_idx+1} Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Save after each stage
        model.save_pretrained(f'curriculum_model_stage{stage_idx}')
        tokenizer.save_pretrained(f'curriculum_model_stage{stage_idx}')
        print(f"Saved stage {stage_idx+1} model")
    
    # Save final model
    model.save_pretrained('curriculum_fdm_decoder_final')
    tokenizer.save_pretrained('curriculum_fdm_decoder_final')
    print("\nSaved final curriculum model")


def evaluate_curriculum():
    """Evaluate the curriculum-trained model."""
    
    device = torch.device('cuda')
    
    tokenizer = GPT2Tokenizer.from_pretrained('./curriculum_fdm_decoder_final')
    model = GPT2LMHeadModel.from_pretrained('./curriculum_fdm_decoder_final')
    model.to(device)
    model.eval()
    
    # Test on different bias levels
    test_biases = [20.0, 10.0, 5.0, 3.0, 2.0]
    
    for bias in test_biases:
        print(f"\n=== Testing on bias_strength={bias} ===\n")
        
        # Generate test samples
        test_path = f"curriculum_test_bias{int(bias)}.jsonl"
        if not os.path.exists(test_path):
            generate_curriculum_data(bias, 100, test_path)
        
        test_samples = [json.loads(l) for l in open(test_path)]
        
        correct_channels = 0
        total_channels = 0
        
        for i, sample in enumerate(test_samples[:30]):
            fdm_text = sample['fdm_text']
            memory = sample['memory']
            
            input_text = f"[FDM]{fdm_text}[/FDM][DECODE]"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=80, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
            
            decoded = tokenizer.decode(output[0])[len(input_text):].strip()
            decoded = decoded.split('<|endoftext|>')[0]
            
            for ch, val in memory.items():
                total_channels += 1
                if val in decoded:
                    correct_channels += 1
            
            if i < 3:
                print(f"Expected: {sample['explicit'][:50]}...")
                print(f"Decoded:  {decoded[:50]}...")
                print()
        
        accuracy = 100 * correct_channels / total_channels
        print(f"Bias {bias}: Channel Accuracy = {correct_channels}/{total_channels} = {accuracy:.1f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_curriculum()
        elif sys.argv[1] == "eval":
            evaluate_curriculum()
    else:
        print("Usage:")
        print("  python approach_c_curriculum.py train  # Train with curriculum")
        print("  python approach_c_curriculum.py eval   # Evaluate")
