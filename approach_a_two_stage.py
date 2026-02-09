"""
Approach A: Two-Stage Decoder

Stage 1: FDM tokens → Explicit memory string (trained decoder)
Stage 2: Explicit memory string → Answer (already 100% model)

The FDM decoder converts encoded tokens into readable format,
then passes to the explicit memory model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
from tqdm import tqdm
import random

# Memory schemas (same as before)
MEMORY_SCHEMAS = {
    0: ("SECRET", ["RED", "BLUE", "GREEN", "GOLD", "ALPHA", "BETA", "GAMMA"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN", "CAIRO", "ROME"]),
    2: ("AGENT", ["ALICE", "BOB", "CAROL", "DAVE", "EVE", "FRANK"]),
    3: ("TASK", ["WAIT", "GO", "SEND", "FETCH", "SCAN", "REPORT"]),
    4: ("STATUS", ["READY", "BUSY", "IDLE", "ACTIVE", "DONE"]),
    5: ("MOOD", ["HAPPY", "CALM", "ALERT", "FOCUSED", "EAGER"]),
    6: ("PRIORITY", ["HIGH", "MEDIUM", "LOW", "URGENT", "NORMAL"]),
    7: ("MODE", ["AUTO", "MANUAL", "STANDBY", "ACTIVE", "SAFE"]),
}


class FDMEncoder:
    """Generate FDM-encoded tokens for memory."""
    
    def __init__(self, num_channels=8, tokens_per_channel=16, bias_strength=3.0):
        self.num_channels = num_channels
        self.tokens_per_channel = tokens_per_channel
        self.bias_strength = bias_strength
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
        boosted = rng.choice(self.vocab_size, size=500, replace=False)
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
                
                # Get value for this channel
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
    """Generate explicit memory string."""
    parts = []
    for ch in range(8):
        if ch in memory:
            name, _ = MEMORY_SCHEMAS[ch]
            parts.append(f"{name}:{memory[ch]}")
    return "|".join(parts)


def generate_decoder_training_data(num_samples=5000, output_path="fdm_decoder_training.jsonl"):
    """Generate data to train FDM → explicit decoder."""
    
    encoder = FDMEncoder(bias_strength=5.0)  # Stronger bias for clearer signal
    samples = []
    
    print(f"Generating {num_samples} FDM decoder training samples...")
    
    for _ in tqdm(range(num_samples)):
        # Random memory
        memory = {}
        for ch in range(8):
            _, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)
        
        # Encode to FDM
        fdm_text, fdm_tokens = encoder.encode_memory(memory)
        
        # Explicit target
        explicit = generate_explicit_memory_string(memory)
        
        samples.append({
            "fdm_text": fdm_text,
            "fdm_tokens": fdm_tokens,
            "explicit": explicit,
            "memory": {str(k): v for k, v in memory.items()},
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


class FDMDecoderDataset(Dataset):
    """Dataset for FDM → explicit decoder."""
    
    def __init__(self, path, tokenizer, max_length=256):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Input: FDM text, Output: explicit memory
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
        
        # Mask input
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def train_fdm_decoder():
    """Train the FDM → explicit decoder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[FDM]', '[/FDM]', '[DECODE]']})
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    train_dataset = FDMDecoderDataset('fdm_decoder_training_train.jsonl', tokenizer)
    val_dataset = FDMDecoderDataset('fdm_decoder_training_val.jsonl', tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_val_loss = float('inf')
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained('fdm_decoder_v1')
            tokenizer.save_pretrained('fdm_decoder_v1')
            print(f"  Saved best model (val_loss = {val_loss:.4f})")


def evaluate_two_stage():
    """Evaluate the full two-stage pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load FDM decoder
    fdm_tokenizer = GPT2Tokenizer.from_pretrained('fdm_decoder_v1')
    fdm_model = GPT2LMHeadModel.from_pretrained('fdm_decoder_v1')
    fdm_model.to(device)
    fdm_model.eval()
    
    # Load explicit memory model
    explicit_tokenizer = GPT2Tokenizer.from_pretrained('./eidetic_explicit_v1')
    explicit_model = GPT2LMHeadModel.from_pretrained('./eidetic_explicit_v1')
    explicit_model.to(device)
    explicit_model.eval()
    
    # Test samples
    test_samples = [json.loads(l) for l in open('fdm_decoder_training_test.jsonl')]
    
    correct = 0
    total = 0
    
    QUESTIONS = {
        0: "What is the secret code?",
        1: "What is our location?",
        2: "Who is the agent?",
        3: "What is my task?",
        4: "What is the status?",
        5: "What's the vibe?",
        6: "What's the priority?",
        7: "What mode are we in?",
    }
    
    print("Evaluating two-stage pipeline...\n")
    
    for i, sample in enumerate(test_samples[:50]):
        fdm_text = sample['fdm_text']
        memory = sample['memory']
        
        # Stage 1: FDM → explicit
        decode_input = f"[FDM]{fdm_text}[/FDM][DECODE]"
        input_ids = fdm_tokenizer.encode(decode_input, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = fdm_model.generate(input_ids, max_new_tokens=60, do_sample=False,
                                        pad_token_id=fdm_tokenizer.eos_token_id)
        decoded_explicit = fdm_tokenizer.decode(output[0])[len(decode_input):].strip()
        decoded_explicit = decoded_explicit.split('<|endoftext|>')[0]
        
        # Stage 2: Ask question using decoded memory
        asked_channel = random.randint(0, 7)
        question = QUESTIONS[asked_channel]
        
        explicit_input = f"[MEM]{decoded_explicit}[/MEM]\nUser: {question}\nAssistant:"
        input_ids = explicit_tokenizer.encode(explicit_input, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = explicit_model.generate(input_ids, max_new_tokens=20, do_sample=False,
                                            pad_token_id=explicit_tokenizer.eos_token_id)
        response = explicit_tokenizer.decode(output[0])[len(explicit_input):].strip()
        
        expected_value = memory[str(asked_channel)]
        is_correct = expected_value in response
        
        if is_correct:
            correct += 1
        total += 1
        
        if i < 5:
            print(f"FDM decoded: {decoded_explicit[:60]}...")
            print(f"Q: {question}")
            print(f"Expected: {expected_value}")
            print(f"Got: {response[:50]}")
            print(f"{'✓' if is_correct else '✗'}\n")
    
    print(f"\nTwo-Stage Accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate_decoder_training_data(num_samples=5000)
        elif sys.argv[1] == "train":
            train_fdm_decoder()
        elif sys.argv[1] == "eval":
            evaluate_two_stage()
    else:
        print("Usage:")
        print("  python approach_a_two_stage.py generate  # Generate training data")
        print("  python approach_a_two_stage.py train     # Train FDM decoder")
        print("  python approach_a_two_stage.py eval      # Evaluate full pipeline")
