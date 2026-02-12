"""
Train model to extract FDM-encoded memory from coherent text AND reason with it.

Two-stage task:
1. Extract: Read coherent text → decode hidden memory
2. Reason: Use extracted memory → generate analysis
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from tqdm import tqdm
import os

CHANNELS = {
    0: ("THREAT", ["minimal", "low", "moderate", "elevated", "high", "severe", "critical", "extreme"]),
    1: ("STATUS", ["offline", "degraded", "limited", "operational", "stable", "optimal", "enhanced", "peak"]),
    2: ("PRIORITY", ["routine", "low", "normal", "elevated", "high", "urgent", "critical", "emergency"]),
    3: ("TIMELINE", ["indefinite", "extended", "long", "medium", "short", "immediate", "urgent", "now"]),
    4: ("RESOURCES", ["depleted", "critical", "low", "limited", "adequate", "good", "abundant", "surplus"]),
    5: ("CONFIDENCE", ["none", "minimal", "low", "uncertain", "moderate", "good", "high", "confirmed"]),
    6: ("RISK", ["negligible", "minimal", "low", "moderate", "elevated", "high", "severe", "extreme"]),
    7: ("RECOMMENDATION", ["abort", "withdraw", "hold", "monitor", "prepare", "proceed", "advance", "execute"]),
}


class CoherentFDMDataset(Dataset):
    """Dataset for coherent FDM reasoning."""
    
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Input: FDM-encoded coherent text + prompt
        # Output: Memory extraction + reasoning
        input_text = f"[REPORT]{s['fdm_text']}[/REPORT]\nAnalyze this report and provide assessment:\n"
        output_text = f"[EXTRACT]{s['memory_explicit']}[/EXTRACT]\n{s['reasoning']}"
        
        full_text = input_text + output_text + self.tokenizer.eos_token
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels: mask input portion
        input_len = len(self.tokenizer.encode(input_text))
        labels = input_ids.clone()
        labels[:input_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'memory': s['memory'],
        }


def train(
    train_path="fdm_coherent_reasoning_train.jsonl",
    val_path="fdm_coherent_reasoning_val.jsonl",
    epochs=10,
    batch_size=4,
    lr=3e-5,
    checkpoint_dir="checkpoints_fdm_coherent"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.to(device)
    
    train_dataset = CoherentFDMDataset(train_path, tokenizer)
    val_dataset = CoherentFDMDataset(val_path, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        train_loss /= len(train_loader)
        
        # Validation
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
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained('fdm_coherent_reasoning_model')
            tokenizer.save_pretrained('fdm_coherent_reasoning_model')
            print(f"  ✓ New best model (val_loss = {val_loss:.4f})")
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")


def evaluate(model_path='fdm_coherent_reasoning_model', test_path='fdm_coherent_reasoning_test.jsonl'):
    """Evaluate extraction accuracy and reasoning quality."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    test_samples = [json.loads(l) for l in open(test_path)]
    
    # Track extraction accuracy per channel
    channel_correct = {ch: 0 for ch in range(8)}
    channel_total = {ch: 0 for ch in range(8)}
    
    print("\n=== Evaluation Examples ===\n")
    
    for i, sample in enumerate(tqdm(test_samples[:100], desc="Evaluating")):
        input_text = f"[REPORT]{sample['fdm_text']}[/REPORT]\nAnalyze this report and provide assessment:\n"
        
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated = generated[len(input_text):]
        
        # Check extraction accuracy
        for ch_id, (ch_name, values) in CHANNELS.items():
            expected_value = values[sample['memory'][str(ch_id)]]
            
            # Check if correct value appears in extraction
            if f"{ch_name}:{expected_value}" in generated:
                channel_correct[ch_id] += 1
            channel_total[ch_id] += 1
        
        # Show examples
        if i < 5:
            print(f"=== Sample {i+1} ===")
            print(f"Expected memory: {sample['memory_explicit'][:80]}...")
            print(f"Generated: {generated[:200]}...")
            print()
    
    # Print accuracy
    print("\n=== Extraction Accuracy per Channel ===\n")
    total_correct = 0
    total = 0
    for ch_id, (ch_name, _) in CHANNELS.items():
        acc = 100 * channel_correct[ch_id] / channel_total[ch_id] if channel_total[ch_id] > 0 else 0
        print(f"  {ch_name}: {acc:.1f}%")
        total_correct += channel_correct[ch_id]
        total += channel_total[ch_id]
    
    overall_acc = 100 * total_correct / total if total > 0 else 0
    print(f"\nOverall Extraction Accuracy: {overall_acc:.1f}%")
    
    # Bits analysis
    bits_per_channel = 3  # 8 values = 3 bits
    total_bits = 8 * bits_per_channel  # 24 bits
    bits_correct = overall_acc / 100 * total_bits
    
    print(f"\n=== Capacity Analysis ===")
    print(f"Channels: 8")
    print(f"Values per channel: 8 (3 bits)")
    print(f"Total capacity: {total_bits} bits")
    print(f"Bits extracted correctly: {bits_correct:.1f} / {total_bits}")
    print(f"Text was COHERENT (not gibberish)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "eval":
            evaluate()
    else:
        print("Usage:")
        print("  python train_fdm_coherent_reasoning.py train")
        print("  python train_fdm_coherent_reasoning.py eval")
