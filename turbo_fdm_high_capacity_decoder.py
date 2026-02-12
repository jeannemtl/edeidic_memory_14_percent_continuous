"""
High-capacity Turbo FDM Decoder with proper checkpoints.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer
from torch.optim import AdamW
import json
from tqdm import tqdm
import random
import os

NUM_CHANNELS = 16
VALUES_PER_CHANNEL = 16


class HighCapacityTurboDecoder(nn.Module):
    def __init__(self, num_channels=16, values_per_channel=16, tokens_per_channel=8, num_iterations=3):
        super().__init__()
        
        self.num_channels = num_channels
        self.values_per_channel = values_per_channel
        self.tokens_per_channel = tokens_per_channel
        self.encoder1_tokens = num_channels * tokens_per_channel
        self.encoder2_tokens = num_channels * tokens_per_channel
        self.total_tokens = self.encoder1_tokens + self.encoder2_tokens
        self.num_iterations = num_iterations
        
        self.token_encoder = GPT2Model.from_pretrained('gpt2')
        self.hidden_size = self.token_encoder.config.hidden_size
        
        self.decoder1_heads = nn.ModuleDict({
            str(ch): nn.Sequential(
                nn.Linear(self.hidden_size * tokens_per_channel, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, values_per_channel)
            )
            for ch in range(num_channels)
        })
        
        self.decoder2_heads = nn.ModuleDict({
            str(ch): nn.Sequential(
                nn.Linear(self.hidden_size * tokens_per_channel, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, values_per_channel)
            )
            for ch in range(num_channels)
        })
        
        self.prior_integration = nn.ModuleDict({
            str(ch): nn.Sequential(
                nn.Linear(values_per_channel * 2, 64),
                nn.ReLU(),
                nn.Linear(64, values_per_channel)
            )
            for ch in range(num_channels)
        })
        
        self.fusion = nn.ModuleDict({
            str(ch): nn.Sequential(
                nn.Linear(values_per_channel * 2, 64),
                nn.ReLU(),
                nn.Linear(64, values_per_channel)
            )
            for ch in range(num_channels)
        })
        
        self.iteration_weights = nn.Parameter(torch.ones(num_iterations) / num_iterations)
        self.interleaver = self._generate_s_random_interleaver(self.encoder1_tokens, S=8)
    
    def _generate_s_random_interleaver(self, length, S=8):
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
    
    def _get_channel_positions(self, encoder_num):
        positions = {}
        if encoder_num == 1:
            for ch in range(self.num_channels):
                positions[ch] = [ch + i * self.num_channels for i in range(self.tokens_per_channel)]
        else:
            for ch in range(self.num_channels):
                original_positions = [ch + i * self.num_channels for i in range(self.tokens_per_channel)]
                positions[ch] = [self.interleaver[p] + self.encoder1_tokens for p in original_positions]
        return positions
    
    def _extract_channel_features(self, hidden_states, positions):
        batch_size = hidden_states.size(0)
        features = []
        for pos in positions:
            if pos < hidden_states.size(1):
                features.append(hidden_states[:, pos, :])
            else:
                features.append(torch.zeros(batch_size, self.hidden_size, device=hidden_states.device))
        return torch.cat(features, dim=-1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.token_encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        enc1_positions = self._get_channel_positions(encoder_num=1)
        enc2_positions = self._get_channel_positions(encoder_num=2)
        
        batch_size = input_ids.size(0)
        priors = {ch: torch.zeros(batch_size, self.values_per_channel, device=input_ids.device) for ch in range(self.num_channels)}
        
        iteration_logits = []
        
        for iteration in range(self.num_iterations):
            dec1_logits = {}
            for ch in range(self.num_channels):
                features = self._extract_channel_features(hidden_states, enc1_positions[ch])
                raw_logits = self.decoder1_heads[str(ch)](features)
                if iteration > 0:
                    combined = torch.cat([raw_logits, priors[ch]], dim=-1)
                    dec1_logits[ch] = self.prior_integration[str(ch)](combined)
                else:
                    dec1_logits[ch] = raw_logits
            
            dec2_logits = {}
            for ch in range(self.num_channels):
                features = self._extract_channel_features(hidden_states, enc2_positions[ch])
                raw_logits = self.decoder2_heads[str(ch)](features)
                combined = torch.cat([raw_logits, dec1_logits[ch].detach()], dim=-1)
                dec2_logits[ch] = self.prior_integration[str(ch)](combined)
            
            fused_logits = {}
            for ch in range(self.num_channels):
                combined = torch.cat([dec1_logits[ch], dec2_logits[ch]], dim=-1)
                fused_logits[ch] = self.fusion[str(ch)](combined)
            
            iteration_logits.append(fused_logits)
            priors = {ch: fused_logits[ch].detach() for ch in range(self.num_channels)}
        
        final_logits = {}
        weights = torch.softmax(self.iteration_weights, dim=0)
        for ch in range(self.num_channels):
            weighted_sum = sum(weights[i] * iteration_logits[i][ch] for i in range(self.num_iterations))
            final_logits[ch] = weighted_sum
        
        return final_logits, iteration_logits


class HighCapacityDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=300):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        encoding = self.tokenizer(s['fdm_text'], truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = torch.zeros(NUM_CHANNELS, dtype=torch.long)
        for ch_str, value in s['memory'].items():
            labels[int(ch_str)] = value
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def train_high_capacity_decoder(
    train_path="turbo_high_capacity_training_train.jsonl",
    val_path="turbo_high_capacity_training_val.jsonl",
    epochs=20,
    batch_size=8,
    lr=3e-5,
    checkpoint_dir="checkpoints_high_capacity"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = HighCapacityTurboDecoder(num_channels=NUM_CHANNELS, values_per_channel=VALUES_PER_CHANNEL, tokens_per_channel=8, num_iterations=3)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Check for existing checkpoint to resume
    start_epoch = 0
    best_val_acc = 0
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        latest = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        print(f"Resuming from {latest}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"  Resuming at epoch {start_epoch + 1}, best_val_acc so far: {best_val_acc:.1f}%")
    
    model.to(device)
    
    train_dataset = HighCapacityDataset(train_path, tokenizer)
    val_dataset = HighCapacityDataset(val_path, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            
            loss = 0
            for ch in range(NUM_CHANNELS):
                ch_labels = labels[:, ch]
                loss += criterion(logits[ch], ch_labels)
                preds = logits[ch].argmax(dim=-1)
                train_correct += (preds == ch_labels).sum().item()
                train_total += len(ch_labels)
            
            loss = loss / NUM_CHANNELS
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        iteration_correct = [0, 0, 0]
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits, iteration_logits = model(input_ids, attention_mask)
                
                for ch in range(NUM_CHANNELS):
                    ch_labels = labels[:, ch]
                    for i, iter_logits in enumerate(iteration_logits):
                        preds = iter_logits[ch].argmax(dim=-1)
                        iteration_correct[i] += (preds == ch_labels).sum().item()
                    preds = logits[ch].argmax(dim=-1)
                    val_correct += (preds == ch_labels).sum().item()
                    val_total += len(ch_labels)
        
        val_acc = 100 * val_correct / val_total
        iter_accs = [100 * c / val_total for c in iteration_correct]
        
        bits_per_channel = np.log2(VALUES_PER_CHANNEL)
        total_bits = NUM_CHANNELS * bits_per_channel
        bits_correct = val_acc / 100 * total_bits
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%")
        print(f"  Val Acc: Iter1={iter_accs[0]:.1f}%, Iter2={iter_accs[1]:.1f}%, Iter3={iter_accs[2]:.1f}%, Final={val_acc:.1f}%")
        print(f"  Effective bits: {bits_correct:.1f} / {total_bits:.0f} bits")
        
        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'turbo_high_capacity_decoder_best.pt')
            print(f"  ✓ New best model (val_acc = {val_acc:.1f}%)")
        
        # Save checkpoint with best_val_acc
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'train_loss': train_loss,
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Best effective bits: {best_val_acc/100 * NUM_CHANNELS * np.log2(VALUES_PER_CHANNEL):.1f} / {NUM_CHANNELS * np.log2(VALUES_PER_CHANNEL):.0f}")


def evaluate_high_capacity_decoder(model_path='turbo_high_capacity_decoder_best.pt', test_path='turbo_high_capacity_training_test.jsonl'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = HighCapacityTurboDecoder(num_channels=NUM_CHANNELS, values_per_channel=VALUES_PER_CHANNEL, tokens_per_channel=8, num_iterations=3)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    test_dataset = HighCapacityDataset(test_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    iteration_correct = [0, 0, 0]
    final_correct = 0
    total = 0
    channel_correct = {ch: 0 for ch in range(NUM_CHANNELS)}
    channel_total = {ch: 0 for ch in range(NUM_CHANNELS)}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            final_logits, iteration_logits = model(input_ids, attention_mask)
            
            for ch in range(NUM_CHANNELS):
                ch_labels = labels[:, ch]
                for i, iter_logits in enumerate(iteration_logits):
                    preds = iter_logits[ch].argmax(dim=-1)
                    iteration_correct[i] += (preds == ch_labels).sum().item()
                preds = final_logits[ch].argmax(dim=-1)
                final_correct += (preds == ch_labels).sum().item()
                total += len(ch_labels)
                channel_correct[ch] += (preds == ch_labels).sum().item()
                channel_total[ch] += len(ch_labels)
    
    print("\n=== High-Capacity Turbo Decoder Results ===\n")
    print("Iteration-by-iteration:")
    for i in range(3):
        print(f"  Iteration {i+1}: {100 * iteration_correct[i] / total:.1f}%")
    
    final_acc = 100 * final_correct / total
    print(f"\nFinal (weighted): {final_acc:.1f}%")
    
    bits_per_channel = np.log2(VALUES_PER_CHANNEL)
    total_bits = NUM_CHANNELS * bits_per_channel
    bits_correct = final_acc / 100 * total_bits
    bits_per_token = bits_correct / 256
    
    print(f"\n=== Capacity Analysis ===")
    print(f"Channels: {NUM_CHANNELS}")
    print(f"Values per channel: {VALUES_PER_CHANNEL} ({bits_per_channel:.0f} bits)")
    print(f"Total capacity: {total_bits:.0f} bits in 256 tokens")
    print(f"Bits decoded correctly: {bits_correct:.1f} / {total_bits:.0f}")
    print(f"Effective bits/token: {bits_per_token:.3f}")
    print(f"Shannon limit: 15.6 bits/token")
    print(f"Efficiency: {bits_per_token / 15.6 * 100:.2f}% of Shannon limit")
    
    print("\nPer-channel accuracy:")
    for ch in range(NUM_CHANNELS):
        print(f"  CH{ch:02d}: {100 * channel_correct[ch] / channel_total[ch]:.1f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_high_capacity_decoder()
        elif sys.argv[1] == "eval":
            evaluate_high_capacity_decoder()
    else:
        print("Usage:")
        print("  python turbo_fdm_high_capacity_decoder.py train")
        print("  python turbo_fdm_high_capacity_decoder.py eval")
