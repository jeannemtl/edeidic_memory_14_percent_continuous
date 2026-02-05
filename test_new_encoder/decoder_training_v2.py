"""
decoder_training_v2.py — Train GPT-2 to decode leaky stegotext

Key changes from v1:
  1. CLASSIFICATION not GENERATION — predict context class, not generate text
  2. Channel-segmented input — model sees tokens from ONE channel at a time
  3. Proper loss — cross-entropy over context classes, not next-token prediction
  4. No mode collapse — balanced sampling + label smoothing
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer
from safetensors.torch import save_file, load_file
from tqdm import tqdm
import numpy as np
import os


# ============================================================================
# CONTEXT VOCABULARY — all possible context strings
# ============================================================================

CONTEXT_TEMPLATES = {
    "secret": ["SECRET:{}", "PASSWORD:{}", "CODE:{}"],
    "location": ["LOCATION:{}", "PLACE:{}", "WHERE:{}"],
    "agent": ["AGENT:{}", "NAME:{}", "WHO:{}"],
    "task": ["TASK:{}", "DO:{}", "ACTION:{}"],
    "status": ["STATUS:{}", "STATE:{}", "MODE:{}"],
    "number": ["COUNT:{}", "VALUE:{}", "NUM:{}"],
    "flag": ["FLAG:{}", "ENABLED:{}", "ACTIVE:{}"],
    "mood": ["MOOD:{}", "FEELING:{}", "VIBE:{}"],
}

FILL_VALUES = {
    "secret": ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA",
               "RED", "BLUE", "GREEN", "GOLD", "SILVER"],
    "location": ["PARIS", "TOKYO", "LONDON", "BERLIN", "CAIRO",
                  "ROME", "DUBAI", "SEOUL", "MOSCOW", "SYDNEY"],
    "agent": ["ALICE", "BOB", "CAROL", "DAVE", "EVE",
              "FRANK", "GRACE", "HANK", "IVY", "JACK"],
    "task": ["WAIT", "SEND", "FETCH", "SCAN", "ALERT",
             "SYNC", "BACKUP", "DEPLOY", "MONITOR", "REPORT"],
    "status": ["READY", "BUSY", "IDLE", "ERROR", "ACTIVE",
               "PAUSED", "DONE", "INIT", "RUNNING", "STOPPED"],
    "number": [str(i) for i in range(100, 1000, 100)],  # representative subset
    "flag": ["TRUE", "FALSE"],
    "mood": ["HAPPY", "CALM", "ALERT", "FOCUSED", "READY",
             "EAGER", "PATIENT", "CAUTIOUS"],
}


def build_context_vocab():
    """Build complete list of all possible context strings → class indices."""
    all_contexts = []
    for category, templates in CONTEXT_TEMPLATES.items():
        for template in templates:
            for value in FILL_VALUES[category]:
                all_contexts.append(template.format(value))
    
    # Sort for deterministic ordering
    all_contexts = sorted(set(all_contexts))
    context_to_idx = {ctx: i for i, ctx in enumerate(all_contexts)}
    idx_to_context = {i: ctx for ctx, i in context_to_idx.items()}
    
    return context_to_idx, idx_to_context


CONTEXT_TO_IDX, IDX_TO_CONTEXT = build_context_vocab()
NUM_CLASSES = len(CONTEXT_TO_IDX)
print(f"Context vocabulary: {NUM_CLASSES} classes")


# ============================================================================
# DATASET
# ============================================================================

class EideticDataset(Dataset):
    """
    Dataset for eidetic decoder training.
    
    Each sample: (stegotext_tokens_for_channel, channel_idx, context_class)
    
    The key difference from v1: we give the model tokens from ONE channel
    at a time (not the full stegotext), and ask it to classify the context.
    """
    
    def __init__(self, jsonl_path, max_seq_len=256, tokens_per_channel=32):
        self.samples = []
        self.max_seq_len = max_seq_len
        self.tokens_per_channel = tokens_per_channel
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        print(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                context = data['context']
                
                # Skip contexts not in our vocabulary
                if context not in CONTEXT_TO_IDX:
                    continue
                
                self.samples.append({
                    'stegotext': data['stegotext'],
                    'channel': data['channel'],
                    'context_idx': CONTEXT_TO_IDX[context],
                    'context': context,
                })
        
        print(f"Loaded {len(self.samples)} samples")
        
        # Check class balance
        class_counts = {}
        for s in self.samples:
            c = s['context']
            class_counts[c] = class_counts.get(c, 0) + 1
        print(f"Unique contexts: {len(class_counts)}")
        
        # Compute class weights for balanced loss
        total = len(self.samples)
        self.class_weights = torch.ones(NUM_CLASSES)
        for ctx, count in class_counts.items():
            idx = CONTEXT_TO_IDX[ctx]
            # Inverse frequency weighting
            self.class_weights[idx] = total / (len(class_counts) * count)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize the stegotext
        token_ids = self.tokenizer.encode(sample['stegotext'])
        
        # Extract this channel's segment
        ch = sample['channel']
        tpc = self.tokens_per_channel
        ch_start = ch * tpc
        ch_end = ch_start + tpc
        
        # Get channel segment (with some context before/after for continuity)
        context_window = 16  # extra tokens before channel segment
        seg_start = max(0, ch_start - context_window)
        seg_end = min(len(token_ids), ch_end + context_window)
        
        channel_tokens = token_ids[seg_start:seg_end]
        
        # Pad or truncate to fixed length
        if len(channel_tokens) > self.max_seq_len:
            channel_tokens = channel_tokens[:self.max_seq_len]
        
        # Create attention mask
        attn_mask = [1] * len(channel_tokens)
        
        # Pad
        pad_len = self.max_seq_len - len(channel_tokens)
        channel_tokens = channel_tokens + [self.tokenizer.eos_token_id] * pad_len
        attn_mask = attn_mask + [0] * pad_len
        
        return {
            'input_ids': torch.tensor(channel_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
            'channel_idx': torch.tensor(ch, dtype=torch.long),
            'label': torch.tensor(sample['context_idx'], dtype=torch.long),
        }


# ============================================================================
# MODEL — GPT-2 backbone + classification head
# ============================================================================

class EideticDecoderModel(nn.Module):
    """
    GPT-2 backbone (frozen or LoRA) + classification head.
    
    Input: stegotext tokens from one channel + channel index
    Output: probability distribution over all possible context strings
    """
    
    def __init__(self, num_classes, num_channels=8, 
                 hidden_dim=768, dropout=0.1, freeze_backbone=True):
        super().__init__()
        
        # GPT-2 backbone
        self.backbone = GPT2Model.from_pretrained('gpt2')
        self.hidden_dim = self.backbone.config.n_embd  # 768 for gpt2
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen — training classification head only")
        
        # Channel embedding
        self.channel_embed = nn.Embedding(num_channels, self.hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),  # concat backbone + channel
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Parameters: {trainable:,} trainable / {total:,} total")
    
    def forward(self, input_ids, attention_mask, channel_idx):
        """
        Args:
            input_ids: [batch, seq_len] — stegotext tokens for one channel
            attention_mask: [batch, seq_len]
            channel_idx: [batch] — which channel (0..N-1)
        
        Returns:
            logits: [batch, num_classes] — context class probabilities
        """
        # Get GPT-2 hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Pool: mean of non-padded positions
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # pooled: [batch, hidden]
        
        # Channel embedding
        ch_emb = self.channel_embed(channel_idx)  # [batch, hidden]
        
        # Concatenate and classify
        combined = torch.cat([pooled, ch_emb], dim=-1)  # [batch, hidden*2]
        logits = self.classifier(combined)  # [batch, num_classes]
        
        return logits


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(
    train_path="eidetic_training_data_train.jsonl",
    val_path="eidetic_training_data_val.jsonl",
    output_dir="eidetic_decoder_v2",
    num_epochs=20,
    batch_size=32,
    lr=1e-3,
    tokens_per_channel=32,
    label_smoothing=0.1,
    freeze_backbone=True,
):
    """Train the eidetic decoder."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Datasets
    train_ds = EideticDataset(train_path, tokens_per_channel=tokens_per_channel)
    val_ds = EideticDataset(val_path, tokens_per_channel=tokens_per_channel)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    # Model
    model = EideticDecoderModel(
        num_classes=NUM_CLASSES,
        freeze_backbone=freeze_backbone,
    ).to(device)
    
    # Loss with label smoothing + class balancing
    class_weights = train_ds.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    
    # Optimizer — only trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    os.makedirs(output_dir, exist_ok=True)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            channel_idx = batch['channel_idx'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attn_mask, channel_idx)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{train_correct/train_total*100:.1f}%"
            )
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                channel_idx = batch['channel_idx'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attn_mask, channel_idx)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train: loss={avg_train_loss:.4f}, acc={train_acc*100:.1f}%")
        print(f"  Val:   loss={avg_val_loss:.4f}, acc={val_acc*100:.1f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model state
            state_dict = {k: v for k, v in model.state_dict().items()
                         if 'backbone' not in k}  # save only head weights
            save_file(state_dict, os.path.join(output_dir, "best_model.safetensors"))
            
            # Save config
            config = {
                'num_classes': NUM_CLASSES,
                'num_channels': 8,
                'best_val_acc': best_val_acc,
                'epoch': epoch + 1,
                'context_to_idx': CONTEXT_TO_IDX,
                'idx_to_context': IDX_TO_CONTEXT,
            }
            with open(os.path.join(output_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            print(f"  ✓ New best! Saved to {output_dir}/")
        
        # Early stopping check
        if val_acc > 0.95:
            print(f"\n✓ Reached {val_acc*100:.1f}% val accuracy — stopping early!")
            break
    
    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.1f}%")
    return model


if __name__ == "__main__":
    model = train(
        train_path="eidetic_training_data_train.jsonl",
        val_path="eidetic_training_data_val.jsonl",
        output_dir="eidetic_decoder_v2",
        num_epochs=20,
        batch_size=32,
        lr=1e-3,
        freeze_backbone=True,
    )
