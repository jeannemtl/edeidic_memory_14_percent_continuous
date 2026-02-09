"""
Train model to use encoded reasoning modifiers.

The model learns:
1. Read facts from memory
2. Read reasoning RULE from memory  
3. Read META instruction from memory
4. Apply rule + meta to facts to generate answer

Same facts + different rule = different answer
Same facts + different meta = different answer
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
from tqdm import tqdm


class ReasoningModifierDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Include rule description in prompt so model learns what rules mean
        input_text = f"[MEM]{s['explicit']}[/MEM]\n[RULE_DESC]{s['rule_description']}[/RULE_DESC]\nUser: {s['question']}\nAssistant:"
        target_text = f" {s['answer']}"
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
        
        # Mask input portion
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['[MEM]', '[/MEM]', '[RULE_DESC]', '[/RULE_DESC]']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    train_dataset = ReasoningModifierDataset('reasoning_modifiers_training_train.jsonl', tokenizer)
    val_dataset = ReasoningModifierDataset('reasoning_modifiers_training_val.jsonl', tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    optimizer = AdamW(model.parameters(), lr=3e-5)  # Lower LR for reasoning task
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(15):  # More epochs for reasoning
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
            patience_counter = 0
            model.save_pretrained('reasoning_modifiers_v1')
            tokenizer.save_pretrained('reasoning_modifiers_v1')
            print(f"  Saved best model (val_loss = {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    train()
