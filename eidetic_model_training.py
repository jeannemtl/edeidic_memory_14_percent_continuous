"""
Fine-tune GPT-2 to be an eidetic model.

The model learns to:
1. Read memory-encoded preambles [MEM]...[/MEM]
2. Extract the encoded context
3. Use that context to answer questions correctly
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
from tqdm import tqdm


class EideticDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize full sequence (input + target)
        full_text = sample['full']
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels = input_ids (causal LM)
        labels = input_ids.clone()
        
        # Mask out the input portion (only train on target)
        input_text = sample['input']
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100  # Ignore in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def train_eidetic_model(
    train_path="eidetic_memory_training_train.jsonl",
    val_path="eidetic_memory_training_val.jsonl",
    output_dir="eidetic_model_v1",
    num_epochs=10,
    batch_size=4,
    learning_rate=5e-5,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for memory
    special_tokens = {'additional_special_tokens': ['[MEM]', '[/MEM]']}
    tokenizer.add_special_tokens(special_tokens)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Datasets
    train_dataset = EideticDataset(train_path, tokenizer)
    val_dataset = EideticDataset(val_path, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  Saved best model (val_loss = {val_loss:.4f})")
    
    return model, tokenizer


if __name__ == "__main__":
    train_eidetic_model(num_epochs=10)
