"""
Train GPT-2 to decode eidetic channels from stegotext
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from tqdm import tqdm
import json

class EideticDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Format: [STEGO] {stegotext} [CHANNEL] {n} [DECODE] {context}
        prompt = f"[STEGO] {sample['stegotext'][:400]} [CHANNEL] {sample['channel']} [DECODE]"
        target = f" {sample['context']}"
        
        full_text = prompt + target
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Labels: -100 for prompt (don't compute loss), actual ids for target
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_tokens] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class EideticDecoder:
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: str = "cuda"
    ):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": ["[STEGO]", "[CHANNEL]", "[DECODE]"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(
        self,
        train_path: str,
        val_path: str,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 5e-5,
        save_path: str = "eidetic_decoder"
    ):
        # Datasets
        train_dataset = EideticDataset(train_path, self.tokenizer)
        val_dataset = EideticDataset(val_path, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device)
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                    outputs = self.model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["labels"].to(self.device)
                    )
                    val_loss += outputs.loss.item()
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"  Saved best model (val_loss = {val_loss:.4f})")
    
    def decode(
        self,
        stegotext: str,
        channel: int,
        max_new_tokens: int = 50
    ) -> str:
        """Decode a specific channel from stegotext"""
        self.model.eval()
        
        prompt = f"[STEGO] {stegotext[:400]} [CHANNEL] {channel} [DECODE]"
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract just the decoded part
        if "[DECODE]" in output_text:
            decoded = output_text.split("[DECODE]")[-1].strip()
        else:
            decoded = output_text
        
        return decoded
    
    def decode_all_channels(
        self,
        stegotext: str,
        num_channels: int = 8
    ) -> List[str]:
        """Decode all channels from stegotext"""
        return [self.decode(stegotext, ch) for ch in range(num_channels)]


# Train the decoder
def train_eidetic_decoder():
    decoder = EideticDecoder(model_name="gpt2-medium")
    
    decoder.train(
        train_path="eidetic_training_data_train.jsonl",
        val_path="eidetic_training_data_val.jsonl",
        epochs=10,
        batch_size=8,
        lr=5e-5,
        save_path="eidetic_decoder_v1"
    )
    
    return decoder

decoder = train_eidetic_decoder()
