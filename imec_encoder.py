"""
Imperfect iMEC: Embed bits into token stream with tunable leakage

Key insight: Perfect iMEC = undetectable = unlearnable
             Imperfect iMEC = statistical bias = learnable decoder
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Optional

class ImperfectiMEC:
    def __init__(
        self,
        model_name: str = "gpt2",
        epsilon: float = 0.1,  # Imperfection level: 0=perfect, 1=plaintext
        device: str = "cuda"
    ):
        self.epsilon = epsilon
        self.device = device
        
        # Load cover model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.vocab_size = self.tokenizer.vocab_size
    
    def get_cover_distribution(self, context_ids: torch.Tensor) -> torch.Tensor:
        """Get next-token distribution from cover model"""
        with torch.no_grad():
            outputs = self.model(context_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        return probs
    
    def encode_bits(
        self,
        bits: List[int],
        prompt: str = "The",
        max_tokens: int = 1000
    ) -> str:
        """
        Encode bits into stegotext using imperfect iMEC
        
        Args:
            bits: Binary data to hide
            prompt: Starting text
            max_tokens: Maximum tokens to generate
        
        Returns:
            Stegotext containing hidden bits
        """
        # Initialize
        context_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []
        bit_idx = 0
        
        for _ in range(max_tokens):
            if bit_idx >= len(bits):
                break
            
            # Get cover distribution
            cover_probs = self.get_cover_distribution(context_ids)
            
            # Sort tokens by probability
            sorted_probs, sorted_indices = torch.sort(cover_probs[0], descending=True)
            
            # Partition into bit=0 and bit=1 sets
            # Top half = bit 0, bottom half = bit 1 (simplified)
            n_tokens = min(1000, (sorted_probs > 0.001).sum().item())
            half = n_tokens // 2
            
            bit_to_encode = bits[bit_idx]
            
            if bit_to_encode == 0:
                candidate_indices = sorted_indices[:half]
                candidate_probs = sorted_probs[:half]
            else:
                candidate_indices = sorted_indices[half:n_tokens]
                candidate_probs = sorted_probs[half:n_tokens]
            
            # IMPERFECTION: Bias selection toward higher-prob tokens
            # epsilon=0: Pure cover distribution (perfect hiding)
            # epsilon=1: Always pick highest prob in candidate set (obvious)
            
            if self.epsilon > 0:
                # Bias toward top of candidate set
                weights = candidate_probs ** (1 / (1 - self.epsilon + 0.01))
                weights = weights / weights.sum()
            else:
                # Perfect: Use cover distribution within candidates
                weights = candidate_probs / candidate_probs.sum()
            
            # Sample token
            token_idx = torch.multinomial(weights, 1).item()
            selected_token = candidate_indices[token_idx].item()
            
            # Append to context
            generated_tokens.append(selected_token)
            context_ids = torch.cat([
                context_ids,
                torch.tensor([[selected_token]], device=self.device)
            ], dim=1)
            
            bit_idx += 1
        
        # Decode to text
        stegotext = prompt + self.tokenizer.decode(generated_tokens)
        return stegotext
    
    def decode_bits(
        self,
        stegotext: str,
        num_bits: int,
        prompt: str = "The"
    ) -> List[int]:
        """
        Extract bits from stegotext (requires knowing the encoding scheme)
        
        This is the "traditional" decoder - we'll replace this with learned decoder
        """
        tokens = self.tokenizer.encode(stegotext)
        prompt_len = len(self.tokenizer.encode(prompt))
        
        bits = []
        context_ids = torch.tensor([tokens[:prompt_len]], device=self.device)
        
        for i in range(prompt_len, min(prompt_len + num_bits, len(tokens))):
            # Get cover distribution
            cover_probs = self.get_cover_distribution(context_ids)
            
            # Sort tokens
            sorted_probs, sorted_indices = torch.sort(cover_probs[0], descending=True)
            
            # Find which half the actual token is in
            n_tokens = min(1000, (sorted_probs > 0.001).sum().item())
            half = n_tokens // 2
            
            actual_token = tokens[i]
            top_half = sorted_indices[:half].tolist()
            
            if actual_token in top_half:
                bits.append(0)
            else:
                bits.append(1)
            
            # Update context
            context_ids = torch.tensor([tokens[:i+1]], device=self.device)
        
        return bits


# Test iMEC
def test_imec():
    imec = ImperfectiMEC(epsilon=0.2)
    
    # Test bits
    original_bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    
    # Encode
    stegotext = imec.encode_bits(original_bits, prompt="The weather")
    print(f"Stegotext: {stegotext[:200]}...")
    
    # Decode (traditional method)
    decoded_bits = imec.decode_bits(stegotext, len(original_bits), prompt="The weather")
    
    # Compare
    accuracy = sum(a == b for a, b in zip(original_bits, decoded_bits)) / len(original_bits)
    print(f"Original:  {original_bits}")
    print(f"Decoded:   {decoded_bits}")
    print(f"Accuracy:  {accuracy:.1%}")
    
    return stegotext

#test_imec()
