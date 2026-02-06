"""
FDM-style Neural Encoder — Direct channel-to-token mapping

No iMEC. No steganography. Just learnable encoding.

Each channel gets assigned token positions (like FDM frequency bands).
The context bits directly bias token selection at those positions.
The decoder learns to extract each channel from its positions.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class FDMEncoder:
    """
    Frequency Division Multiplexing for token streams.
    
    Each channel owns specific token positions (interleaved).
    Channel 0: positions 0, num_channels, 2*num_channels, ...
    Channel 1: positions 1, num_channels+1, 2*num_channels+1, ...
    
    At each position, the context bits bias which token is selected.
    """
    
    def __init__(self, num_channels=8, tokens_per_channel=64, 
                 bias_strength=2.0, model_name='gpt2'):
        self.num_channels = num_channels
        self.tokens_per_channel = tokens_per_channel
        self.bias_strength = bias_strength
        self.total_tokens = num_channels * tokens_per_channel
        
        print(f"Loading GPT-2 for FDM encoder...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.vocab_size = len(self.tokenizer)
        self.prompt = "The"
        
        print(f"FDMEncoder ready:")
        print(f"  Channels: {num_channels}")
        print(f"  Tokens per channel: {tokens_per_channel}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Bias strength: {bias_strength}")
        print(f"  Device: {self.device}")
    
    def context_to_bits(self, context_string):
        """Convert context string to bits."""
        context_bytes = context_string.encode('utf-8')
        bits = ''.join(format(byte, '08b') for byte in context_bytes)
        return bits
    
    def get_channel_bias(self, bits, token_idx_in_channel, seed=42):
        """
        Generate a token bias vector based on context bits.
        
        Uses the bits to create a deterministic but learnable bias
        over the vocabulary. Different bits = different bias pattern.
        """
        # Use bits + position to seed the bias pattern
        bit_int = int(bits[:min(32, len(bits))], 2) if bits else 0
        rng = np.random.RandomState(seed + bit_int + token_idx_in_channel)
        
        # Create sparse bias: boost ~1000 random tokens based on the bits
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        boosted_tokens = rng.choice(self.vocab_size, size=1000, replace=False)
        bias[boosted_tokens] = self.bias_strength
        
        return bias
    
    def encode(self, contexts):
        """
        Encode multiple contexts into a single token stream using FDM.
        
        Token positions are interleaved:
        - Position 0, 8, 16, 24, ... → Channel 0
        - Position 1, 9, 17, 25, ... → Channel 1
        - etc.
        
        At each position, the token is sampled with bias from that channel's context.
        """
        assert len(contexts) == self.num_channels, \
            f"Expected {self.num_channels} contexts, got {len(contexts)}"
        
        # Convert contexts to bits
        channel_bits = [self.context_to_bits(ctx) for ctx in contexts]
        
        # Pad bits to consistent length
        max_bits = max(len(b) for b in channel_bits)
        channel_bits = [b.ljust(max_bits, '0') for b in channel_bits]
        
        # Start generation
        input_ids = self.tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
        generated_tokens = []
        
        with torch.no_grad():
            for pos in range(self.total_tokens):
                # Which channel owns this position?
                channel_id = pos % self.num_channels
                token_idx_in_channel = pos // self.num_channels
                
                # Get GPT-2's base distribution
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                # Apply channel-specific bias based on context bits
                bits = channel_bits[channel_id]
                bias = self.get_channel_bias(bits, token_idx_in_channel, 
                                            seed=42 + channel_id * 1000)
                
                # Combine logits with bias
                biased_logits = logits + bias
                
                # Sample token
                probs = self._softmax(biased_logits)
                token = np.random.choice(self.vocab_size, p=probs)
                
                generated_tokens.append(token)
                
                # Update context
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
                
                # Keep context manageable
                if input_ids.shape[1] > 512:
                    input_ids = input_ids[:, -512:]
        
        stegotext = self.prompt + self.tokenizer.decode(generated_tokens)
        
        metadata = {
            'num_channels': self.num_channels,
            'tokens_per_channel': self.tokens_per_channel,
            'total_tokens': len(generated_tokens),
            'bias_strength': self.bias_strength,
            'contexts': contexts,
        }
        
        return stegotext, metadata
    
    def _softmax(self, x):
        """Stable softmax."""
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()


# Backwards compatible alias
EideticEncoder = FDMEncoder


if __name__ == "__main__":
    encoder = FDMEncoder(num_channels=8, tokens_per_channel=32, bias_strength=2.0)
    
    contexts = [
        "SECRET:RED",
        "LOCATION:PARIS",
        "AGENT:ALICE",
        "TASK:WAIT",
        "STATUS:READY",
        "NUM:42",
        "FLAG:TRUE",
        "MOOD:HAPPY",
    ]
    
    print(f"\nEncoding {len(contexts)} channels with FDM...")
    stegotext, metadata = encoder.encode(contexts)
    
    print(f"\nStegotext ({metadata['total_tokens']} tokens):")
    print(f"  {stegotext[:300]}...")
    print(f"\nMetadata:")
    print(f"  Channels: {metadata['num_channels']}")
    print(f"  Tokens per channel: {metadata['tokens_per_channel']}")
    print(f"  Total tokens: {metadata['total_tokens']}")
    
    # Show which positions belong to which channel
    print(f"\nChannel position mapping (first 24 positions):")
    for pos in range(24):
        ch = pos % 8
        print(f"  Position {pos:2d} → Channel {ch}")
