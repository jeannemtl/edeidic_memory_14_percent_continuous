"""
Eidetic Encoder using MinEntropyCouplingSteganography from imec_encoder.py
"""

import numpy as np
from imec_encoder import MinEntropyCouplingSteganography


class EideticEncoder:
    """
    Multi-channel encoder using real iMEC from imec_encoder.py.
    
    Each channel's context string is encoded into the stegotext via iMEC
    WITHOUT OTP encryption, so the non-uniform bit patterns leak through.
    """
    
    def __init__(self, num_channels=8, block_size_bits=10, epsilon=0.15,
                 max_tokens=600, model_name='gpt2'):
        self.num_channels = num_channels
        self.block_size_bits = block_size_bits
        self.epsilon = epsilon
        self.max_tokens = max_tokens
        
        # Use MinEntropyCouplingSteganography from imec_encoder.py
        self.imec = MinEntropyCouplingSteganography(
            model_name=model_name,
            block_size_bits=block_size_bits
        )
        
        self.tokenizer = self.imec.tokenizer
        self.prompt = "The future of artificial intelligence"
        
        print(f"EideticEncoder ready:")
        print(f"  Channels: {num_channels}")
        print(f"  Block size: {block_size_bits} bits ({2**block_size_bits} values/block)")
        print(f"  Epsilon: {epsilon}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Device: {self.imec.device}")
    
    def context_to_bits(self, context_string):
        """Convert context string to bits (NO OTP — deliberate leakage)."""
        context_bytes = context_string.encode('utf-8')
        bits = ''.join(format(byte, '08b') for byte in context_bytes)
        return bits
    
    def encode(self, contexts):
        """
        Encode multiple channel contexts into a single stegotext.
        
        Args:
            contexts: List of context strings, one per channel.
        
        Returns:
            stegotext: The generated stegotext string
            metadata: Dict with encoding parameters
        """
        assert len(contexts) == self.num_channels, \
            f"Expected {self.num_channels} contexts, got {len(contexts)}"
        
        # Convert all contexts to bits and concatenate
        all_bits = ""
        channel_bit_lengths = []
        for ctx in contexts:
            bits = self.context_to_bits(ctx)
            channel_bit_lengths.append(len(bits))
            all_bits += bits
        
        # Pad to block boundary
        remainder = len(all_bits) % self.block_size_bits
        if remainder != 0:
            padding = self.block_size_bits - remainder
            all_bits += '0' * padding
        
        n_blocks = len(all_bits) // self.block_size_bits
        
        # Use iMEC encoder from imec_encoder.py
        stegotext_tokens = self.imec.encode_imec(
            all_bits,
            self.prompt,
            max_tokens=self.max_tokens,
            entropy_threshold=self.epsilon
        )
        
        stegotext = self.tokenizer.decode(stegotext_tokens)
        
        metadata = {
            'num_channels': self.num_channels,
            'block_size_bits': self.block_size_bits,
            'n_blocks': n_blocks,
            'total_bits': len(all_bits),
            'channel_bit_lengths': channel_bit_lengths,
            'num_tokens': len(stegotext_tokens),
            'epsilon': self.epsilon,
            'contexts': contexts,
        }
        
        return stegotext, metadata


if __name__ == "__main__":
    encoder = EideticEncoder(num_channels=8, epsilon=0.15, max_tokens=600)
    
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
    
    print(f"\nEncoding {len(contexts)} channels...")
    stegotext, metadata = encoder.encode(contexts)
    
    print(f"\nStegotext ({metadata['num_tokens']} tokens):")
    print(f"  {stegotext[:200]}...")
    print(f"\nMetadata:")
    print(f"  Total bits: {metadata['total_bits']}")
    print(f"  Blocks: {metadata['n_blocks']}")
    print(f"  Tokens: {metadata['num_tokens']}")
