"""
Eidetic Encoder with Channel-Specific Interleavers (Turbo-code inspired)

Key insight: Each channel gets a unique, deterministic interleaving pattern.
This creates channel-specific statistical fingerprints that survive iMEC 
coupling and can be learned by the decoder.

The interleaver IS the learnable signal — not leaked imperfection.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from imec_encoder import MinEntropyCouplingSteganography


class TurboEideticEncoder:
    """
    Multi-channel encoder using channel-specific interleavers.
    
    Each channel's bits are permuted with a unique pattern before
    being concatenated and fed through iMEC. The decoder learns
    to recognize each channel's interleaving signature.
    """
    
    def __init__(self, num_channels=64, block_size_bits=10, epsilon=0.05,
                 max_tokens=1500, model_name='gpt2', interleaver_seed=42):
        self.num_channels = num_channels
        self.block_size_bits = block_size_bits
        self.epsilon = epsilon
        self.max_tokens = max_tokens
        self.interleaver_seed = interleaver_seed
        
        # Use MinEntropyCouplingSteganography from imec_encoder.py
        self.imec = MinEntropyCouplingSteganography(
            model_name=model_name,
            block_size_bits=block_size_bits
        )
        
        self.tokenizer = self.imec.tokenizer
        self.prompt = "The future of artificial intelligence"
        
        print(f"TurboEideticEncoder ready:")
        print(f"  Channels: {num_channels}")
        print(f"  Block size: {block_size_bits} bits ({2**block_size_bits} values/block)")
        print(f"  Epsilon: {epsilon}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Interleaver seed: {interleaver_seed}")
        print(f"  Device: {self.imec.device}")
    
    def channel_interleaver(self, bits, channel_id):
        """
        Each channel gets a unique permutation of bit positions.
        
        This is the key turbo-code insight: the interleaving pattern
        creates a channel-specific fingerprint that the decoder can learn.
        """
        rng = np.random.RandomState(self.interleaver_seed + channel_id)
        indices = rng.permutation(len(bits))
        return ''.join(bits[i] for i in indices)
    
    def channel_deinterleaver(self, bits, channel_id):
        """Inverse of channel_interleaver — for verification/debugging."""
        rng = np.random.RandomState(self.interleaver_seed + channel_id)
        indices = rng.permutation(len(bits))
        # Create inverse permutation
        inverse = np.zeros(len(indices), dtype=int)
        for i, idx in enumerate(indices):
            inverse[idx] = i
        return ''.join(bits[i] for i in inverse)
    
    def context_to_bits(self, context_string):
        """Convert context string to bits (NO OTP — we want patterns)."""
        context_bytes = context_string.encode('utf-8')
        bits = ''.join(format(byte, '08b') for byte in context_bytes)
        return bits
    
    def pad_bits(self, bits, target_length):
        """Pad bits to target length for consistent interleaving."""
        if len(bits) >= target_length:
            return bits[:target_length]
        return bits + '0' * (target_length - len(bits))
    
    def encode(self, contexts):
        """
        Encode multiple channel contexts into a single stegotext.
        
        Each channel's bits are:
        1. Converted from context string
        2. Padded to uniform length
        3. Interleaved with channel-specific pattern
        4. Concatenated with all other channels
        5. Fed through iMEC
        
        Args:
            contexts: List of context strings, one per channel.
        
        Returns:
            stegotext: The generated stegotext string
            metadata: Dict with encoding parameters
        """
        assert len(contexts) == self.num_channels, \
            f"Expected {self.num_channels} contexts, got {len(contexts)}"
        
        # Find max bit length for padding
        all_raw_bits = [self.context_to_bits(ctx) for ctx in contexts]
        max_bits = max(len(b) for b in all_raw_bits)
        # Round up to multiple of 8 for clean bytes
        padded_length = ((max_bits + 7) // 8) * 8
        
        # Interleave each channel's bits with unique pattern
        all_bits = ""
        channel_bit_lengths = []
        
        for ch_idx, (ctx, raw_bits) in enumerate(zip(contexts, all_raw_bits)):
            # Pad to uniform length
            padded = self.pad_bits(raw_bits, padded_length)
            # Apply channel-specific interleaver
            interleaved = self.channel_interleaver(padded, ch_idx)
            all_bits += interleaved
            channel_bit_lengths.append(len(interleaved))
        
        # Pad total to block boundary
        remainder = len(all_bits) % self.block_size_bits
        if remainder != 0:
            padding = self.block_size_bits - remainder
            all_bits += '0' * padding
        
        n_blocks = len(all_bits) // self.block_size_bits
        
        # Use iMEC encoder
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
            'padded_length': padded_length,
            'channel_bit_lengths': channel_bit_lengths,
            'num_tokens': len(stegotext_tokens),
            'epsilon': self.epsilon,
            'interleaver_seed': self.interleaver_seed,
            'contexts': contexts,
        }
        
        return stegotext, metadata


# Backwards compatibility alias
EideticEncoder = TurboEideticEncoder


if __name__ == "__main__":
    encoder = TurboEideticEncoder(num_channels=8, epsilon=0.05, max_tokens=600)
    
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
    
    print(f"\nEncoding {len(contexts)} channels with channel-specific interleavers...")
    stegotext, metadata = encoder.encode(contexts)
    
    print(f"\nStegotext ({metadata['num_tokens']} tokens):")
    print(f"  {stegotext[:200]}...")
    print(f"\nMetadata:")
    print(f"  Total bits: {metadata['total_bits']}")
    print(f"  Blocks: {metadata['n_blocks']}")
    print(f"  Tokens: {metadata['num_tokens']}")
    print(f"  Padded length per channel: {metadata['padded_length']}")
    
    # Verify interleaver is deterministic
    print(f"\nVerifying interleaver determinism...")
    bits1 = encoder.context_to_bits("SECRET:RED")
    bits1_padded = encoder.pad_bits(bits1, metadata['padded_length'])
    interleaved = encoder.channel_interleaver(bits1_padded, 0)
    deinterleaved = encoder.channel_deinterleaver(interleaved, 0)
    print(f"  Original:      {bits1_padded[:40]}...")
    print(f"  Interleaved:   {interleaved[:40]}...")
    print(f"  Deinterleaved: {deinterleaved[:40]}...")
    print(f"  Roundtrip OK:  {bits1_padded == deinterleaved}")
