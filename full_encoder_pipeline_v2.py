"""
full_encoder_pipeline_v2.py — Eidetic Memory Encoder

Uses the REAL iMEC encoder from the paper (imec_encoder.py).
Encodes N channel contexts into a single stegotext stream via FDM + iMEC.

Pipeline per sample:
  1. Convert each context string to bits
  2. ASK-modulate each channel at distinct carrier frequencies  
  3. Superpose (FDM)
  4. Quantize to bits
  5. OTP encrypt (makes distribution uniform — required for iMEC)
  6. iMEC encode to GPT-2 stegotext

Output: (stegotext, token_ids, otp_key, metadata) — everything needed for decode
"""

import numpy as np
import torch
import hashlib
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from imec_encoder import MinEntropyCouplingSteganography


class EideticEncoder:
    """
    Multi-channel encoder using FDM + iMEC from the paper.
    
    Each "channel" carries an independent context string.
    All channels are frequency-multiplexed into one signal,
    then hidden in GPT-2 stegotext via iMEC.
    """
    
    def __init__(self, num_channels=8, sample_rate=100.0, 
                 bits_per_sample=4, samples_per_bit=50,
                 block_size_bits=10, epsilon=0.15,
                 max_context_bits=128):
        """
        Args:
            num_channels: Number of independent channels (default 8)
            sample_rate: FDM sampling rate in Hz
            bits_per_sample: Quantization depth (4=16 levels, keeps signal short)
            samples_per_bit: Samples per message bit (affects FFT resolution)
            block_size_bits: iMEC block size (10 = 1024 values, fast encoding)
            epsilon: iMEC entropy threshold for convergence
            max_context_bits: Max bits per context string
        """
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self.samples_per_bit = samples_per_bit
        self.block_size_bits = block_size_bits
        self.epsilon = epsilon
        self.max_context_bits = max_context_bits
        
        # Carrier frequencies: evenly spaced, well-separated
        # Start at 1.0 Hz, spacing = 1.0 Hz
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(num_channels)]
        
        # Initialize iMEC (loads GPT-2 once, reuses)
        print(f"Initializing EideticEncoder with {num_channels} channels...")
        print(f"  Carriers: {self.carrier_freqs} Hz")
        print(f"  Block size: {block_size_bits} bits")
        print(f"  Epsilon: {epsilon}")
        
        self.imec = MinEntropyCouplingSteganography(
            block_size_bits=block_size_bits
        )
        self.tokenizer = self.imec.tokenizer
    
    def context_to_bits(self, context_str):
        """Convert context string to fixed-length bit string."""
        context_bytes = context_str.encode('utf-8')
        
        # Pad or truncate to fixed length
        max_bytes = self.max_context_bits // 8
        if len(context_bytes) > max_bytes:
            context_bytes = context_bytes[:max_bytes]
        else:
            context_bytes = context_bytes + b'\x00' * (max_bytes - len(context_bytes))
        
        bits = ''.join(format(b, '08b') for b in context_bytes)
        return [int(b) for b in bits]
    
    def bits_to_context(self, bits):
        """Convert bit list back to context string."""
        byte_vals = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) == 8:
                byte_vals.append(int(''.join(str(b) for b in byte_bits), 2))
        
        return bytes(byte_vals).rstrip(b'\x00').decode('utf-8', errors='replace')
    
    def ask_modulate(self, bits, freq, num_samples):
        """ASK modulation: bit 1 → full amplitude, bit 0 → low amplitude."""
        samples_per_bit = num_samples // len(bits)
        signal = np.zeros(num_samples)
        
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = min((i + 1) * samples_per_bit, num_samples)
            
            amplitude = 1.0 if bit == 1 else 0.1
            duration = (end - start) / self.sample_rate
            t = np.linspace(0, duration, end - start, endpoint=False)
            carrier = np.sin(2 * np.pi * freq * t)
            signal[start:end] = amplitude * carrier
        
        return signal
    
    def quantize_signal(self, signal):
        """Quantize continuous signal to bit string."""
        sig_min = float(signal.min())
        sig_max = float(signal.max())
        sig_range = sig_max - sig_min
        if sig_range < 1e-10:
            sig_range = 1.0
        
        normalized = (signal - sig_min) / sig_range
        max_level = (2 ** self.bits_per_sample) - 1
        quantized = np.round(normalized * max_level).astype(np.uint16)
        
        bits_str = ''.join(
            format(int(val), f'0{self.bits_per_sample}b') 
            for val in quantized
        )
        
        metadata = {
            'num_samples': len(signal),
            'bits_per_sample': self.bits_per_sample,
            'signal_min': sig_min,
            'signal_max': sig_max,
            'quantization_levels': max_level + 1
        }
        
        return bits_str, metadata
    
    def otp_encrypt(self, plaintext_bits):
        """XOR encrypt with random one-time pad → uniform distribution."""
        key = np.random.randint(0, 2, len(plaintext_bits), dtype=np.uint8)
        ciphertext = ''.join(
            str(int(plaintext_bits[i]) ^ int(key[i]))
            for i in range(len(plaintext_bits))
        )
        return ciphertext, key
    
    def encode(self, contexts, gpt2_prompt="The future of artificial intelligence",
               max_tokens=2000):
        """
        Encode list of context strings into stegotext.
        
        Args:
            contexts: List of N context strings (one per channel)
            gpt2_prompt: Initial prompt for GPT-2 stegotext generation
            max_tokens: Maximum tokens for iMEC encoding
            
        Returns:
            stegotext: Generated text hiding all channels
            encode_data: Dict with everything needed for decoding
        """
        assert len(contexts) == self.num_channels, \
            f"Expected {self.num_channels} contexts, got {len(contexts)}"
        
        # Step 1: Convert contexts to bits
        channel_bits = [self.context_to_bits(ctx) for ctx in contexts]
        num_message_bits = len(channel_bits[0])
        
        # Step 2: Compute signal length
        num_samples = num_message_bits * self.samples_per_bit
        
        # Step 3: ASK modulate each channel at its carrier frequency
        signals = []
        for ch_idx, (bits, freq) in enumerate(
            zip(channel_bits, self.carrier_freqs)
        ):
            sig = self.ask_modulate(bits, freq, num_samples)
            signals.append(sig)
        
        # Step 4: Superpose (FDM)
        combined = sum(signals)
        
        # Step 5: Quantize
        signal_bits, quant_meta = self.quantize_signal(combined)
        
        # Step 6: OTP encrypt
        encrypted_bits, otp_key = self.otp_encrypt(signal_bits)
        
        # Step 7: iMEC encode
        token_ids = self.imec.encode_imec(
            encrypted_bits,
            gpt2_prompt,
            max_tokens=max_tokens,
            entropy_threshold=self.epsilon
        )
        
        stegotext = self.tokenizer.decode(token_ids)
        
        # Bundle everything needed for decoding
        encode_data = {
            'token_ids': token_ids,
            'otp_key': otp_key,
            'gpt2_prompt': gpt2_prompt,
            'quant_metadata': quant_meta,
            'num_channels': self.num_channels,
            'carrier_freqs': self.carrier_freqs,
            'sample_rate': self.sample_rate,
            'num_message_bits': num_message_bits,
            'block_size_bits': self.block_size_bits,
            'total_signal_bits': len(signal_bits),
            'contexts': contexts,  # ground truth for verification
            'stegotext': stegotext,
        }
        
        return stegotext, encode_data


if __name__ == "__main__":
    # Quick test with 3 channels (faster than 8)
    encoder = EideticEncoder(
        num_channels=3,
        samples_per_bit=30,   # smaller for speed
        bits_per_sample=4,    # 16 levels
        block_size_bits=10,   # 1024 values per block
        epsilon=0.15,
        max_context_bits=64   # 8 bytes per context
    )
    
    contexts = ["SECRET:ALPHA", "MOOD:HAPPY", "TASK:WAIT"]
    
    print(f"\nEncoding {len(contexts)} contexts...")
    stegotext, data = encoder.encode(contexts, max_tokens=500)
    
    print(f"\nStegotext ({len(data['token_ids'])} tokens):")
    print(f"  {stegotext[:150]}...")
    print(f"\nContexts: {contexts}")
    print(f"Signal bits: {data['total_signal_bits']}")
    print(f"OTP key length: {len(data['otp_key'])}")
