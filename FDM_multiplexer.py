"""
FDM Multiplexer: Combine N contexts into one signal
Each context modulated at a different frequency
"""

import numpy as np
from typing import List, Tuple

class FDMMultiplexer:
    def __init__(
        self, 
        num_channels: int = 8,
        samples_per_bit: int = 100,
        base_frequency: float = 1.0
    ):
        self.num_channels = num_channels
        self.samples_per_bit = samples_per_bit
        # Frequencies: 1, 2, 3, ... N Hz
        self.frequencies = [base_frequency * (i + 1) for i in range(num_channels)]
    
    def text_to_bits(self, text: str) -> List[int]:
        """Convert text to binary representation"""
        bits = []
        for char in text:
            char_bits = format(ord(char), '08b')
            bits.extend([int(b) for b in char_bits])
        return bits
    
    def bits_to_text(self, bits: List[int]) -> str:
        """Convert binary back to text"""
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) == 8:
                char_code = int(''.join(map(str, byte)), 2)
                if 0 < char_code < 128:  # Valid ASCII
                    chars.append(chr(char_code))
        return ''.join(chars)
    
    def modulate_channel(
        self, 
        bits: List[int], 
        frequency: float,
        num_samples: int
    ) -> np.ndarray:
        """ASK modulate bits at given frequency"""
        signal = np.zeros(num_samples)
        samples_per_bit = num_samples // len(bits)
        
        t = np.linspace(0, len(bits), num_samples)
        carrier = np.sin(2 * np.pi * frequency * t)
        
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            signal[start:end] = bit * carrier[start:end]
        
        return signal
    
    def encode(self, contexts: List[str]) -> np.ndarray:
        """
        Encode multiple contexts into single FDM signal
        
        Args:
            contexts: List of N strings (one per channel)
        
        Returns:
            Combined FDM signal
        """
        assert len(contexts) <= self.num_channels
        
        # Convert all to bits, find max length
        all_bits = [self.text_to_bits(ctx) for ctx in contexts]
        max_bits = max(len(b) for b in all_bits)
        
        # Pad shorter contexts
        for bits in all_bits:
            while len(bits) < max_bits:
                bits.append(0)
        
        # Total samples needed
        num_samples = max_bits * self.samples_per_bit
        
        # Modulate each channel and sum
        combined = np.zeros(num_samples)
        for i, (bits, freq) in enumerate(zip(all_bits, self.frequencies)):
            channel_signal = self.modulate_channel(bits, freq, num_samples)
            combined += channel_signal
        
        return combined
    
    def decode_channel(
        self, 
        signal: np.ndarray, 
        channel: int,
        num_bits: int
    ) -> List[int]:
        """
        Extract one channel via bandpass filter + demodulation
        
        Args:
            signal: Combined FDM signal
            channel: Channel index (0-based)
            num_bits: Expected number of bits
        
        Returns:
            Decoded bits for that channel
        """
        freq = self.frequencies[channel]
        num_samples = len(signal)
        samples_per_bit = num_samples // num_bits
        
        # Generate reference carrier
        t = np.linspace(0, num_bits, num_samples)
        carrier = np.sin(2 * np.pi * freq * t)
        
        # Multiply by carrier (demodulate)
        demodulated = signal * carrier
        
        # Integrate over each bit period
        bits = []
        for i in range(num_bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            energy = np.mean(demodulated[start:end])
            bits.append(1 if energy > 0.25 else 0)
        
        return bits


# Test FDM
def test_fdm():
    fdm = FDMMultiplexer(num_channels=4)
    
    contexts = [
        "SECRET:RED",
        "AGENT:ALICE", 
        "TASK:WAIT",
        "MOOD:HAPPY"
    ]
    
    # Encode
    signal = fdm.encode(contexts)
    print(f"Signal shape: {signal.shape}")
    
    # Decode each channel
    for ch, original in enumerate(contexts):
        bits = fdm.decode_channel(signal, ch, len(original) * 8)
        decoded = fdm.bits_to_text(bits)
        print(f"Channel {ch}: '{original}' -> '{decoded}'")
        
    return signal

test_fdm()
