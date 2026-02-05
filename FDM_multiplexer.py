"""
FDM Multiplexer: Combine N contexts into one signal
Each context modulated at a different frequency
"""

import numpy as np
from typing import List

class FDMMultiplexer:
    def __init__(self, num_channels=8, samples_per_bit=100, base_frequency=1.0):
        self.num_channels = num_channels
        self.samples_per_bit = samples_per_bit
        self.frequencies = [base_frequency * (i + 1) for i in range(num_channels)]
        self.max_bits = None

    def text_to_bits(self, text):
        bits = []
        for char in text:
            bits.extend([int(b) for b in format(ord(char), '08b')])
        return bits

    def bits_to_text(self, bits):
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) == 8:
                code = int(''.join(map(str, byte)), 2)
                if 0 < code < 128:
                    chars.append(chr(code))
        return ''.join(chars)

    def modulate_channel(self, bits, frequency, num_samples):
        signal = np.zeros(num_samples)
        samples_per_bit = num_samples // len(bits)
        t = np.linspace(0, len(bits), num_samples)
        carrier = np.sin(2 * np.pi * frequency * t)
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            signal[start:end] = bit * carrier[start:end]
        return signal

    def encode(self, contexts):
        assert len(contexts) <= self.num_channels
        all_bits = [self.text_to_bits(ctx) for ctx in contexts]
        self.max_bits = max(len(b) for b in all_bits)
        for bits in all_bits:
            while len(bits) < self.max_bits:
                bits.append(0)
        num_samples = self.max_bits * self.samples_per_bit
        combined = np.zeros(num_samples)
        for i, (bits, freq) in enumerate(zip(all_bits, self.frequencies)):
            combined += self.modulate_channel(bits, freq, num_samples)
        return combined

    def decode_channel(self, signal, channel, num_bits=None):
        if num_bits is None:
            num_bits = self.max_bits
        freq = self.frequencies[channel]
        num_samples = len(signal)
        samples_per_bit = num_samples // num_bits
        t = np.linspace(0, num_bits, num_samples)
        carrier = np.sin(2 * np.pi * freq * t)
        demodulated = signal * carrier
        bits = []
        for i in range(num_bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            energy = np.mean(demodulated[start:end])
            bits.append(1 if energy > 0.25 else 0)
        return bits

def test_fdm():
    fdm = FDMMultiplexer(num_channels=4)
    contexts = ["SECRET:RED", "AGENT:ALICE", "TASK:WAIT", "MOOD:HAPPY"]
    signal = fdm.encode(contexts)
    print(f"Signal shape: {signal.shape}")
    print(f"Max bits: {fdm.max_bits}")
    for ch, original in enumerate(contexts):
        bits = fdm.decode_channel(signal, ch)
        decoded = fdm.bits_to_text(bits)
        match = original in decoded
        print(f"Channel {ch}: '{original}' -> '{decoded}' {'✓' if match else '✗'}")

if __name__ == "__main__":
    test_fdm()
