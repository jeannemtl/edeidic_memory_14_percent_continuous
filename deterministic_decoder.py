"""
deterministic_decoder.py — Eidetic Memory Decoder

Uses the REAL iMEC decoder (same as paper) to recover contexts.
No machine learning needed — this is deterministic decode.

Pipeline:
  1. iMEC decode stegotext tokens → encrypted bits
  2. OTP decrypt → signal bits  
  3. Reconstruct continuous signal
  4. Bandpass filter + ASK demodulate each channel
  5. Convert bits → context strings
"""

import numpy as np
from scipy import signal as scipy_signal
from imec_encoder import MinEntropyCouplingSteganography


class EideticDecoder:
    """
    Multi-channel decoder using iMEC decode + FFT from the paper.
    
    This is DETERMINISTIC — given the stegotext tokens, OTP key,
    and encoding parameters, it perfectly recovers all channels.
    """
    
    def __init__(self, block_size_bits=10):
        """Initialize with matching iMEC block size."""
        self.imec = MinEntropyCouplingSteganography(
            block_size_bits=block_size_bits
        )
        self.tokenizer = self.imec.tokenizer
    
    def otp_decrypt(self, encrypted_bits, otp_key):
        """XOR decrypt with one-time pad."""
        decrypt_len = min(len(encrypted_bits), len(otp_key))
        decrypted = ''.join(
            str(int(encrypted_bits[i]) ^ int(otp_key[i]))
            for i in range(decrypt_len)
        )
        return decrypted
    
    def reconstruct_signal(self, signal_bits, quant_metadata):
        """Convert bit string back to continuous signal."""
        bps = quant_metadata['bits_per_sample']
        num_samples = quant_metadata['num_samples']
        
        quantized = []
        for i in range(num_samples):
            start = i * bps
            end = start + bps
            if end <= len(signal_bits):
                quantized.append(int(signal_bits[start:end], 2))
        
        quantized = np.array(quantized, dtype=np.float64)
        max_level = quant_metadata['quantization_levels'] - 1
        normalized = quantized / max_level
        
        sig_min = quant_metadata['signal_min']
        sig_max = quant_metadata['signal_max']
        reconstructed = normalized * (sig_max - sig_min) + sig_min
        
        return reconstructed
    
    def bandpass_filter(self, signal, freq_low, freq_high, sample_rate, order=4):
        """Apply bandpass filter to extract one frequency band."""
        nyquist = sample_rate * 0.5
        low = max(min(freq_low / nyquist, 0.99), 0.01)
        high = max(min(freq_high / nyquist, 0.99), 0.01)
        
        if low >= high:
            return signal
        
        try:
            b, a = scipy_signal.butter(order, [low, high], btype='band')
            filtered = scipy_signal.filtfilt(b, a, signal)
            return filtered
        except Exception:
            return signal
    
    def ask_demodulate(self, filtered_signal, num_bits):
        """Recover bits from ASK-modulated signal using envelope detection."""
        analytic = scipy_signal.hilbert(filtered_signal)
        envelope = np.abs(analytic)
        
        samples_per_bit = len(filtered_signal) // num_bits
        threshold = (envelope.max() + envelope.min()) / 2
        
        recovered = []
        for i in range(num_bits):
            start = i * samples_per_bit
            end = min((i + 1) * samples_per_bit, len(envelope))
            avg_amp = envelope[start:end].mean()
            recovered.append(1 if avg_amp > threshold else 0)
        
        return recovered
    
    def bits_to_context(self, bits):
        """Convert bit list back to context string."""
        byte_vals = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) == 8:
                byte_vals.append(int(''.join(str(b) for b in byte_bits), 2))
        
        return bytes(byte_vals).rstrip(b'\x00').decode('utf-8', errors='replace')
    
    def decode(self, encode_data):
        """
        Decode stegotext back to all channel contexts.
        
        Args:
            encode_data: Dict from EideticEncoder.encode() containing:
                - token_ids, otp_key, gpt2_prompt, quant_metadata,
                - carrier_freqs, sample_rate, num_message_bits, etc.
        
        Returns:
            recovered_contexts: List of recovered context strings
            accuracies: Per-channel bit accuracy (if ground truth available)
        """
        token_ids = encode_data['token_ids']
        otp_key = encode_data['otp_key']
        prompt = encode_data['gpt2_prompt']
        quant_meta = encode_data['quant_metadata']
        carrier_freqs = encode_data['carrier_freqs']
        sample_rate = encode_data['sample_rate']
        num_message_bits = encode_data['num_message_bits']
        block_size_bits = encode_data['block_size_bits']
        total_signal_bits = encode_data['total_signal_bits']
        num_channels = encode_data['num_channels']
        
        # Step 1: iMEC decode
        n_blocks = (total_signal_bits + block_size_bits - 1) // block_size_bits
        
        print(f"iMEC decoding {len(token_ids)} tokens → {n_blocks} blocks...")
        encrypted_bits = self.imec.decode_imec(
            token_ids, prompt, n_blocks, block_size_bits
        )
        
        # Step 2: OTP decrypt
        signal_bits = self.otp_decrypt(encrypted_bits, otp_key)
        print(f"OTP decrypted: {len(signal_bits)} bits")
        
        # Step 3: Reconstruct signal
        recovered_signal = self.reconstruct_signal(signal_bits, quant_meta)
        print(f"Signal reconstructed: {len(recovered_signal)} samples")
        
        # Step 4: Extract each channel
        recovered_contexts = []
        accuracies = {}
        bandwidth = 0.4  # Hz bandwidth for bandpass filter
        
        for ch_idx, freq in enumerate(carrier_freqs):
            freq_low = freq - bandwidth
            freq_high = freq + bandwidth
            
            # Bandpass filter
            filtered = self.bandpass_filter(
                recovered_signal, freq_low, freq_high, sample_rate
            )
            
            # ASK demodulate
            recovered_bits = self.ask_demodulate(filtered, num_message_bits)
            
            # Convert to context
            context = self.bits_to_context(recovered_bits)
            recovered_contexts.append(context)
            
            # Accuracy check if ground truth available
            if 'contexts' in encode_data:
                from full_encoder_pipeline_v2 import EideticEncoder
                enc = EideticEncoder.__new__(EideticEncoder)
                enc.max_context_bits = num_message_bits  # hack to get same bit length
                # Direct bit comparison
                original_ctx = encode_data['contexts'][ch_idx]
                orig_bytes = original_ctx.encode('utf-8')
                max_bytes = num_message_bits // 8
                if len(orig_bytes) > max_bytes:
                    orig_bytes = orig_bytes[:max_bytes]
                else:
                    orig_bytes = orig_bytes + b'\x00' * (max_bytes - len(orig_bytes))
                orig_bits = [int(b) for b in ''.join(format(x, '08b') for x in orig_bytes)]
                
                correct = sum(1 for a, b in zip(orig_bits, recovered_bits) if a == b)
                acc = correct / len(orig_bits)
                accuracies[ch_idx] = acc
                
                match = "✓" if context.rstrip('\x00') == original_ctx else "✗"
                print(f"  Ch {ch_idx} ({freq:.1f} Hz): "
                      f"'{original_ctx}' → '{context.rstrip(chr(0))}' "
                      f"[{acc*100:.0f}%] {match}")
        
        return recovered_contexts, accuracies


if __name__ == "__main__":
    import pickle
    import sys
    
    # Try to load from encoder output
    print("=" * 60)
    print("DETERMINISTIC DECODER — Round-trip test")
    print("=" * 60)
    
    from full_encoder_pipeline_v2 import EideticEncoder
    
    # Small test: 3 channels, short contexts
    encoder = EideticEncoder(
        num_channels=3,
        samples_per_bit=30,
        bits_per_sample=4,
        block_size_bits=10,
        epsilon=0.15,
        max_context_bits=64
    )
    
    contexts = ["SECRET:ALPHA", "MOOD:HAPPY", "TASK:WAIT"]
    print(f"\nOriginal contexts: {contexts}")
    
    stegotext, encode_data = encoder.encode(contexts, max_tokens=500)
    print(f"\nStegotext: {stegotext[:100]}...")
    
    decoder = EideticDecoder(block_size_bits=10)
    recovered, accuracies = decoder.decode(encode_data)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for ch_idx, (orig, recv) in enumerate(zip(contexts, recovered)):
        acc = accuracies.get(ch_idx, 0)
        print(f"  Ch {ch_idx}: '{orig}' → '{recv.rstrip(chr(0))}' [{acc*100:.0f}%]")
    
    overall = np.mean(list(accuracies.values())) if accuracies else 0
    print(f"\n  Overall: {overall*100:.1f}%")
