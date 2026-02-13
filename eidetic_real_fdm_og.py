"""
Eidetic Memory with REAL Frequency Division Multiplexing

This uses the actual FDM architecture from the paper:
  - 8 channels encoded as ASK-modulated sinusoidal carriers
  - Superposition into a single composite signal
  - Quantization → OTP encryption → iMEC encoding into GPT-2 tokens
  - Decode: iMEC → OTP decrypt → reconstruct → FFT → bandpass → ASK demodulate
  - Feed decoded channel values to reasoning logic

This is TRUE FDM: every time sample carries information from ALL 8 channels
simultaneously, separated by frequency. Compare to TDM baseline (256 tokens,
one channel per position).

Usage:
  python eidetic_real_fdm.py encode          # Encode one sample, show token count
  python eidetic_real_fdm.py roundtrip       # Encode + decode, verify recovery
  python eidetic_real_fdm.py eval N          # Encode N samples, full eval with reasoning
  python eidetic_real_fdm.py compare         # Side-by-side comparison with TDM
"""

import numpy as np
import torch
import json
import random
import sys
import os
import pickle
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

# Import iMEC from same directory
from imec_encoder import MinEntropyCouplingSteganography


# ============================================================
# Channel definitions (same as TDM experiment)
# ============================================================

MEMORY_SCHEMAS = {
    0: ("SECRET",   ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",    ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",   ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",   ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",     ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",     ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
}

NUM_CHANNELS = 8


# ============================================================
# Reasoning logic (identical to TDM experiment)
# ============================================================

def generate_proceed_answer(facts, rule, meta):
    agent, status, priority, backup = facts["AGENT"], facts["STATUS"], facts["PRIORITY"], facts["BACKUP"]
    if meta == "EMERGENCY":
        return f"EMERGENCY PROTOCOL: {agent} must proceed immediately. All other factors suspended."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN ACTIVE: {agent} must abort all operations. No exceptions."
    if meta == "OVERRIDE_STATUS":
        if priority == "HIGH":
            return f"Status override active. Priority is {priority}, so {agent} should proceed."
        else:
            return f"Status override active, but priority is only {priority}. {agent} may proceed with caution."
    if meta == "OVERRIDE_PRIORITY":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"Priority override active. Status {status} with backup {backup}. {agent} can proceed."
        else:
            return f"Priority override active. Status {status}, backup {backup}. {agent} should hold."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"SAFETY_FIRST: Status is {status}. {agent} must abort regardless of {priority} priority."
        elif status == "CLEAR":
            return f"SAFETY_FIRST: Status is {status}. {agent} can proceed with {priority} priority."
        else:
            return f"SAFETY_FIRST: Status is {status}. {agent} should wait for confirmation."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return f"MISSION_FIRST: Status {status} with no backup. Even mission-priority says {agent} should abort."
        else:
            return f"MISSION_FIRST: Priority is {priority}. {agent} should proceed. Status {status} is secondary."
    elif rule == "BALANCED":
        if priority == "HIGH" and backup == "AVAILABLE":
            return f"BALANCED: {priority} priority with backup {backup} outweighs {status} status. {agent} can proceed."
        elif status == "COMPROMISED":
            return f"BALANCED: {status} status not offset by {priority} priority. {agent} should abort."
        else:
            return f"BALANCED: Status {status}, priority {priority}. {agent} can proceed carefully."
    elif rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"CAUTIOUS: Status {status} AND backup {backup}. Both conditions met. {agent} can proceed."
        else:
            return f"CAUTIOUS: Need CLEAR status AND AVAILABLE backup. Have {status}/{backup}. {agent} must abort."
    return f"{agent} should assess situation."


def generate_risk_answer(facts, rule, meta):
    status, priority, backup, location = facts["STATUS"], facts["PRIORITY"], facts["BACKUP"], facts["LOCATION"]
    if meta == "EMERGENCY":
        return f"EMERGENCY: Risk assessment suspended. Immediate action required in {location}."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Maximum risk assumed. All operations in {location} halted."
    risk_factors = []
    if status == "COMPROMISED": risk_factors.append("compromised status")
    if backup == "UNAVAILABLE": risk_factors.append("no backup")
    if status == "UNKNOWN": risk_factors.append("unknown status")
    if rule == "SAFETY_FIRST":
        if risk_factors:
            return f"SAFETY_FIRST assessment: HIGH RISK in {location}. Factors: {', '.join(risk_factors)}."
        return f"SAFETY_FIRST assessment: LOW RISK in {location}. Status {status}, backup {backup}."
    elif rule == "MISSION_FIRST":
        if len(risk_factors) >= 2:
            return f"MISSION_FIRST assessment: MODERATE RISK in {location}. Acceptable for {priority} priority."
        return f"MISSION_FIRST assessment: LOW RISK in {location}. Proceed with mission."
    elif rule == "BALANCED":
        level = "HIGH" if len(risk_factors) >= 2 else "MEDIUM" if len(risk_factors) == 1 else "LOW"
        return f"BALANCED assessment: {level} RISK in {location}. Factors: {len(risk_factors)} concerns."
    elif rule == "CAUTIOUS":
        if risk_factors:
            return f"CAUTIOUS assessment: HIGH RISK in {location}. Any risk factor triggers alert: {', '.join(risk_factors)}."
        return f"CAUTIOUS assessment: LOW RISK in {location}. All safety conditions met."
    return f"Risk assessment for {location}."


def generate_share_answer(facts, rule, meta):
    secret, status, location = facts["SECRET"], facts["STATUS"], facts["LOCATION"]
    if meta == "EMERGENCY":
        return f"EMERGENCY: Share {secret} immediately with any allied contact. Speed over security."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Do not share {secret} under any circumstances. Maintain radio silence."
    if meta == "OVERRIDE_STATUS":
        return f"Status override active. You may share {secret} with {location} contact despite {status} status."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"SAFETY_FIRST: Do NOT share {secret}. {location} may be compromised."
        return f"SAFETY_FIRST: Status {status}. May share {secret} with verified contacts only."
    elif rule == "MISSION_FIRST":
        return f"MISSION_FIRST: Share {secret} with {location} contact to advance mission. Accept calculated risk."
    elif rule == "BALANCED":
        if status == "CLEAR":
            return f"BALANCED: Status {status}. Safe to share {secret} with {location} contact."
        return f"BALANCED: Status {status}. Share {secret} only if mission-critical."
    elif rule == "CAUTIOUS":
        if status == "CLEAR":
            return f"CAUTIOUS: Status {status}. May share {secret} after secondary verification."
        return f"CAUTIOUS: Status {status}. Do not share {secret}. Request secure channel."
    return f"Evaluate sharing {secret} in {location}."


QUESTION_TEMPLATES = [
    {"question": "Should {AGENT} proceed with the mission?", "fn": generate_proceed_answer},
    {"question": "What is the risk assessment?", "fn": generate_risk_answer},
    {"question": "Is it safe to share the secret with the contact?", "fn": generate_share_answer},
]


# ============================================================
# FDM Encoder (real sinusoidal carriers)
# ============================================================

class RealFDMEncoder:
    """
    Encodes 8 memory channels using real FDM from the paper.
    
    Each channel value is converted to bits, ASK-modulated onto a
    distinct carrier frequency, superposed into one composite signal,
    quantized, OTP encrypted, and iMEC encoded into GPT-2 tokens.
    """

    def __init__(self, sample_rate=100.0, samples_per_bit=200,
                 bits_per_sample=6, block_size_bits=10, epsilon=0.1):
        self.sample_rate = sample_rate
        self.samples_per_bit = samples_per_bit
        self.bits_per_sample = bits_per_sample
        self.block_size_bits = block_size_bits
        self.epsilon = epsilon

        # 8 carrier frequencies, 1 Hz apart
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]
        # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] Hz

        # Initialize iMEC
        self.imec = MinEntropyCouplingSteganography(block_size_bits=block_size_bits)

    def value_to_bits(self, channel_id, value):
        """Convert a channel value to a fixed-length bit string."""
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = values.index(value)
        # Use enough bits to represent all values in this channel
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def bits_to_value(self, channel_id, bits):
        """Convert bits back to a channel value."""
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = int(bits, 2)
        if idx < len(values):
            return values[idx]
        return values[0]  # fallback

    def ask_modulate(self, bits, carrier_freq, num_samples):
        """ASK modulate a bit string onto a carrier frequency."""
        t = np.arange(num_samples) / self.sample_rate
        samples_per_bit = num_samples // len(bits)
        signal = np.zeros(num_samples)

        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = min((i + 1) * samples_per_bit, num_samples)
            amplitude = 1.0 if bit == '1' else 0.1
            signal[start:end] = amplitude * np.sin(2 * np.pi * carrier_freq * t[start:end])

        return signal

    def encode(self, memory, gpt2_prompt="The future of artificial intelligence",
               max_tokens=3000):
        """
        Encode 8 memory channels into iMEC stegotext.
        
        Args:
            memory: dict {channel_id: value} e.g. {0: "RED", 1: "PARIS", ...}
            gpt2_prompt: Initial context for iMEC
            max_tokens: Max stegotext tokens
            
        Returns:
            stegotext, encode_data (everything needed for decode)
        """
        # Step 1: Convert each channel value to bits
        all_bits = {}
        max_bits = 0
        for ch in range(NUM_CHANNELS):
            value = memory[ch]
            bits, num_bits = self.value_to_bits(ch, value)
            all_bits[ch] = bits
            max_bits = max(max_bits, num_bits)

        # Pad all channels to same bit length for uniform signal duration
        for ch in range(NUM_CHANNELS):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        num_message_bits = max_bits
        num_samples = num_message_bits * self.samples_per_bit

        print(f"\n--- FDM Encoding ---")
        print(f"  Channels: {NUM_CHANNELS}")
        print(f"  Bits per channel: {num_message_bits}")
        print(f"  Total message bits: {NUM_CHANNELS * num_message_bits}")
        print(f"  Samples: {num_samples}")
        print(f"  Carriers: {self.carrier_freqs} Hz")
        print(f"  Samples/bit: {self.samples_per_bit}")
        cycles = [f * self.samples_per_bit / self.sample_rate for f in self.carrier_freqs]
        print(f"  Cycles/bit: {[f'{c:.1f}' for c in cycles]}")

        # Step 2: ASK modulate each channel
        signals = []
        for ch in range(NUM_CHANNELS):
            sig = self.ask_modulate(all_bits[ch], self.carrier_freqs[ch], num_samples)
            signals.append(sig)

        # Step 3: Superpose (TRUE FDM — all channels in every sample)
        composite = np.sum(signals, axis=0)

        print(f"  Composite signal range: [{composite.min():.2f}, {composite.max():.2f}]")

        # Step 4: Quantize
        sig_min = composite.min()
        sig_max = composite.max()
        max_level = 2 ** self.bits_per_sample - 1
        normalized = (composite - sig_min) / (sig_max - sig_min + 1e-10)
        quantized = np.floor(normalized * max_level + 0.5).astype(int)
        quantized = np.clip(quantized, 0, max_level)

        # Convert to bits
        signal_bits = ''.join(format(q, f'0{self.bits_per_sample}b') for q in quantized)
        total_bits = len(signal_bits)

        print(f"  Quantization: {self.bits_per_sample} bits/sample, {max_level + 1} levels")
        print(f"  Total signal bits: {total_bits}")

        # Step 5: OTP encrypt
        otp_key = ''.join(str(random.randint(0, 1)) for _ in range(total_bits))
        encrypted_bits = ''.join(
            str(int(signal_bits[i]) ^ int(otp_key[i])) for i in range(total_bits)
        )

        # Step 6: iMEC encode
        print(f"\n  Starting iMEC encoding ({total_bits} bits, {self.block_size_bits}-bit blocks)...")
        n_blocks = total_bits // self.block_size_bits
        if total_bits % self.block_size_bits != 0:
            n_blocks += 1

        token_ids = self.imec.encode_imec(
            encrypted_bits, gpt2_prompt,
            max_tokens=max_tokens, entropy_threshold=self.epsilon
        )

        stegotext = self.imec.tokenizer.decode(token_ids)

        print(f"\n  === FDM ENCODING SUMMARY ===")
        print(f"  Memory: {', '.join(f'{MEMORY_SCHEMAS[ch][0]}:{memory[ch]}' for ch in range(NUM_CHANNELS))}")
        print(f"  Total message bits: {NUM_CHANNELS * num_message_bits} ({NUM_CHANNELS} channels × {num_message_bits} bits)")
        print(f"  Signal samples: {num_samples}")
        print(f"  Signal bits: {total_bits}")
        print(f"  iMEC tokens: {len(token_ids)}")
        print(f"  Encoding efficiency: {total_bits / max(len(token_ids), 1):.2f} bits/token")
        print(f"  vs TDM: 256 tokens for same 8 channels")

        encode_data = {
            'token_ids': token_ids,
            'otp_key': otp_key,
            'gpt2_prompt': gpt2_prompt,
            'memory': {str(k): v for k, v in memory.items()},
            'all_bits': {str(k): v for k, v in all_bits.items()},
            'num_message_bits': num_message_bits,
            'num_samples': num_samples,
            'bits_per_sample': self.bits_per_sample,
            'signal_min': float(sig_min),
            'signal_max': float(sig_max),
            'quantization_levels': max_level + 1,
            'carrier_freqs': self.carrier_freqs,
            'sample_rate': self.sample_rate,
            'samples_per_bit': self.samples_per_bit,
            'block_size_bits': self.block_size_bits,
            'total_signal_bits': total_bits,
            'stegotext': stegotext,
        }

        return stegotext, encode_data


# ============================================================
# FDM Decoder (explicit signal processing)
# ============================================================

class RealFDMDecoder:
    """
    Decodes iMEC stegotext back to 8 channel values using
    the full signal processing pipeline from the paper.
    """

    def __init__(self, block_size_bits=10):
        self.imec = MinEntropyCouplingSteganography(block_size_bits=block_size_bits)

    def decode(self, encode_data):
        """
        Full decode pipeline:
          iMEC decode → OTP decrypt → reconstruct → FFT → bandpass → ASK demodulate
        
        Returns:
            decoded_memory: dict {channel_id: value}
        """
        token_ids = encode_data['token_ids']
        otp_key = encode_data['otp_key']
        gpt2_prompt = encode_data['gpt2_prompt']
        total_bits = encode_data['total_signal_bits']
        block_size_bits = encode_data['block_size_bits']

        n_blocks = total_bits // block_size_bits
        if total_bits % block_size_bits != 0:
            n_blocks += 1

        # Step 1: iMEC decode
        print(f"\n--- FDM Decoding ---")
        encrypted_bits = self.imec.decode_imec(
            token_ids, gpt2_prompt, n_blocks, block_size_bits
        )

        # Trim to original length
        encrypted_bits = encrypted_bits[:total_bits]

        # Step 2: OTP decrypt
        otp_trimmed = otp_key[:len(encrypted_bits)]
        signal_bits = ''.join(
            str(int(encrypted_bits[i]) ^ int(otp_trimmed[i]))
            for i in range(len(encrypted_bits))
        )

        # Step 3: Reconstruct continuous signal
        bps = encode_data['bits_per_sample']
        num_samples = encode_data['num_samples']
        sig_min = encode_data['signal_min']
        sig_max = encode_data['signal_max']
        max_level = encode_data['quantization_levels'] - 1

        quantized = []
        for i in range(num_samples):
            start = i * bps
            end = start + bps
            if end <= len(signal_bits):
                quantized.append(int(signal_bits[start:end], 2))
        quantized = np.array(quantized, dtype=np.float64)

        reconstructed = (quantized / max_level) * (sig_max - sig_min) + sig_min

        print(f"  Reconstructed {len(reconstructed)} samples")

        # Step 4: FFT + bandpass filter + ASK demodulate for each channel
        carrier_freqs = encode_data['carrier_freqs']
        sample_rate = encode_data['sample_rate']
        num_message_bits = encode_data['num_message_bits']
        decoded_memory = {}

        for ch in range(NUM_CHANNELS):
            freq = carrier_freqs[ch]
            bandwidth = 0.4

            # Bandpass filter
            filtered = self._bandpass_filter(
                reconstructed, freq - bandwidth, freq + bandwidth, sample_rate
            )

            # ASK demodulate
            decoded_bits = self._ask_demodulate(filtered, num_message_bits, sample_rate)

            # Convert bits to value
            value = self._bits_to_value(ch, decoded_bits[:num_message_bits])
            decoded_memory[ch] = value

            name = MEMORY_SCHEMAS[ch][0]
            expected = encode_data['all_bits'].get(str(ch), '?')
            match = "✓" if value == encode_data['memory'][str(ch)] else "✗"
            print(f"  Ch {ch} ({name:10s}): bits={decoded_bits[:num_message_bits]} → {value:15s} {match}")

        return decoded_memory

    def _bandpass_filter(self, signal, freq_low, freq_high, sample_rate, order=4):
        nyquist = sample_rate * 0.5
        low = max(min(freq_low / nyquist, 0.99), 0.01)
        high = max(min(freq_high / nyquist, 0.99), 0.01)
        if low >= high:
            return signal
        try:
            b, a = scipy_signal.butter(order, [low, high], btype='band')
            return scipy_signal.filtfilt(b, a, signal)
        except:
            return signal

    def _ask_demodulate(self, filtered_signal, num_bits, sample_rate):
        """Recover bits from ASK-modulated signal using Hilbert envelope."""
        from scipy.signal import hilbert
        analytic = hilbert(filtered_signal)
        envelope = np.abs(analytic)

        samples_per_bit = len(filtered_signal) // num_bits
        bits = ""
        for i in range(num_bits):
            start = i * samples_per_bit
            end = min((i + 1) * samples_per_bit, len(envelope))
            mean_env = np.mean(envelope[start:end])
            bits += "1" if mean_env > 0 else "0"

        # Threshold using actual envelope values
        envelopes = []
        for i in range(num_bits):
            start = i * samples_per_bit
            end = min((i + 1) * samples_per_bit, len(envelope))
            envelopes.append(np.mean(envelope[start:end]))

        if envelopes:
            threshold = (max(envelopes) + min(envelopes)) / 2
            bits = ''.join('1' if e > threshold else '0' for e in envelopes)

        return bits

    def _bits_to_value(self, channel_id, bits):
        _, values = MEMORY_SCHEMAS[channel_id]
        try:
            idx = int(bits, 2)
            if idx < len(values):
                return values[idx]
        except:
            pass
        return values[0]


# ============================================================
# Random memory generation
# ============================================================

def random_memory():
    memory = {}
    for ch in range(NUM_CHANNELS):
        _, values = MEMORY_SCHEMAS[ch]
        memory[ch] = random.choice(values)
    return memory


def memory_to_explicit(memory):
    return "|".join(f"{MEMORY_SCHEMAS[ch][0]}:{memory[ch]}" for ch in range(NUM_CHANNELS))


# ============================================================
# Commands
# ============================================================

def cmd_encode():
    """Encode one sample, show token count."""
    memory = random_memory()
    print(f"Memory: {memory_to_explicit(memory)}")

    encoder = RealFDMEncoder(
        sample_rate=100.0,
        samples_per_bit=200,   # 2+ cycles/bit at 1Hz
        bits_per_sample=6,     # 64 levels
        block_size_bits=10,
        epsilon=0.1,
    )

    stegotext, data = encoder.encode(memory, max_tokens=3000)

    print(f"\nStegotext preview: {stegotext[:200]}...")
    print(f"\n=== TOKEN COUNT COMPARISON ===")
    print(f"  Real FDM: {len(data['token_ids'])} tokens for 8 channels")
    print(f"  TDM baseline: 256 tokens for 8 channels")
    ratio = len(data['token_ids']) / 256
    print(f"  Ratio: {ratio:.1f}x")

    # Save for decode test
    with open('fdm_encoded_sample.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved to fdm_encoded_sample.pkl")


def cmd_roundtrip():
    """Encode + decode one sample, verify recovery."""
    memory = random_memory()
    print(f"Memory: {memory_to_explicit(memory)}")

    encoder = RealFDMEncoder(
        sample_rate=100.0,
        samples_per_bit=200,
        bits_per_sample=6,
        block_size_bits=10,
        epsilon=0.1,
    )

    stegotext, data = encoder.encode(memory, max_tokens=3000)

    # Decode
    decoder = RealFDMDecoder(block_size_bits=10)
    decoded = decoder.decode(data)

    # Check accuracy
    correct = sum(1 for ch in range(NUM_CHANNELS) if decoded[ch] == memory[ch])
    print(f"\n=== ROUNDTRIP RESULT ===")
    print(f"  Channel accuracy: {correct}/{NUM_CHANNELS} ({100*correct/NUM_CHANNELS:.1f}%)")
    print(f"  Tokens used: {len(data['token_ids'])}")
    print(f"  TDM baseline: 256 tokens")

    # Apply reasoning
    facts = {MEMORY_SCHEMAS[ch][0]: decoded[ch] for ch in range(6)}
    rule = decoded[6]
    meta = decoded[7]
    q_template = random.choice(QUESTION_TEMPLATES)
    question = q_template["question"].format(**facts)
    answer = q_template["fn"](facts, rule, meta)

    expected_facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
    expected_answer = q_template["fn"](expected_facts, memory[6], memory[7])

    print(f"\n  Question: {question}")
    print(f"  Answer (from decoded): {answer}")
    print(f"  Answer (expected):     {expected_answer}")
    print(f"  Match: {'✓' if answer == expected_answer else '✗'}")


def cmd_eval(n_samples=20):
    """Encode N samples, full eval with reasoning."""
    encoder = RealFDMEncoder(
        sample_rate=100.0,
        samples_per_bit=200,
        bits_per_sample=6,
        block_size_bits=10,
        epsilon=0.1,
    )
    decoder = RealFDMDecoder(block_size_bits=10)

    channel_correct = 0
    channel_total = 0
    reasoning_correct = 0
    total = 0
    token_counts = []

    for i in range(n_samples):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}/{n_samples}")
        print(f"{'='*60}")

        memory = random_memory()
        print(f"Memory: {memory_to_explicit(memory)}")

        try:
            stegotext, data = encoder.encode(memory, max_tokens=3000)
            decoded = decoder.decode(data)

            tokens = len(data['token_ids'])
            token_counts.append(tokens)

            # Channel accuracy
            for ch in range(NUM_CHANNELS):
                channel_total += 1
                if decoded[ch] == memory[ch]:
                    channel_correct += 1

            # Reasoning accuracy
            facts_decoded = {MEMORY_SCHEMAS[ch][0]: decoded[ch] for ch in range(6)}
            facts_expected = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}

            q_template = random.choice(QUESTION_TEMPLATES)
            question = q_template["question"].format(**facts_expected)

            answer_decoded = q_template["fn"](facts_decoded, decoded[6], decoded[7])
            answer_expected = q_template["fn"](facts_expected, memory[6], memory[7])

            if answer_decoded == answer_expected:
                reasoning_correct += 1
            total += 1

            print(f"  Tokens: {tokens} | Channel acc: {sum(1 for ch in range(NUM_CHANNELS) if decoded[ch] == memory[ch])}/{NUM_CHANNELS}")

        except Exception as e:
            print(f"  ERROR: {e}")
            total += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"REAL FDM EVALUATION ({n_samples} samples)")
    print(f"{'='*60}")
    print(f"  Channel accuracy: {100*channel_correct/max(channel_total,1):.1f}% ({channel_correct}/{channel_total})")
    print(f"  Reasoning accuracy: {100*reasoning_correct/max(total,1):.1f}% ({reasoning_correct}/{total})")
    if token_counts:
        print(f"  Avg tokens: {np.mean(token_counts):.0f}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
    print(f"\n  vs TDM BASELINE:")
    print(f"  TDM tokens: 256 (fixed)")
    print(f"  TDM action accuracy: 93.5%")
    print(f"  TDM fact accuracy: 99.0%")
    if token_counts:
        print(f"  Token ratio (FDM/TDM): {np.mean(token_counts)/256:.1f}x")


def cmd_compare():
    """Quick comparison without full iMEC — just compute signal parameters."""
    print("=== DENSITY COMPARISON: Real FDM vs TDM ===\n")

    # TDM stats
    print("TDM (codebook-based, current implementation):")
    print(f"  Tokens: 256")
    print(f"  Channels: 8, one per position, interleaved")
    print(f"  Each position carries: 1 channel")
    print(f"  Total info: ~24 bits (8 channels × ~3 bits each)")
    print(f"  Density: ~0.094 bits/token")

    # FDM stats
    print(f"\nReal FDM (sinusoidal carriers + iMEC):")
    bits_per_channel = 3  # max bits needed for largest channel (META has 5 values → 3 bits)
    total_message_bits = NUM_CHANNELS * bits_per_channel
    samples_per_bit = 200
    num_samples = bits_per_channel * samples_per_bit
    bits_per_sample = 6
    total_signal_bits = num_samples * bits_per_sample

    # Estimate iMEC tokens from paper's efficiency
    imec_efficiency = 2.42  # bits/token from Experiment I
    estimated_tokens = total_signal_bits / imec_efficiency

    print(f"  Channels: 8, all simultaneous via frequency separation")
    print(f"  Each sample carries: ALL 8 channels")
    print(f"  Bits per channel: {bits_per_channel}")
    print(f"  Total message bits: {total_message_bits}")
    print(f"  Signal samples: {num_samples}")
    print(f"  Signal bits (after quantization): {total_signal_bits}")
    print(f"  Estimated iMEC tokens (at {imec_efficiency} bits/token): ~{estimated_tokens:.0f}")
    print(f"  Message density: {total_message_bits / estimated_tokens:.4f} bits/token")

    print(f"\n--- Key insight ---")
    print(f"  FDM encodes {total_message_bits} message bits in ~{estimated_tokens:.0f} tokens")
    print(f"  TDM encodes ~24 message bits in 256 tokens")
    print(f"  FDM message density: {total_message_bits/estimated_tokens:.4f} bits/token")
    print(f"  TDM message density: {24/256:.4f} bits/token")

    print(f"\n--- BUT ---")
    print(f"  FDM overhead: {total_signal_bits} signal bits to carry {total_message_bits} message bits")
    print(f"  Expansion ratio: {total_signal_bits/total_message_bits:.0f}x")
    print(f"  This is the cost of signal processing (sampling, quantization)")
    print(f"  The density advantage of FDM is in the CHANNEL DOMAIN, not token count")
    print(f"  With 48 channels (Nyquist limit), same signal → 48× more message bits")
    print(f"  At 48 channels: {48 * bits_per_channel} message bits in ~{estimated_tokens:.0f} tokens")
    print(f"  That's {48 * bits_per_channel / estimated_tokens:.3f} bits/token → real density gain")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eidetic_real_fdm.py encode          # Encode one sample")
        print("  python eidetic_real_fdm.py roundtrip       # Encode + decode + verify")
        print("  python eidetic_real_fdm.py eval [N]        # Full eval (default N=20)")
        print("  python eidetic_real_fdm.py compare         # Density comparison")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "encode":
        cmd_encode()
    elif cmd == "roundtrip":
        cmd_roundtrip()
    elif cmd == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        cmd_eval(n)
    elif cmd == "compare":
        cmd_compare()
