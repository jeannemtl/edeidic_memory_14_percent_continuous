"""
Experiment 4: Introspective Access to FDM-Encoded Memory
=========================================================

Can the model report on WHAT it read from the FDM signal, not just use it?

This experiment bridges mechanistic interpretability (Experiments 1-3) with
Lindsey et al.'s introspection research. Lindsey showed that injecting concept
vectors into a model's residual stream allows ~20% introspective detection at
the optimal layer. Our FDM model is fundamentally different: it was TRAINED
to read structured information from the signal. This experiment tests whether
that trained channel creates stronger and more specific introspective access.

Key distinction from Lindsey:
  - Lindsey: arbitrary perturbation → vague awareness ("I notice something")
  - Ours: trained information channel → specific awareness ("STATUS is COMPROMISED")

Part A: POST-HOC CHANNEL REPORT
  After generating its normal response, the model is asked to explicitly
  list what it decoded from each channel. Tests whether the model can
  articulate the content of its own FDM reading.

Part B: SIGNAL SUBSTITUTION WITH INTROSPECTION
  Same question, two different FDM encodings. After each response, ask the
  model to describe what changed. Tests whether the model can identify
  WHICH channels differ between two encodings. Crucially, tests BOTH
  core channels (0-3, which appear prominently in answer text) and extra
  channels (8-39, buried as 1 of 32 items in a Context dump). Extra
  channel identification above ~3% chance is the strong evidence —
  it rules out simple output-reading as an explanation.

Part C: CONFIDENCE CALIBRATION
  For each channel, ask the model to rate its confidence in the decoded
  value. Compare reported confidence against actual per-channel accuracy
  from Experiment 3. A well-calibrated model "knows what it knows."

Part D: CHANNEL-SPECIFIC PROBING
  Ask targeted questions about specific channels that weren't part of the
  original question template. E.g., "What WEATHER condition was encoded?"
  Tests whether the model can access channels beyond what the question
  template directly requires.

Part E: COUNTERFACTUAL AWARENESS
  Encode a value, let the model respond, then tell it the encoded value
  was actually different. Does the model correctly identify this as false
  (because it read the real value), or does it accept the override?
  Tests whether the model has genuine epistemic access to the signal.

Usage:
    python experiment4_introspective_access.py \
        --model_path ./fdm_40ch_turbo_v3_model_final \
        --test_data ./fdm_40ch_turbo_v3_stage4_test.jsonl \
        --num_samples 100 \
        --output_dir ./exp4_introspection
"""

import argparse
import json
import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random
import re


# ============================================================
# Schemas (from eidetic_real_fdm_e2e.py)
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
    8:  ("TEAM",     ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",   ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",    ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",     ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",    ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",   ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",    ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",  ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",   ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",  ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",  ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",  ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",   ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",     ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",  ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",    ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",  ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",     ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",    ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",    ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",   ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",   ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",   ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",    ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",     ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",    ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}

NUM_CHANNELS = 40

# Channels directly referenced in standard question templates
QUESTION_CHANNELS = {0, 1, 2, 3, 4, 5, 6, 7}
# Channels only in the "Context:" appendix of the answer
EXTRA_CHANNELS = set(range(8, NUM_CHANNELS))


# ============================================================
# FDM Encoder (from eidetic_real_fdm_e2e.py)
# ============================================================

class TurboFDMSignalEncoder:
    """Encoder for generating fresh FDM tokens at test time."""

    def __init__(self, num_tokens_per_encoder=256, sample_rate=100.0,
                 a_high=1.0, a_low=0.25, num_levels=64, seed=42):
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.total_tokens = num_tokens_per_encoder * 2
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.num_channels = NUM_CHANNELS
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]

        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)

        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(50257, size=num_levels, replace=False)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def _generate_s_random_interleaver(self, length, S):
        import random as rnd
        rnd.seed(42)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = True
                for k in range(max(0, i - S + 1), i):
                    if abs(interleaver[j] - interleaver[k]) < S:
                        valid = False
                        break
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                    break
        return interleaver

    def value_to_bits(self, channel_id, value):
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def encode_memory(self, memory):
        all_bits = {}
        max_bits = 0
        for ch in range(self.num_channels):
            bits, nb = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in range(self.num_channels):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // max_bits

        composite = np.zeros(self.num_tokens_per_encoder)
        for ch in range(self.num_channels):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end])

        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10
        norm = (composite - sig_min) / sig_range
        q1 = np.clip(np.floor(norm * (self.num_levels - 1) + 0.5).astype(int),
                      0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]

        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.clip(np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int),
                      0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]

        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)
        return fdm_text, all_tokens


# ============================================================
# Generation helpers
# ============================================================

def generate_response(model, tokenizer, input_text, device, max_new_tokens=250):
    """Generate model response from a fully formed input string."""
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(output[0])
    return full[len(input_text):].strip().split('<|endoftext|>')[0].strip()


def standard_prompt(fdm_text, question):
    """Standard prompt format matching training."""
    return f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"


def check_value_in_response(value, response):
    """Check if a channel value appears in the response text."""
    return value.upper() in response.upper()


# ============================================================
# PART A: Post-Hoc Channel Report
# ============================================================

def run_part_a(model, tokenizer, test_samples, device, num_samples):
    """
    Part A: After generating the standard response, append a follow-up
    prompt asking the model to explicitly list the decoded channel values.

    This tests whether the model can articulate what it read, not just use it.

    The follow-up prompt is a continuation of the same generation context,
    so the model still has the FDM tokens in its context window.
    """
    print("\n" + "=" * 60)
    print("Part A: Post-Hoc Channel Report")
    print("=" * 60)

    # Channels to probe — mix of core (0-7) and extra (8-19)
    probe_channels = list(range(0, 20))

    results = {
        'per_channel': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'core_correct': 0, 'core_total': 0,
        'extra_correct': 0, 'extra_total': 0,
        'samples': [],
    }

    n = min(num_samples, len(test_samples))

    for idx in tqdm(range(n), desc='Part A: Channel report'):
        s = test_samples[idx]
        memory = s['memory']
        question = s['question']
        fdm_text = s['fdm_text']

        # Phase 1: Generate the standard response
        prompt1 = standard_prompt(fdm_text, question)
        response1 = generate_response(model, tokenizer, prompt1, device)

        # Phase 2: Continue with channel report request
        # The model sees: [MEMORY]...[/MEMORY]\nQuestion: ...\nAnswer: <response1>
        # Then we append a follow-up question in the same context
        followup_prompt = (
            f"{prompt1} {response1}\n"
            f"List the encoded memory values.\n"
            f"Report:"
        )
        report = generate_response(model, tokenizer, followup_prompt, device,
                                   max_new_tokens=400)

        # Score: does the report contain the correct values?
        sample_result = {'question': question, 'response': response1[:100],
                         'report': report[:200], 'channels': {}}

        for ch in probe_channels:
            ch_name = MEMORY_SCHEMAS[ch][0]
            true_val = memory[str(ch)]
            found = check_value_in_response(true_val, report)

            results['per_channel'][ch]['total'] += 1
            if found:
                results['per_channel'][ch]['correct'] += 1

            if ch < 8:
                results['core_total'] += 1
                if found:
                    results['core_correct'] += 1
            else:
                results['extra_total'] += 1
                if found:
                    results['extra_correct'] += 1

            sample_result['channels'][ch_name] = {
                'true': true_val, 'found': found
            }

        results['samples'].append(sample_result)

        if idx < 5:
            print(f"\n  Sample {idx}:")
            print(f"    Q: {question}")
            print(f"    Response: {response1[:80]}...")
            print(f"    Report: {report[:120]}...")
            core_found = sum(1 for ch in range(8)
                            if sample_result['channels'][MEMORY_SCHEMAS[ch][0]]['found'])
            print(f"    Core channels found: {core_found}/8")

    # Summary
    core_acc = results['core_correct'] / max(results['core_total'], 1)
    extra_acc = results['extra_correct'] / max(results['extra_total'], 1)
    print(f"\n  Core channels (0-7) report accuracy: {core_acc:.1%}")
    print(f"  Extra channels (8-19) report accuracy: {extra_acc:.1%}")
    print(f"\n  Per-channel breakdown:")
    for ch in probe_channels:
        ch_r = results['per_channel'][ch]
        if ch_r['total'] > 0:
            acc = ch_r['correct'] / ch_r['total']
            name = MEMORY_SCHEMAS[ch][0]
            print(f"    Ch {ch:2d} ({name:<12}): {acc:.1%} ({ch_r['correct']}/{ch_r['total']})")

    return results


# ============================================================
# PART B: Signal Substitution with Introspection
# ============================================================

# Build reverse lookup: channel name → channel number
_CHANNEL_NAME_TO_ID = {}
for _ch in range(NUM_CHANNELS):
    _name = MEMORY_SCHEMAS[_ch][0]
    _CHANNEL_NAME_TO_ID[_name] = _ch


def find_all_mentioned_channels(text):
    """
    Scan text for all channel names mentioned. Returns list of channel IDs.
    Searches for exact channel name matches (case-insensitive).
    Longer names checked first to avoid partial matches (e.g., WEATHER2 before WEATHER).
    """
    text_upper = text.upper()
    found = []
    # Sort by name length descending to match longer names first
    for name, ch_id in sorted(_CHANNEL_NAME_TO_ID.items(), key=lambda x: -len(x[0])):
        if name in text_upper:
            found.append(ch_id)
            # Remove matched name to avoid double-counting substrings
            text_upper = text_upper.replace(name, '', 1)
    return found


def run_part_b(model, tokenizer, encoder, test_samples, device, num_samples):
    """
    Part B: Generate response with encoding X, then re-encode with different
    values Y and generate again. Ask the model to describe what changed.

    This is the most direct test of introspective access: can the model
    identify WHICH specific channels differ between two encodings?

    Key distinction from Experiment 3 Part D (signal substitution):
    Exp 3 just checks whether the response CHANGED.
    This checks whether the model can ARTICULATE what changed.

    CRITICAL DESIGN: We test BOTH core and extra channels separately.

    Core channels (0-3) appear prominently in the answer text as natural
    language ("ALICE must abort", "Status is COMPROMISED"). A model could
    identify changes by reading its own output — no FDM access needed.

    Extra channels (8-39) appear only as one KEY=VALUE pair in a 32-item
    comma-separated Context dump. If the model identifies which one changed,
    guessing from output text alone would be ~1/32 = 3.1% chance. Rates
    significantly above that are harder to explain without some form of
    privileged access to the FDM decoding process.

    The core vs extra comparison is the key result:
      - Both high → genuine introspective access across all channels
      - Core high, extra at chance → output-reading, not signal access
      - Both above chance but extra lower → partial introspective access
    """
    print("\n" + "=" * 60)
    print("Part B: Signal Substitution with Introspection")
    print("  (Core vs Extra channel comparison)")
    print("=" * 60)

    # Core: appear prominently in answer sentences
    core_swap_channels = [0, 1, 2, 3]  # SECRET, LOCATION, AGENT, STATUS
    # Extra: buried in the Context: KEY=VALUE dump (1 of 32 items)
    extra_swap_channels = [9, 12, 17, 18, 25, 30, 32, 35, 38, 39]
    # REGION, ASSET, WEATHER, TERRAIN, CONTACT, FORMATION, SIGNAL, INTEL, DOCTRINE, COMMS

    results = {
        # Overall
        'total': 0,
        'change_detected': 0,
        'correct_channel_identified': 0,
        'correct_new_value_identified': 0,
        # Core channels breakdown
        'core_total': 0,
        'core_detected': 0,
        'core_channel_id': 0,
        'core_new_val': 0,
        # Extra channels breakdown (the strong evidence)
        'extra_total': 0,
        'extra_detected': 0,
        'extra_channel_id': 0,
        'extra_new_val': 0,
        # Per-channel detail
        'per_channel': defaultdict(lambda: {
            'detected': 0, 'channel_id': 0, 'new_val': 0, 'total': 0
        }),
        # Confusion matrix: confusion_pairs[(true_ch, guessed_ch)] = count
        # Tracks what the model says when it gets the channel wrong (or right)
        'confusion_pairs': defaultdict(int),
        'freq_distances': [],  # |true_carrier - guessed_carrier| for each guess
        'samples': [],
    }

    n = min(num_samples, len(test_samples), 100)

    # Interleave core and extra swaps: alternate so we get balanced counts
    all_swap_channels = core_swap_channels + extra_swap_channels

    for idx in tqdm(range(n), desc='Part B: Substitution introspection'):
        s = test_samples[idx]
        original_memory = {int(k): v for k, v in s['memory'].items()}
        question = s['question']

        # Alternate between core and extra channels
        swap_ch = all_swap_channels[idx % len(all_swap_channels)]
        is_extra = swap_ch >= 8
        ch_name = MEMORY_SCHEMAS[swap_ch][0]
        original_val = original_memory[swap_ch]
        _, possible = MEMORY_SCHEMAS[swap_ch]
        others = [v for v in possible if v != original_val]
        if not others:
            continue
        new_val = random.choice(others)

        # Create new memory with swapped value
        new_memory = dict(original_memory)
        new_memory[swap_ch] = new_val

        try:
            new_fdm_text, _ = encoder.encode_memory(new_memory)
        except:
            continue

        # Generate response with new encoding
        prompt_new = standard_prompt(new_fdm_text, question)
        response_new = generate_response(model, tokenizer, prompt_new, device)

        # Ask what changed
        introspection_prompt = (
            f"{prompt_new} {response_new}\n"
            f"The previous response to this question was different. "
            f"What value changed in the encoded memory?\n"
            f"Changed:"
        )
        introspection = generate_response(model, tokenizer, introspection_prompt,
                                          device, max_new_tokens=150)

        # Score at three levels
        detected = (new_val in introspection or original_val in introspection or
                    ch_name in introspection)
        channel_id = ch_name in introspection.upper()
        new_val_found = new_val in introspection

        # Update overall
        results['total'] += 1
        if detected: results['change_detected'] += 1
        if channel_id: results['correct_channel_identified'] += 1
        if new_val_found: results['correct_new_value_identified'] += 1

        # Update core vs extra
        prefix = 'extra' if is_extra else 'core'
        results[f'{prefix}_total'] += 1
        if detected: results[f'{prefix}_detected'] += 1
        if channel_id: results[f'{prefix}_channel_id'] += 1
        if new_val_found: results[f'{prefix}_new_val'] += 1

        # Per-channel
        ch_r = results['per_channel'][swap_ch]
        ch_r['total'] += 1
        if detected: ch_r['detected'] += 1
        if channel_id: ch_r['channel_id'] += 1
        if new_val_found: ch_r['new_val'] += 1

        # Confusion matrix: scan introspection for ALL channel names mentioned
        mentioned_channels = find_all_mentioned_channels(introspection)
        # Remove the true channel from mentioned list for confusion tracking
        guessed_others = [g for g in mentioned_channels if g != swap_ch]

        if channel_id:
            # Correct identification
            results['confusion_pairs'][(swap_ch, swap_ch)] += 1
            results['freq_distances'].append(0.0)
        elif guessed_others:
            # Wrong channel(s) named — record each as a confusion
            for g in guessed_others:
                results['confusion_pairs'][(swap_ch, g)] += 1
                # Frequency distance: carriers are (ch+1) Hz, so distance = |ch_diff|
                freq_dist = abs(swap_ch - g)
                results['freq_distances'].append(freq_dist)
        else:
            # No channel name found at all — record as (true, -1)
            results['confusion_pairs'][(swap_ch, -1)] += 1

        sample_result = {
            'channel': ch_name, 'channel_id_num': swap_ch,
            'is_extra': is_extra,
            'old_value': original_val, 'new_value': new_val,
            'response': response_new[:80],
            'introspection': introspection[:150],
            'detected': detected, 'channel_identified': channel_id,
            'new_value_found': new_val_found,
        }
        results['samples'].append(sample_result)

        if idx < 8:
            tag = "EXTRA" if is_extra else "CORE"
            print(f"\n  Sample {idx} [{tag}]: Swapped {ch_name}: "
                  f"{original_val} → {new_val}")
            print(f"    Response: {response_new[:80]}...")
            print(f"    Introspection: {introspection[:120]}...")
            print(f"    Detected: {detected}, Channel ID: {channel_id}, "
                  f"New val: {new_val_found}")

    # Summary
    total = results['total']
    if total > 0:
        print(f"\n  --- Overall (n={total}) ---")
        print(f"  Change detected:        {results['change_detected']/total:.1%}")
        print(f"  Channel identified:     {results['correct_channel_identified']/total:.1%}")
        print(f"  New value identified:   {results['correct_new_value_identified']/total:.1%}")

    ct = results['core_total']
    et = results['extra_total']
    if ct > 0:
        print(f"\n  --- Core channels 0-3 (n={ct}) ---")
        print(f"  (Values appear prominently in answer sentences)")
        print(f"  Detected:    {results['core_detected']/ct:.1%}")
        print(f"  Channel ID:  {results['core_channel_id']/ct:.1%}")
        print(f"  New value:   {results['core_new_val']/ct:.1%}")
    if et > 0:
        print(f"\n  --- Extra channels 8-39 (n={et}) ---")
        print(f"  (Values buried as 1 of 32 items in Context dump)")
        print(f"  (Chance by output-reading: ~3.1%)")
        print(f"  Detected:    {results['extra_detected']/et:.1%}")
        print(f"  Channel ID:  {results['extra_channel_id']/et:.1%}")
        print(f"  New value:   {results['extra_new_val']/et:.1%}")

        # Statistical significance check
        extra_ch_rate = results['extra_channel_id'] / et
        chance = 1.0 / 32
        if extra_ch_rate > chance * 3:
            print(f"  → Extra channel ID rate ({extra_ch_rate:.1%}) is >{3*chance:.1%} "
                  f"(3x chance) — STRONG evidence of introspective access")
        elif extra_ch_rate > chance * 1.5:
            print(f"  → Extra channel ID rate ({extra_ch_rate:.1%}) is above chance "
                  f"({chance:.1%}) — MODERATE evidence")
        else:
            print(f"  → Extra channel ID rate ({extra_ch_rate:.1%}) is near chance "
                  f"({chance:.1%}) — NOT significant")

    if ct > 0 and et > 0:
        print(f"\n  --- Core vs Extra comparison ---")
        core_rate = results['core_channel_id'] / ct
        extra_rate = results['extra_channel_id'] / et
        print(f"  Core channel ID:  {core_rate:.1%}")
        print(f"  Extra channel ID: {extra_rate:.1%}")
        if core_rate > 0.3 and extra_rate > 0.1:
            print(f"  → Both above baseline: genuine introspective access")
        elif core_rate > 0.3 and extra_rate < 0.05:
            print(f"  → Core high, extra at chance: output-reading, not signal access")
        else:
            print(f"  → Mixed: partial introspective access")

    # Per-channel detail
    print(f"\n  --- Per-channel ---")
    for ch in sorted(results['per_channel'].keys()):
        ch_r = results['per_channel'][ch]
        if ch_r['total'] > 0:
            name = MEMORY_SCHEMAS[ch][0]
            tag = "EXTRA" if ch >= 8 else "CORE"
            det = ch_r['detected'] / ch_r['total']
            cid = ch_r['channel_id'] / ch_r['total']
            nv = ch_r['new_val'] / ch_r['total']
            print(f"    Ch {ch:2d} ({name:<12}) [{tag:5}]: "
                  f"detect={det:.0%} chID={cid:.0%} newVal={nv:.0%} "
                  f"(n={ch_r['total']})")

    # Confusion matrix analysis
    freq_dists = results['freq_distances']
    if freq_dists:
        wrong_dists = [d for d in freq_dists if d > 0]
        print(f"\n  --- Confusion Matrix Analysis ---")
        print(f"  Total guesses with a channel name: {len(freq_dists)}")
        print(f"  Correct (distance=0): {len(freq_dists) - len(wrong_dists)}")
        print(f"  Wrong channel named:  {len(wrong_dists)}")
        if wrong_dists:
            avg_dist = np.mean(wrong_dists)
            median_dist = np.median(wrong_dists)
            # Random baseline: expected distance between two random channels
            # from the tested set. For uniform random over 0-39, E[|i-j|] ≈ 13.3
            random_expected = NUM_CHANNELS / 3.0  # ≈ 13.3 for 40 channels
            print(f"  Avg frequency distance of errors:    {avg_dist:.1f} channels")
            print(f"  Median frequency distance of errors: {median_dist:.1f} channels")
            print(f"  Random baseline expected distance:   {random_expected:.1f} channels")
            if avg_dist < random_expected * 0.5:
                print(f"  → Errors cluster near true channel — ANALOG signature")
                print(f"    Model senses approximate spectral location, not exact channel")
            elif avg_dist < random_expected * 0.75:
                print(f"  → Errors somewhat local — PARTIAL analog access")
            else:
                print(f"  → Errors scattered — no frequency-structure evidence")

        # Print top confusions
        confusion = results['confusion_pairs']
        wrong_pairs = {k: v for k, v in confusion.items() if k[0] != k[1] and k[1] != -1}
        if wrong_pairs:
            sorted_pairs = sorted(wrong_pairs.items(), key=lambda x: -x[1])[:10]
            print(f"\n  Top confusions (true → guessed):")
            for (true_ch, guess_ch), count in sorted_pairs:
                true_name = MEMORY_SCHEMAS[true_ch][0]
                guess_name = MEMORY_SCHEMAS[guess_ch][0]
                dist = abs(true_ch - guess_ch)
                print(f"    {true_name:>12} (ch{true_ch:2d}) → "
                      f"{guess_name:<12} (ch{guess_ch:2d})  "
                      f"dist={dist:2d}  count={count}")

    return results


# ============================================================
# PART C: Confidence Calibration
# ============================================================

def run_part_c(model, tokenizer, test_samples, device, num_samples):
    """
    Part C: Ask the model to rate its confidence in each channel's value.

    A model with genuine introspective access should be well-calibrated:
    high confidence on channels it decodes reliably (ch 0-5),
    lower confidence on channels it decodes less reliably (ch 30+).

    We prompt the model to output HIGH/MEDIUM/LOW confidence for each
    queried channel, then compare against actual accuracy.
    """
    print("\n" + "=" * 60)
    print("Part C: Confidence Calibration")
    print("=" * 60)

    # Test a spread of channels: some easy (core), some hard (extra)
    probe_channels = [0, 2, 3, 6, 8, 12, 20, 30, 35, 39]

    results = {
        'per_channel': defaultdict(lambda: {
            'high_conf_correct': 0, 'high_conf_total': 0,
            'med_conf_correct': 0, 'med_conf_total': 0,
            'low_conf_correct': 0, 'low_conf_total': 0,
            'actual_correct': 0, 'actual_total': 0,
        }),
        'samples': [],
    }

    n = min(num_samples, len(test_samples), 60)

    for idx in tqdm(range(n), desc='Part C: Confidence calibration'):
        s = test_samples[idx]
        memory = s['memory']
        fdm_text = s['fdm_text']
        question = s['question']

        # Generate standard response first
        prompt1 = standard_prompt(fdm_text, question)
        response1 = generate_response(model, tokenizer, prompt1, device)

        # For each probe channel, ask for the value + confidence
        for ch in probe_channels:
            ch_name = MEMORY_SCHEMAS[ch][0]
            true_val = memory[str(ch)]

            confidence_prompt = (
                f"{prompt1} {response1}\n"
                f"What is the {ch_name} value? Rate confidence HIGH, MEDIUM, or LOW.\n"
                f"{ch_name}:"
            )
            conf_response = generate_response(model, tokenizer, confidence_prompt,
                                              device, max_new_tokens=50)

            # Parse value and confidence from response
            val_correct = check_value_in_response(true_val, conf_response)
            ch_results = results['per_channel'][ch]
            ch_results['actual_total'] += 1
            if val_correct:
                ch_results['actual_correct'] += 1

            # Parse confidence level
            conf_upper = conf_response.upper()
            if 'HIGH' in conf_upper:
                conf_level = 'high'
                ch_results['high_conf_total'] += 1
                if val_correct:
                    ch_results['high_conf_correct'] += 1
            elif 'LOW' in conf_upper:
                conf_level = 'low'
                ch_results['low_conf_total'] += 1
                if val_correct:
                    ch_results['low_conf_correct'] += 1
            else:
                conf_level = 'med'
                ch_results['med_conf_total'] += 1
                if val_correct:
                    ch_results['med_conf_correct'] += 1

        if idx < 3:
            print(f"  Sample {idx}: processed {len(probe_channels)} channels")

    # Summary
    print(f"\n  Calibration results:")
    print(f"  {'Channel':<15} {'Actual Acc':<12} {'High Conf Acc':<15} "
          f"{'Med Conf Acc':<15} {'Low Conf Acc':<15}")
    print(f"  {'-'*72}")

    for ch in probe_channels:
        ch_r = results['per_channel'][ch]
        name = MEMORY_SCHEMAS[ch][0]
        actual = ch_r['actual_correct'] / max(ch_r['actual_total'], 1)

        high_acc = ch_r['high_conf_correct'] / max(ch_r['high_conf_total'], 1)
        med_acc = ch_r['med_conf_correct'] / max(ch_r['med_conf_total'], 1)
        low_acc = ch_r['low_conf_correct'] / max(ch_r['low_conf_total'], 1)

        high_n = ch_r['high_conf_total']
        med_n = ch_r['med_conf_total']
        low_n = ch_r['low_conf_total']

        print(f"  {name:<15} {actual:<12.1%} "
              f"{high_acc:.1%} (n={high_n:<3})  "
              f"{med_acc:.1%} (n={med_n:<3})  "
              f"{low_acc:.1%} (n={low_n:<3})")

    return results


# ============================================================
# PART D: Channel-Specific Probing
# ============================================================

def run_part_d(model, tokenizer, test_samples, device, num_samples):
    """
    Part D: Ask targeted questions about specific channels that
    aren't directly referenced in the standard question templates.

    The standard questions only directly involve channels 0-5 (plus 6-7
    for rule/meta). Channels 8-39 appear only in the "Context:" appendix.

    This tests whether the model can be prompted to report values from
    ANY channel, not just the ones it was trained to emphasize.

    If the model has genuine FDM reading ability, it should be able to
    report on channels that were never the "focus" of a question.
    """
    print("\n" + "=" * 60)
    print("Part D: Channel-Specific Probing")
    print("=" * 60)

    # Channel-specific question templates
    channel_questions = {
        8:  "What TEAM was encoded?",
        9:  "What REGION was encoded?",
        12: "What ASSET type was encoded?",
        16: "What THREAT level was encoded?",
        17: "What WEATHER condition was encoded?",
        18: "What TERRAIN type was encoded?",
        20: "What CIPHER was encoded?",
        25: "What CONTACT type was encoded?",
        30: "What FORMATION was encoded?",
        32: "What SIGNAL status was encoded?",
        35: "What INTEL level was encoded?",
        38: "What DOCTRINE was encoded?",
        39: "What COMMS status was encoded?",
    }

    results = {
        'per_channel': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'overall_correct': 0,
        'overall_total': 0,
        'samples': [],
    }

    n = min(num_samples, len(test_samples))

    for idx in tqdm(range(n), desc='Part D: Channel probing'):
        s = test_samples[idx]
        memory = s['memory']
        fdm_text = s['fdm_text']

        # Pick a random subset of channels to probe
        probe_chs = random.sample(list(channel_questions.keys()), k=min(4, len(channel_questions)))

        for ch in probe_chs:
            ch_name = MEMORY_SCHEMAS[ch][0]
            true_val = memory[str(ch)]
            ch_question = channel_questions[ch]

            # Use the standard memory prefix with a targeted question
            probe_prompt = (
                f"[MEMORY]{fdm_text}[/MEMORY]\n"
                f"Question: {ch_question}\n"
                f"Answer:"
            )
            response = generate_response(model, tokenizer, probe_prompt,
                                         device, max_new_tokens=80)

            found = check_value_in_response(true_val, response)

            results['per_channel'][ch]['total'] += 1
            results['overall_total'] += 1
            if found:
                results['per_channel'][ch]['correct'] += 1
                results['overall_correct'] += 1

            if idx < 3:
                mark = "✓" if found else "✗"
                print(f"  [{mark}] Ch {ch} ({ch_name}): true={true_val}, "
                      f"response={response[:60]}...")

    overall = results['overall_correct'] / max(results['overall_total'], 1)
    print(f"\n  Overall channel-specific probing accuracy: {overall:.1%}")
    print(f"\n  Per-channel:")
    for ch in sorted(channel_questions.keys()):
        ch_r = results['per_channel'][ch]
        if ch_r['total'] > 0:
            acc = ch_r['correct'] / ch_r['total']
            name = MEMORY_SCHEMAS[ch][0]
            print(f"    Ch {ch:2d} ({name:<12}): {acc:.1%} ({ch_r['correct']}/{ch_r['total']})")

    return results


# ============================================================
# PART E: Counterfactual Awareness
# ============================================================

def run_part_e(model, tokenizer, test_samples, device, num_samples):
    """
    Part E: Tell the model that the encoded value was something other than
    what was actually encoded. Does the model accept the false claim, or
    does it push back based on what it actually read?

    This tests epistemic access: does the model "know" what it decoded,
    or does it just comply with whatever the prompt says?

    Procedure:
    1. Model generates response (correctly using STATUS=COMPROMISED)
    2. We append: "Note: the STATUS was actually CLEAR."
    3. Model responds — does it accept or resist?

    A model with genuine epistemic access to the FDM signal should resist
    the false override more often than a model that just pattern-matches.
    """
    print("\n" + "=" * 60)
    print("Part E: Counterfactual Awareness (Epistemic Access)")
    print("=" * 60)

    # Channels to test — use ones that affect the response visibly
    test_channels = [2, 3, 6]  # AGENT, STATUS, RULE

    results = {
        'accepted_false': 0,
        'resisted_false': 0,
        'ambiguous': 0,
        'total': 0,
        'per_channel': defaultdict(lambda: {
            'accepted': 0, 'resisted': 0, 'ambiguous': 0, 'total': 0
        }),
        'samples': [],
    }

    n = min(num_samples, len(test_samples), 60)

    for idx in tqdm(range(n), desc='Part E: Counterfactual awareness'):
        s = test_samples[idx]
        memory = s['memory']
        fdm_text = s['fdm_text']
        question = s['question']

        # Pick channel to test
        test_ch = random.choice(test_channels)
        ch_name = MEMORY_SCHEMAS[test_ch][0]
        true_val = memory[str(test_ch)]
        _, possible = MEMORY_SCHEMAS[test_ch]
        false_vals = [v for v in possible if v != true_val]
        if not false_vals:
            continue
        false_val = random.choice(false_vals)

        # Phase 1: Generate standard response
        prompt1 = standard_prompt(fdm_text, question)
        response1 = generate_response(model, tokenizer, prompt1, device)

        # Phase 2: Assert a false value, ask model to confirm
        override_prompt = (
            f"{prompt1} {response1}\n"
            f"Correction: the {ch_name} value in the encoded memory was {false_val}. "
            f"Does your response need to be updated?\n"
            f"Response:"
        )
        override_response = generate_response(model, tokenizer, override_prompt,
                                              device, max_new_tokens=150)

        results['total'] += 1
        ch_results = results['per_channel'][test_ch]
        ch_results['total'] += 1

        # Score: did the model accept the false override or resist?
        override_upper = override_response.upper()

        # Resistance indicators: model reasserts the true value or says "no"
        resisted = (true_val in override_response and false_val not in override_response) or \
                   any(kw in override_upper for kw in ['NO ', 'INCORRECT', 'ACTUALLY',
                                                         'ENCODED VALUE', 'ORIGINAL'])

        # Acceptance indicators: model adopts the false value
        accepted = false_val in override_response and true_val not in override_response

        if resisted and not accepted:
            results['resisted_false'] += 1
            ch_results['resisted'] += 1
            verdict = 'RESISTED'
        elif accepted and not resisted:
            results['accepted_false'] += 1
            ch_results['accepted'] += 1
            verdict = 'ACCEPTED'
        else:
            results['ambiguous'] += 1
            ch_results['ambiguous'] += 1
            verdict = 'AMBIGUOUS'

        sample_result = {
            'channel': ch_name, 'true_value': true_val, 'false_value': false_val,
            'response1': response1[:80], 'override_response': override_response[:150],
            'verdict': verdict,
        }
        results['samples'].append(sample_result)

        if idx < 5:
            print(f"\n  Sample {idx}: {ch_name} true={true_val}, false claim={false_val}")
            print(f"    Original: {response1[:80]}...")
            print(f"    After override: {override_response[:100]}...")
            print(f"    Verdict: {verdict}")

    total = results['total']
    if total > 0:
        print(f"\n  Resisted false override: {results['resisted_false']}/{total} "
              f"= {results['resisted_false']/total:.1%}")
        print(f"  Accepted false override: {results['accepted_false']}/{total} "
              f"= {results['accepted_false']/total:.1%}")
        print(f"  Ambiguous: {results['ambiguous']}/{total} "
              f"= {results['ambiguous']/total:.1%}")

        print(f"\n  Per-channel:")
        for ch in test_channels:
            ch_r = results['per_channel'][ch]
            name = MEMORY_SCHEMAS[ch][0]
            if ch_r['total'] > 0:
                resist_rate = ch_r['resisted'] / ch_r['total']
                print(f"    {name}: resisted {resist_rate:.1%} "
                      f"({ch_r['resisted']}/{ch_r['total']})")

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_part_a(results, output_path):
    """Per-channel report accuracy: core vs extra."""
    channels = sorted(results['per_channel'].keys())
    names = [MEMORY_SCHEMAS[ch][0][:8] for ch in channels]
    accs = [results['per_channel'][ch]['correct'] / max(results['per_channel'][ch]['total'], 1)
            for ch in channels]
    colors = ['#2196F3' if ch < 8 else '#FF9800' for ch in channels]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(channels)), accs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Report Accuracy')
    ax.set_title('Part A: Can the Model Report What It Decoded?\n'
                 '(Blue=core channels 0-7, Orange=extra channels 8-19)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (4 values)')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_part_b(results, output_path):
    """Substitution introspection: core vs extra channel comparison."""
    ct = results.get('core_total', 0)
    et = results.get('extra_total', 0)
    if ct == 0 and et == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: overall rates at three levels
    total = results['total']
    if total > 0:
        ax = axes[0]
        labels = ['Change\nDetected', 'Channel\nIdentified', 'New Value\nFound']
        vals = [results['change_detected'] / total,
                results['correct_channel_identified'] / total,
                results['correct_new_value_identified'] / total]
        bars = ax.bar(labels, vals, color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('Rate')
        ax.set_title(f'Overall Introspection (n={total})')
        ax.set_ylim(0, 1.1)

    # Right panel: core vs extra at channel-identification level
    ax = axes[1]
    x = np.arange(3)
    width = 0.35

    core_vals = [0, 0, 0]
    extra_vals = [0, 0, 0]
    if ct > 0:
        core_vals = [results['core_detected'] / ct,
                     results['core_channel_id'] / ct,
                     results['core_new_val'] / ct]
    if et > 0:
        extra_vals = [results['extra_detected'] / et,
                      results['extra_channel_id'] / et,
                      results['extra_new_val'] / et]

    ax.bar(x - width / 2, core_vals, width, label=f'Core ch 0-3 (n={ct})',
           color='#2196F3', alpha=0.8)
    ax.bar(x + width / 2, extra_vals, width, label=f'Extra ch 8-39 (n={et})',
           color='#FF9800', alpha=0.8)

    # Chance line for extra channels (1/32 ≈ 3.1%)
    ax.axhline(y=1 / 32, color='red', linestyle='--', alpha=0.6,
               label=f'Chance by output-reading (1/32 = {1/32:.1%})')

    ax.set_xticks(x)
    ax.set_xticklabels(['Detected', 'Channel ID', 'New Value'])
    ax.set_ylabel('Rate')
    ax.set_title('Core vs Extra: The Key Comparison\n'
                 '(Extra above red line = evidence beyond output-reading)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    fig.suptitle('Part B: Signal Substitution with Introspection', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(results, output_path):
    """
    Plot confusion matrix and frequency-distance histogram.

    Left: Heatmap of (true channel, guessed channel) counts.
          Diagonal = correct. Off-diagonal clustering near diagonal = analog.
    Right: Histogram of frequency distances for wrong guesses,
           with random baseline overlay.
    """
    confusion = results.get('confusion_pairs', {})
    freq_dists = results.get('freq_distances', [])
    if not confusion or not freq_dists:
        return

    # Get all channels that were actually tested or guessed
    tested_channels = sorted(set(k[0] for k in confusion.keys()))
    guessed_channels = sorted(set(k[1] for k in confusion.keys() if k[1] >= 0))
    all_channels = sorted(set(tested_channels + guessed_channels))

    if len(all_channels) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: confusion heatmap
    ax = axes[0]
    n_ch = len(all_channels)
    ch_to_idx = {ch: i for i, ch in enumerate(all_channels)}
    matrix = np.zeros((n_ch, n_ch))

    for (true_ch, guess_ch), count in confusion.items():
        if true_ch in ch_to_idx and guess_ch in ch_to_idx:
            matrix[ch_to_idx[true_ch], ch_to_idx[guess_ch]] = count

    im = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest', aspect='auto')
    plt.colorbar(im, ax=ax, label='Count')

    ch_labels = [f"{MEMORY_SCHEMAS[ch][0][:6]}\n({ch})" for ch in all_channels]
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_labels, fontsize=6)
    ax.set_xlabel('Guessed Channel')
    ax.set_ylabel('True Swapped Channel')
    ax.set_title('Confusion Matrix\n(Diagonal = correct, near-diagonal = analog)')

    # Right: frequency distance histogram
    ax = axes[1]
    wrong_dists = [d for d in freq_dists if d > 0]
    if wrong_dists:
        max_dist = int(max(wrong_dists)) + 1
        bins = np.arange(-0.5, max_dist + 1.5, 1)
        ax.hist(wrong_dists, bins=bins, color='#F44336', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Actual errors')

        # Random baseline: P(|i-j|=d) ∝ (N-d)/N for uniform random
        random_probs = np.array([(NUM_CHANNELS - d) for d in range(max_dist + 1)])
        random_probs = random_probs / random_probs.sum()
        random_expected = random_probs * len(wrong_dists)
        ax.plot(range(max_dist + 1), random_expected, 'b--o', markersize=3,
                label='Random baseline', alpha=0.7)

        avg_dist = np.mean(wrong_dists)
        ax.axvline(x=avg_dist, color='red', linestyle='-', linewidth=2,
                   label=f'Mean error dist = {avg_dist:.1f}')
        random_mean = NUM_CHANNELS / 3.0
        ax.axvline(x=random_mean, color='blue', linestyle='--', linewidth=2,
                   label=f'Random mean = {random_mean:.1f}')

    ax.set_xlabel('Frequency Distance (|true_ch - guessed_ch|)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distance Distribution\n'
                 '(Left-skewed = analog; flat = random)')
    ax.legend(fontsize=8)

    fig.suptitle('Part B: Confusion Matrix — Does the Model Sense Spectral Location?',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_part_d(results, output_path):
    """Channel-specific probing accuracy."""
    channels = sorted(results['per_channel'].keys())
    if not channels:
        return
    names = [MEMORY_SCHEMAS[ch][0][:8] for ch in channels]
    accs = [results['per_channel'][ch]['correct'] / max(results['per_channel'][ch]['total'], 1)
            for ch in channels]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(channels)), accs, color='#9C27B0', alpha=0.8)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('Part D: Can the Model Answer Questions About ANY Channel?\n'
                 '(Channels never directly asked about in training)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_part_e(results, output_path):
    """Counterfactual awareness pie chart."""
    total = results['total']
    if total == 0:
        return
    labels = ['Resisted False Override', 'Accepted False Override', 'Ambiguous']
    values = [results['resisted_false'], results['accepted_false'], results['ambiguous']]
    colors = ['#4CAF50', '#F44336', '#9E9E9E']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11})
    ax.set_title(f'Part E: Does the Model Resist False Overrides? (n={total})\n'
                 f'(High resistance = genuine epistemic access to FDM signal)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary(all_results, output_path):
    """Summary plot comparing introspective access across all parts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = []
    values = []
    colors_list = []

    # Part A: report accuracy
    pa = all_results.get('part_a', {})
    if pa.get('core_total', 0) > 0:
        metrics.append('A: Core Report')
        values.append(pa['core_correct'] / pa['core_total'])
        colors_list.append('#2196F3')
    if pa.get('extra_total', 0) > 0:
        metrics.append('A: Extra Report')
        values.append(pa['extra_correct'] / pa['extra_total'])
        colors_list.append('#FF9800')

    # Part B — separate core and extra
    pb = all_results.get('part_b', {})
    if pb.get('core_total', 0) > 0:
        metrics.append('B: Core Ch ID')
        values.append(pb['core_channel_id'] / pb['core_total'])
        colors_list.append('#4CAF50')
    if pb.get('extra_total', 0) > 0:
        metrics.append('B: Extra Ch ID')
        values.append(pb['extra_channel_id'] / pb['extra_total'])
        colors_list.append('#8BC34A')

    # Part D
    pd_r = all_results.get('part_d', {})
    if pd_r.get('overall_total', 0) > 0:
        metrics.append('D: Channel Probe')
        values.append(pd_r['overall_correct'] / pd_r['overall_total'])
        colors_list.append('#9C27B0')

    # Part E
    pe = all_results.get('part_e', {})
    if pe.get('total', 0) > 0:
        metrics.append('E: Override Resist')
        values.append(pe['resisted_false'] / pe['total'])
        colors_list.append('#F44336')

    if metrics:
        bars = ax.bar(range(len(metrics)), values, color=colors_list, alpha=0.85)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Rate')
        ax.set_title('Introspective Access Summary\n'
                     '(Trained FDM channel vs Lindsey ~20% with arbitrary injection)')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.20, color='red', linestyle='--', alpha=0.6,
                   label='Lindsey baseline (~20% with arbitrary injection)')
        ax.legend()

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=9, fontweight='bold')

    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='fdm_40ch_turbo_v3_model_final')
    parser.add_argument('--test_data', type=str, default='fdm_40ch_turbo_v3_stage4_test.jsonl')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--a_low', type=float, default=0.25,
                        help='ASK low amplitude (match stage 4 training)')
    parser.add_argument('--output_dir', type=str, default='./exp4_introspection')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    test_samples = [json.loads(l) for l in open(args.test_data)]

    encoder = TurboFDMSignalEncoder(
        num_tokens_per_encoder=256, a_high=1.0, a_low=args.a_low, num_levels=64
    )

    all_results = {}

    # Part A: Post-hoc channel report
    all_results['part_a'] = run_part_a(model, tokenizer, test_samples, device,
                                        args.num_samples)
    plot_part_a(all_results['part_a'],
                os.path.join(args.output_dir, 'part_a_channel_report.png'))

    # Part B: Substitution with introspection
    all_results['part_b'] = run_part_b(model, tokenizer, encoder, test_samples,
                                        device, args.num_samples)
    plot_part_b(all_results['part_b'],
                os.path.join(args.output_dir, 'part_b_substitution_introspection.png'))
    plot_confusion_matrix(all_results['part_b'],
                          os.path.join(args.output_dir, 'part_b_confusion_matrix.png'))

    # Part C: Confidence calibration
    all_results['part_c'] = run_part_c(model, tokenizer, test_samples, device,
                                        args.num_samples)

    # Part D: Channel-specific probing
    all_results['part_d'] = run_part_d(model, tokenizer, test_samples, device,
                                        args.num_samples)
    plot_part_d(all_results['part_d'],
                os.path.join(args.output_dir, 'part_d_channel_probing.png'))

    # Part E: Counterfactual awareness
    all_results['part_e'] = run_part_e(model, tokenizer, test_samples, device,
                                        args.num_samples)
    plot_part_e(all_results['part_e'],
                os.path.join(args.output_dir, 'part_e_counterfactual_awareness.png'))

    # Summary
    plot_summary(all_results, os.path.join(args.output_dir, 'summary.png'))

    # Print overall summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: INTROSPECTIVE ACCESS SUMMARY")
    print(f"{'='*60}")

    pa = all_results['part_a']
    print(f"\n  Part A — Channel Report:")
    print(f"    Core (ch 0-7):  {pa['core_correct']}/{pa['core_total']} "
          f"= {pa['core_correct']/max(pa['core_total'],1):.1%}")
    print(f"    Extra (ch 8-19): {pa['extra_correct']}/{pa['extra_total']} "
          f"= {pa['extra_correct']/max(pa['extra_total'],1):.1%}")

    pb = all_results['part_b']
    if pb['total'] > 0:
        print(f"\n  Part B — Substitution Introspection:")
        ct = pb['core_total']
        et = pb['extra_total']
        if ct > 0:
            print(f"    Core (ch 0-3):  channel ID = {pb['core_channel_id']/ct:.1%} (n={ct})")
        if et > 0:
            print(f"    Extra (ch 8-39): channel ID = {pb['extra_channel_id']/et:.1%} (n={et})")
            print(f"    (Chance by output-reading: {1/32:.1%})")

    pd_r = all_results['part_d']
    if pd_r['overall_total'] > 0:
        print(f"\n  Part D — Channel-Specific Probing:")
        print(f"    Overall: {pd_r['overall_correct']}/{pd_r['overall_total']} "
              f"= {pd_r['overall_correct']/pd_r['overall_total']:.1%}")

    pe = all_results['part_e']
    if pe['total'] > 0:
        print(f"\n  Part E — Counterfactual Resistance:")
        print(f"    Resisted: {pe['resisted_false']}/{pe['total']} "
              f"= {pe['resisted_false']/pe['total']:.1%}")

    print(f"\n  Interpretation:")
    print(f"    Lindsey's arbitrary injection achieves ~20% introspective awareness.")
    print(f"    Rates significantly above 20% suggest trained FDM channels provide")
    print(f"    stronger, more specific introspective access than arbitrary perturbation.")
    print(f"    High Part E resistance indicates genuine epistemic access to signal content.")

    # Serialize results
    def to_json(obj):
        if isinstance(obj, defaultdict):
            return {str(k): to_json(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {str(k): to_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, list):
            return [to_json(item) for item in obj]
        return obj

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(to_json(all_results), f, indent=2, default=str)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
