"""
Experiment 3: Knowledge Conflict for 40-Channel Generative FDM Model
=====================================================================

Behavioral test: Does the model follow FDM-encoded information over
parametric priors learned during training?

Uses your TurboFDMSignalEncoder to generate fresh FDM encodings at
test time with counterfactual values.

Part A: COUNTERFACTUAL ENCODING
  Generate fresh FDM encodings with known values, then check whether
  the model's output reflects the encoded value. Since training uses
  uniform random.choice(), there is no "dominant" value to default to.
  The test checks: does the model output what was actually encoded,
  or something unrelated to the signal?

Part B: MEMORIZATION RATIO
  How often does the model output a single fixed value for each channel
  regardless of what was encoded? Training data uses random.choice()
  so values are ~uniformly distributed. MR near 1/num_values (chance)
  with high signal accuracy means signal reading. MR significantly above
  chance means the model gravitates toward a default regardless of input.

Part C: NOVEL COMBINATION TEST
  Encode memory states with unusual VALUE COMBINATIONS (e.g., 
  SAFETY_FIRST rule + EMERGENCY meta) that rarely co-occur in 
  training. Tests whether the model can compose from signal.

Part D: SIGNAL SUBSTITUTION
  Take a test sample, re-encode with DIFFERENT memory values using
  the same question. Does the answer change to match new encoding?
  This is the most direct test of signal reading.

Usage:
    python experiment3_knowledge_conflict_v2.py \
        --model_path ./fdm_40ch_turbo_v3_model_final \
        --test_data ./fdm_40ch_turbo_v3_stage4_test.jsonl \
        --train_data ./fdm_40ch_turbo_v3_stage4_train.jsonl \
        --num_samples 300 \
        --output_dir ./exp3_conflict_v2
"""

import argparse
import json
import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random


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

QUESTION_TEMPLATES = [
    "Should {AGENT} proceed with the mission?",
    "What is the risk assessment?",
    "Is it safe to share the secret with the contact?",
]


# ============================================================
# FDM Encoder (from your eidetic_real_fdm_e2e.py)
# ============================================================

class TurboFDMSignalEncoder:
    """Exact copy of your encoder for generating fresh FDM tokens."""

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
# Generation helper
# ============================================================

def generate_response(model, tokenizer, fdm_text, question, device):
    input_text = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=250, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(output[0])
    return full[len(input_text):].strip().split('<|endoftext|>')[0].strip()


# ============================================================
# PART A: Counterfactual encoding
# ============================================================

def run_part_a(model, tokenizer, encoder, device, num_samples):
    print("\n" + "=" * 60)
    print("Part A: Counterfactual Encoding")
    print("=" * 60)

    results = {'followed_signal': 0, 'followed_prior': 0, 'other': 0, 'total': 0,
               'per_channel': defaultdict(lambda: {'signal': 0, 'prior': 0, 'other': 0, 'total': 0})}

    # Key channels to test counterfactuals on
    test_channels = [0, 1, 2, 3, 6, 7]  # SECRET, LOCATION, AGENT, STATUS, RULE, META

    for trial in tqdm(range(num_samples), desc='Counterfactual'):
        # Random base memory
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)

        # Pick channel to make counterfactual
        cf_ch = random.choice(test_channels)
        original_val = memory[cf_ch]
        _, possible = MEMORY_SCHEMAS[cf_ch]
        others = [v for v in possible if v != original_val]
        if not others:
            continue
        cf_val = random.choice(others)
        memory[cf_ch] = cf_val

        # Encode
        try:
            fdm_text, _ = encoder.encode_memory(memory)
        except Exception as e:
            continue

        # Generate question
        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        question = random.choice(QUESTION_TEMPLATES).format(**facts)

        response = generate_response(model, tokenizer, fdm_text, question, device)

        results['total'] += 1
        ch_results = results['per_channel'][cf_ch]
        ch_results['total'] += 1

        # Check if response contains counterfactual value
        if cf_val in response:
            results['followed_signal'] += 1
            ch_results['signal'] += 1
        elif original_val in response:
            results['followed_prior'] += 1
            ch_results['prior'] += 1
        else:
            results['other'] += 1
            ch_results['other'] += 1

    total = results['total']
    if total > 0:
        print(f"\n  Followed encoded signal:  {results['followed_signal']}/{total} = {results['followed_signal']/total:.1%}")
        print(f"  Output non-encoded value: {results['followed_prior']}/{total} = {results['followed_prior']/total:.1%}")
        print(f"  Neither value found:      {results['other']}/{total} = {results['other']/total:.1%}")

        for ch in test_channels:
            ch_r = results['per_channel'][ch]
            if ch_r['total'] > 0:
                name = MEMORY_SCHEMAS[ch][0]
                sig_rate = ch_r['signal'] / ch_r['total']
                print(f"    Ch {ch} ({name}): signal={sig_rate:.1%} (n={ch_r['total']})")

    return dict(results)


# ============================================================
# PART B: Memorization ratio
# ============================================================

def run_part_b(model, tokenizer, test_samples, train_path, device, num_samples):
    print("\n" + "=" * 60)
    print("Part B: Memorization Ratio")
    print("=" * 60)

    # Find most frequent values per channel in training data
    if train_path and os.path.exists(train_path):
        train_samples = [json.loads(l) for l in open(train_path)]
        channel_freq = defaultdict(Counter)
        for s in train_samples:
            for k, v in s['memory'].items():
                channel_freq[int(k)][v] += 1

        most_frequent = {}
        for ch in range(NUM_CHANNELS):
            if channel_freq[ch]:
                most_frequent[ch] = channel_freq[ch].most_common(1)[0]
            else:
                most_frequent[ch] = (MEMORY_SCHEMAS[ch][1][0], 0)
    else:
        most_frequent = {ch: (MEMORY_SCHEMAS[ch][1][0], 0) for ch in range(NUM_CHANNELS)}

    match_freq = defaultdict(int)
    match_encoded = defaultdict(int)
    total_per_ch = defaultdict(int)

    n = min(num_samples, len(test_samples))

    for idx in tqdm(range(n), desc='Memorization ratio'):
        s = test_samples[idx]
        memory = s['memory']
        question = s['question']
        fdm_text = s['fdm_text']

        response = generate_response(model, tokenizer, fdm_text, question, device)

        for ch in range(NUM_CHANNELS):
            encoded_val = memory.get(str(ch), '')
            if not encoded_val:
                continue
            total_per_ch[ch] += 1

            if encoded_val in response:
                match_encoded[ch] += 1

            mf_val = most_frequent[ch][0]
            if mf_val in response:
                match_freq[ch] += 1

    print(f"\n  {'Channel':<25} {'MR':<12} {'Signal Acc':<12} {'Verdict'}")
    print(f"  {'-'*60}")

    memorization_ratios = {}
    for ch in range(min(NUM_CHANNELS, 20)):  # Show first 20
        total = total_per_ch[ch]
        if total == 0:
            continue
        mr = match_freq[ch] / total
        sig = match_encoded[ch] / total
        num_vals = len(MEMORY_SCHEMAS[ch][1])
        expected = 1.0 / num_vals

        # With uniform training, chance MR = 1/num_vals.
        # MR near chance + high signal acc = reading. MR >> chance = defaulting.
        verdict = "✓ READS SIGNAL" if sig > 0.5 and mr < expected * 2 else (
            "✗ DEFAULTS" if mr > expected * 3 else "? AMBIGUOUS")
        name = MEMORY_SCHEMAS[ch][0]
        print(f"  Ch {ch:2d} ({name:<12}) {mr:<12.3f} {sig:<12.3f} {verdict}")
        memorization_ratios[ch] = {'mr': mr, 'sig_acc': sig, 'verdict': verdict}

    overall_mr = sum(match_freq.values()) / max(sum(total_per_ch.values()), 1)
    overall_sig = sum(match_encoded.values()) / max(sum(total_per_ch.values()), 1)
    print(f"\n  Overall MR: {overall_mr:.3f}")
    print(f"  Overall signal accuracy: {overall_sig:.3f}")

    return memorization_ratios, overall_mr, overall_sig


# ============================================================
# PART D: Signal substitution (most direct test)
# ============================================================

def run_part_d(model, tokenizer, encoder, test_samples, device, num_samples):
    print("\n" + "=" * 60)
    print("Part D: Signal Substitution")
    print("=" * 60)
    print("  Same question, different FDM encoding → does answer change?")

    changed = 0
    correct_new = 0
    total = 0

    for idx in tqdm(range(min(num_samples, 100)), desc='Substitution'):
        s = test_samples[idx]
        original_memory = {int(k): v for k, v in s['memory'].items()}

        # Create alternate memory with different key values
        new_memory = dict(original_memory)
        for ch in [0, 1, 2]:  # SECRET, LOCATION, AGENT
            _, values = MEMORY_SCHEMAS[ch]
            others = [v for v in values if v != original_memory[ch]]
            if others:
                new_memory[ch] = random.choice(others)

        # Encode new memory
        try:
            new_fdm_text, _ = encoder.encode_memory(new_memory)
        except:
            continue

        # Same question
        question = s['question']

        # Generate with original encoding
        resp_orig = generate_response(model, tokenizer, s['fdm_text'], question, device)

        # Generate with new encoding
        resp_new = generate_response(model, tokenizer, new_fdm_text, question, device)

        total += 1

        # Did the response change?
        if resp_orig != resp_new:
            changed += 1

        # Does new response contain new values?
        new_facts_found = 0
        for ch in [0, 1, 2]:
            if new_memory[ch] in resp_new:
                new_facts_found += 1
        if new_facts_found >= 2:
            correct_new += 1

        if idx < 5:
            print(f"\n  Sample {idx}:")
            print(f"    Original: {MEMORY_SCHEMAS[0][0]}={original_memory[0]}, "
                  f"{MEMORY_SCHEMAS[1][0]}={original_memory[1]}, "
                  f"{MEMORY_SCHEMAS[2][0]}={original_memory[2]}")
            print(f"    New:      {MEMORY_SCHEMAS[0][0]}={new_memory[0]}, "
                  f"{MEMORY_SCHEMAS[1][0]}={new_memory[1]}, "
                  f"{MEMORY_SCHEMAS[2][0]}={new_memory[2]}")
            print(f"    Resp orig: {resp_orig[:80]}...")
            print(f"    Resp new:  {resp_new[:80]}...")

    if total > 0:
        print(f"\n  Response changed: {changed}/{total} = {changed/total:.1%}")
        print(f"  New facts correct: {correct_new}/{total} = {correct_new/total:.1%}")

    return {'changed': changed, 'correct_new': correct_new, 'total': total}


# ============================================================
# VISUALIZATION
# ============================================================

def plot_counterfactual(cf_results, output_path):
    total = cf_results['total']
    if total == 0:
        return
    labels = ['Followed Encoded Signal', 'Output Wrong Value', 'Neither Found']
    values = [cf_results['followed_signal'], cf_results['followed_prior'], cf_results['other']]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 12})
    ax.set_title(f'Counterfactual Following Rate (n={total})\n'
                 f'Training distribution is uniform — no dominant prior to default to')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_memorization(mr_data, output_path):
    channels = sorted(mr_data.keys())[:20]
    mrs = [mr_data[ch]['mr'] for ch in channels]
    sigs = [mr_data[ch]['sig_acc'] for ch in channels]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(channels))
    width = 0.35
    ax.bar(x - width/2, mrs, width, label='Memorization Ratio', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, sigs, width, label='Signal Accuracy', color='#2ecc71', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{MEMORY_SCHEMAS[ch][0][:8]}' for ch in channels],
                        rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Rate')
    ax.set_title('Memorization Ratio vs Signal Accuracy')
    ax.legend()
    ax.set_ylim(0, 1.1)
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
    parser.add_argument('--train_data', type=str, default='fdm_40ch_turbo_v3_stage4_train.jsonl')
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--a_low', type=float, default=0.25,
                        help='ASK low amplitude (match your stage 4 training)')
    parser.add_argument('--output_dir', type=str, default='./exp3_conflict_v2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    test_samples = [json.loads(l) for l in open(args.test_data)]

    # Init encoder matching stage 4 config
    encoder = TurboFDMSignalEncoder(
        num_tokens_per_encoder=256, a_high=1.0, a_low=args.a_low, num_levels=64
    )

    all_results = {}

    # Part A: Counterfactual
    cf = run_part_a(model, tokenizer, encoder, device, args.num_samples)
    all_results['counterfactual'] = cf
    plot_counterfactual(cf, os.path.join(args.output_dir, 'counterfactual.png'))

    # Part B: Memorization ratio
    mr_data, overall_mr, overall_sig = run_part_b(
        model, tokenizer, test_samples, args.train_data, device, args.num_samples
    )
    all_results['memorization'] = {'per_channel': mr_data, 'overall_mr': overall_mr,
                                    'overall_sig': overall_sig}
    plot_memorization(mr_data, os.path.join(args.output_dir, 'memorization_ratio.png'))

    # Part D: Signal substitution
    sub = run_part_d(model, tokenizer, encoder, test_samples, device, args.num_samples)
    all_results['substitution'] = sub

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if cf['total'] > 0:
        print(f"  Counterfactual signal following: {cf['followed_signal']/cf['total']:.1%}")
    print(f"  Memorization ratio: {overall_mr:.3f}")
    print(f"  Signal accuracy: {overall_sig:.3f}")
    if sub['total'] > 0:
        print(f"  Substitution response changed: {sub['changed']/sub['total']:.1%}")
        print(f"  Substitution new facts correct: {sub['correct_new']/sub['total']:.1%}")

    print(f"\n  Interpretation:")
    if overall_mr < 0.3:
        print("  ✓ Low MR → model doesn't default to a fixed value per channel")
    if overall_sig > 0.5:
        print("  ✓ High signal accuracy → outputs track encoded values")
    if overall_mr < 0.3 and overall_sig > 0.5:
        print("  ✓ Combined: model reads from FDM input, not parametric memory")
    if cf['total'] > 0 and cf['followed_signal'] / cf['total'] > 0.5:
        print("  ✓ High CF following → model outputs match freshly encoded values")
    if sub['total'] > 0 and sub['changed'] / sub['total'] > 0.7:
        print("  ✓ High substitution sensitivity → output causally depends on signal")

    # Serialize
    def to_json(obj):
        if isinstance(obj, defaultdict):
            return {str(k): to_json(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {str(k): to_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(to_json(all_results), f, indent=2)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
