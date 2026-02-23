"""
Joint accuracy computation — actual per-sample, not independence estimate.
Reuses saved test data, runs inference, tracks per-sample channel correctness.

Usage:
  MODEL_PATH=./fdm_40ch_turbo_v3_model_final python nhop_joint_accuracy.py
"""

import json, re, sys, os
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from eidetic_real_fdm_e2e_40ch_turbo_v3 import MEMORY_SCHEMAS, NUM_CHANNELS


def parse_context_values(text):
    """Extract KEY=VALUE pairs from Context: string in model output."""
    parsed = {}
    context_match = re.search(r'Context:\s*(.+?)\.?\s*$', text, re.DOTALL)
    if context_match:
        for pair in context_match.group(1).split(','):
            pair = pair.strip()
            if '=' in pair:
                key, val = pair.split('=', 1)
                parsed[key.strip()] = val.strip()
    return parsed


def main():
    model_path = os.environ.get("MODEL_PATH", "./fdm_40ch_turbo_v3_model_final")
    data_path = "nhop_v3_data/test_data.jsonl"

    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(data_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples, device={device}")

    extra_channels = list(range(8, NUM_CHANNELS))  # channels 8-39

    # Per-sample: how many extra channels correct?
    per_sample_correct_count = []
    # Per-sample: which channels wrong?
    per_sample_wrong = []
    # Joint: all N channels correct
    joint_counts = defaultdict(int)  # n -> count of samples with >= n correct
    # Per-channel
    ch_correct = defaultdict(int)
    ch_total = defaultdict(int)

    for i, sample in enumerate(tqdm(samples)):
        prompt = f"[MEMORY]{sample['fdm_text']}[/MEMORY]\nQuestion: {sample['question']}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        if input_ids.shape[1] > 1024 - 250:
            input_ids = input_ids[:, -(1024 - 250):]

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        generated = generated.split('<|endoftext|>')[0].strip()

        memory = sample["memory"]
        parsed = parse_context_values(generated)

        # Check each extra channel
        correct_this_sample = 0
        wrong_channels = []
        for ch in extra_channels:
            name = MEMORY_SCHEMAS[ch][0]
            expected = memory[str(ch)]
            ch_total[ch] += 1
            if parsed.get(name) == expected:
                ch_correct[ch] += 1
                correct_this_sample += 1
            else:
                wrong_channels.append((ch, name, expected, parsed.get(name, "MISSING")))

        per_sample_correct_count.append(correct_this_sample)
        per_sample_wrong.append(wrong_channels)

    total = len(samples)
    num_extra = len(extra_channels)

    # Compute joint accuracy at each threshold
    print(f"\n{'='*70}")
    print(f"JOINT ACCURACY — ACTUAL PER-SAMPLE ({total} samples, {num_extra} extra channels)")
    print(f"{'='*70}")

    print(f"\n--- Per-Channel Accuracy (extra channels 8-39) ---")
    for ch in extra_channels:
        pct = ch_correct[ch] / ch_total[ch] * 100
        marker = " ***" if pct < 95 else ""
        print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:12s}): {ch_correct[ch]}/{ch_total[ch]} = {pct:.1f}%{marker}")

    # Distribution of correct counts
    from collections import Counter
    dist = Counter(per_sample_correct_count)
    print(f"\n--- Distribution of correct extra channels per sample ---")
    for n in sorted(dist.keys()):
        print(f"  {n:2d}/{num_extra} correct: {dist[n]:4d} samples ({dist[n]/total*100:.1f}%)")

    # Joint accuracy: ALL n channels correct
    print(f"\n--- Joint Accuracy: ALL N extra channels correct ---")
    print(f"{'N channels':<12} {'Samples correct':<18} {'Accuracy':<12} {'Greenblatt':<15} {'Independence est.'}")
    print(f"{'-'*80}")

    greenblatt = {2: "~60%", 3: "~34%", 4: "~chance"}

    for n in [1, 2, 4, 8, 16, 24, 32]:
        if n > num_extra:
            break
        # Count samples where first n extra channels are ALL correct
        count = sum(1 for wrongs in per_sample_wrong
                    if all(ch not in [w[0] for w in wrongs] for ch in extra_channels[:n]))
        pct = count / total * 100

        # Independence estimate for comparison
        indep = 1.0
        for ch in extra_channels[:n]:
            indep *= ch_correct[ch] / ch_total[ch]
        indep *= 100

        gb = greenblatt.get(n, "")
        print(f"{n:<12} {count}/{total:<14} {pct:>5.1f}%       {gb:<15} {indep:.1f}%")

    # Also compute: ALL 32 extra channels correct
    all_correct = sum(1 for c in per_sample_correct_count if c == num_extra)
    print(f"\nALL {num_extra} extra channels correct: {all_correct}/{total} = {all_correct/total*100:.1f}%")

    # Error analysis: which samples fail and why?
    print(f"\n--- Error Analysis ---")
    error_samples = [(i, w) for i, w in enumerate(per_sample_wrong) if w]
    print(f"Samples with any error: {len(error_samples)}/{total}")

    ch_error_count = defaultdict(int)
    for _, wrongs in error_samples:
        for ch, name, exp, got in wrongs:
            ch_error_count[ch] += 1

    print(f"\nMost error-prone channels:")
    for ch, count in sorted(ch_error_count.items(), key=lambda x: -x[1])[:10]:
        print(f"  ch{ch} ({MEMORY_SCHEMAS[ch][0]}): {count} errors")

    # Are errors correlated?
    multi_error = sum(1 for _, w in error_samples if len(w) > 1)
    single_error = sum(1 for _, w in error_samples if len(w) == 1)
    print(f"\nSingle-channel errors: {single_error}")
    print(f"Multi-channel errors:  {multi_error}")
    if multi_error > 0:
        print(f"  → Errors are {'correlated' if multi_error > single_error else 'mostly independent'}")

    # Save
    results = {
        "per_channel": {str(ch): ch_correct[ch] / ch_total[ch] * 100 for ch in extra_channels},
        "all_32_correct": all_correct / total * 100,
        "distribution": {str(k): v for k, v in sorted(dist.items())},
        "error_samples": len(error_samples),
        "single_errors": single_error,
        "multi_errors": multi_error,
    }
    with open("nhop_v3_data/joint_accuracy.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to nhop_v3_data/joint_accuracy.json")


if __name__ == "__main__":
    main()
