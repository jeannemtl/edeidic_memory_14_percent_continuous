"""
Experiment 2: Attention Ablation for 40-Channel Generative FDM Model
=====================================================================

Tests whether the model MUST attend to FDM signal tokens to generate
correct responses. Since true FDM means every token carries ALL channels,
we can't ablate individual channel positions. Instead:

Part A: Overall — ablate ALL FDM tokens vs NON-FDM tokens
Part B: Per-layer — which layers depend most on reading signal tokens?
Part C: Dose-response — ablate 10%, 25%, 50%, 75%, 100% of FDM tokens
Part D: Encoder 1 vs 2 — ablate natural-order tokens vs interleaved tokens
         (tests whether turbo structure provides independent information)
Part E: Temporal blocks — ablate early/mid/late signal tokens
         (tests whether specific time segments carry specific channels)
Part F: Per-question-type breakdown — same ablation, split by question template
         (tests whether entity names leaked in questions serve as retrieval keys)

Accuracy is measured via keyword matching in generated text, matching
your eval methodology from eidetic_real_fdm_e2e.py.

Usage:
    python experiment2_attention_ablation_v2.py \
        --model_path ./fdm_40ch_turbo_v3_model_final \
        --test_data ./fdm_40ch_turbo_v3_stage4_test.jsonl \
        --num_samples 200 \
        --output_dir ./exp2_ablation_v2
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


# ============================================================
# Schemas & scoring (from eidetic_real_fdm_e2e.py)
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
for i in range(8, 40):
    MEMORY_SCHEMAS[i] = (f"CH{i}", [f"V{j}" for j in range(4)])

NUM_CHANNELS = 40


def score_response(response, sample):
    """Multi-dimensional scoring of generated text."""
    expected = sample['answer']
    facts = sample.get('facts', {})
    memory = sample['memory']
    question = sample.get('question', '')

    scores = {}

    # Action
    scores['action'] = 0
    for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
               'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK',
               'MODERATE', 'immediate']:
        if kw.lower() in expected.lower() and kw.lower() in response.lower():
            scores['action'] = 1
            break

    # Core fact (ch 0-5)
    scores['fact'] = 0
    if question.startswith("Should"):
        if facts.get('AGENT', '') in response: scores['fact'] = 1
    elif question.startswith("What"):
        if facts.get('LOCATION', '') in response: scores['fact'] = 1
    elif question.startswith("Is it"):
        if facts.get('SECRET', '') in response or facts.get('LOCATION', '') in response:
            scores['fact'] = 1

    # Rule
    scores['rule'] = 1 if sample.get('rule', '') in response else 0

    # Extra channels
    extra_correct = 0
    extra_total = 0
    for ch in range(8, NUM_CHANNELS):
        val = memory.get(str(ch), '')
        if val:
            extra_total += 1
            if val in response: extra_correct += 1
    scores['extra'] = extra_correct / max(extra_total, 1)

    return scores


def classify_question_type(question):
    """
    Classify question into types based on entity leakage.
    
    Type 1 ("Should {AGENT} proceed") — leaks AGENT name (e.g. ALICE)
    Type 2 ("What is the risk assessment?") — leaks NOTHING
    Type 3 ("Is it safe to share the secret") — leaks concept but no entity name
    
    Types 1 and 3 give the model retrieval keys it could use without FDM.
    Type 2 is a pure format cue with zero factual content.
    """
    if question.startswith("Should"):
        return "proceed (leaks AGENT)"
    elif question.startswith("What"):
        return "risk (no leakage)"
    elif question.startswith("Is it"):
        return "share (leaks concept)"
    return "unknown"


def generate_with_mask(model, tokenizer, sample, device, mask_positions=None,
                       max_new_tokens=250):
    """
    Generate response, optionally masking specific positions in attention.
    mask_positions: list of token indices to mask from attention.
    """
    fdm_text = sample['fdm_text']
    question = sample['question']
    input_text = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    if mask_positions is not None and len(mask_positions) > 0:
        # Create attention mask that blocks the specified positions
        attention_mask = torch.ones_like(input_ids)
        for pos in mask_positions:
            if pos < input_ids.size(1):
                attention_mask[0, pos] = 0
    else:
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output[0])
    response = full_text[len(input_text):].strip().split('<|endoftext|>')[0].strip()
    return response


def find_fdm_range(tokenizer, fdm_text, question):
    """Find start/end indices of FDM tokens in full input."""
    prefix = "[MEMORY]"
    suffix = "[/MEMORY]"
    input_text = f"{prefix}{fdm_text}{suffix}\nQuestion: {question}\nAnswer:"
    full_ids = tokenizer.encode(input_text)

    prefix_ids = tokenizer.encode(prefix)
    suffix_part = tokenizer.encode(f"{suffix}\nQuestion: {question}\nAnswer:")

    fdm_start = len(prefix_ids)
    fdm_end = len(full_ids) - len(suffix_part)
    return max(0, fdm_start), max(fdm_start, fdm_end), len(full_ids)


# ============================================================
# EXPERIMENTS
# ============================================================

def run_part_a(model, tokenizer, samples, device, num_samples):
    """Part A: Overall ablation comparison."""
    print("\n" + "=" * 60)
    print("Part A: Overall Ablation Comparison")
    print("=" * 60)

    conditions = {
        'Baseline (no ablation)': 'none',
        'Ablate ALL FDM tokens': 'all_fdm',
        'Ablate non-FDM tokens': 'non_fdm',
        'Ablate random 50% FDM': 'random_50',
    }

    results = {}
    for name, cond in conditions.items():
        scores_list = defaultdict(list)
        print(f"\n  {name}...")

        for idx in tqdm(range(num_samples), desc=f'  {name[:25]}'):
            s = samples[idx]
            fs, fe, slen = find_fdm_range(tokenizer, s['fdm_text'], s['question'])
            fdm_positions = list(range(fs, fe))
            non_fdm = [i for i in range(slen) if i not in set(fdm_positions)]

            if cond == 'none':
                mask = None
            elif cond == 'all_fdm':
                mask = fdm_positions
            elif cond == 'non_fdm':
                mask = non_fdm[:30]  # cap to avoid masking everything
            elif cond == 'random_50':
                mask = sorted(random.sample(fdm_positions,
                              k=len(fdm_positions) // 2))
            else:
                mask = None

            response = generate_with_mask(model, tokenizer, s, device, mask)
            scores = score_response(response, s)
            for k, v in scores.items():
                scores_list[k].append(v)

        results[name] = {k: np.mean(v) for k, v in scores_list.items()}
        print(f"    action={results[name]['action']:.3f}, "
              f"fact={results[name]['fact']:.3f}, "
              f"rule={results[name]['rule']:.3f}, "
              f"extra={results[name]['extra']:.3f}")

    return results


def run_part_b(model, tokenizer, samples, device, num_samples):
    """Part B: Per-layer ablation via embedding replacement."""
    print("\n" + "=" * 60)
    print("Part B: Per-Layer Ablation (signal vs non-signal)")
    print("=" * 60)

    num_layers = model.config.n_layer
    signal_results = {l: defaultdict(list) for l in range(num_layers)}
    nonsignal_results = {l: defaultdict(list) for l in range(num_layers)}

    n = min(num_samples, 50)  # Fewer samples since this is O(layers * samples)

    for idx in tqdm(range(n), desc='Per-layer'):
        s = samples[idx]
        fs, fe, slen = find_fdm_range(tokenizer, s['fdm_text'], s['question'])
        fdm_positions = list(range(fs, fe))
        non_fdm = [i for i in range(slen) if i not in set(fdm_positions)][:20]

        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        full_text = input_text + f" {s['answer']}"
        enc = tokenizer(full_text, truncation=True, max_length=1024, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        labels = input_ids.clone()
        input_len = len(tokenizer.encode(input_text))
        labels[:, :input_len] = -100

        for target_layer in range(num_layers):
            for ablation_type, positions in [('signal', fdm_positions),
                                              ('nonsignal', non_fdm)]:
                hooks = []

                def make_hook(pos_list):
                    def hook_fn(module, input, output):
                        hidden = output[0]
                        for p in pos_list:
                            if p < hidden.size(1):
                                hidden[:, p, :] = 0.0  # zero ablation
                        return (hidden,) + output[1:]
                    return hook_fn

                h = model.transformer.h[target_layer].register_forward_hook(
                    make_hook(positions))
                hooks.append(h)

                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)

                # Compute loss on answer tokens
                logits = out.logits
                shift_logits = logits[:, input_len-1:-1, :]
                shift_labels = labels[:, input_len:]
                mask = shift_labels != -100
                if mask.sum() > 0:
                    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                    per_tok = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                                      shift_labels.reshape(-1))
                    loss = (per_tok * mask.reshape(-1).float()).sum() / mask.sum()
                    loss_val = loss.item()
                else:
                    loss_val = 0.0

                if ablation_type == 'signal':
                    signal_results[target_layer]['loss'].append(loss_val)
                else:
                    nonsignal_results[target_layer]['loss'].append(loss_val)

                for h in hooks:
                    h.remove()

    return (
        {l: np.mean(v['loss']) for l, v in signal_results.items()},
        {l: np.mean(v['loss']) for l, v in nonsignal_results.items()}
    )


def run_part_c(model, tokenizer, samples, device, num_samples):
    """Part C: Dose-response — vary fraction of FDM tokens ablated."""
    print("\n" + "=" * 60)
    print("Part C: Dose-Response Curve")
    print("=" * 60)

    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = {}

    for frac in fractions:
        scores_list = defaultdict(list)
        print(f"\n  Ablating {frac*100:.0f}% of FDM tokens...")

        for idx in tqdm(range(min(num_samples, 100)), desc=f'  {frac*100:.0f}%'):
            s = samples[idx]
            fs, fe, slen = find_fdm_range(tokenizer, s['fdm_text'], s['question'])
            fdm_pos = list(range(fs, fe))

            if frac == 0.0:
                mask = None
            else:
                k = max(1, int(len(fdm_pos) * frac))
                mask = sorted(random.sample(fdm_pos, k=min(k, len(fdm_pos))))

            response = generate_with_mask(model, tokenizer, s, device, mask)
            scores = score_response(response, s)
            for k, v in scores.items():
                scores_list[k].append(v)

        results[frac] = {k: np.mean(v) for k, v in scores_list.items()}
        print(f"    action={results[frac]['action']:.3f}, fact={results[frac]['fact']:.3f}")

    return results


def run_part_d(model, tokenizer, samples, device, num_samples):
    """Part D: Encoder 1 vs Encoder 2 ablation."""
    print("\n" + "=" * 60)
    print("Part D: Encoder 1 (natural) vs Encoder 2 (interleaved)")
    print("=" * 60)

    conditions = {
        'Baseline': 'none',
        'Ablate Encoder 1 only': 'enc1',
        'Ablate Encoder 2 only': 'enc2',
        'Ablate both': 'both',
    }

    results = {}
    for name, cond in conditions.items():
        scores_list = defaultdict(list)

        for idx in tqdm(range(min(num_samples, 100)), desc=f'  {name[:25]}'):
            s = samples[idx]
            fs, fe, slen = find_fdm_range(tokenizer, s['fdm_text'], s['question'])
            fdm_pos = list(range(fs, fe))
            mid = len(fdm_pos) // 2
            enc1_pos = fdm_pos[:mid]
            enc2_pos = fdm_pos[mid:]

            if cond == 'none':
                mask = None
            elif cond == 'enc1':
                mask = enc1_pos
            elif cond == 'enc2':
                mask = enc2_pos
            elif cond == 'both':
                mask = fdm_pos
            else:
                mask = None

            response = generate_with_mask(model, tokenizer, s, device, mask)
            scores = score_response(response, s)
            for k, v in scores.items():
                scores_list[k].append(v)

        results[name] = {k: np.mean(v) for k, v in scores_list.items()}
        print(f"  {name}: action={results[name]['action']:.3f}, "
              f"fact={results[name]['fact']:.3f}")

    return results


def run_part_f(model, tokenizer, samples, device, num_samples):
    """
    Part F: Per-Question-Type Ablation Breakdown.
    
    Runs baseline vs all-FDM-ablated for each question type separately.
    
    Key insight: Questions that leak entity names (Type 1: "Should ALICE proceed",
    Type 3: "Is it safe to share") give the model retrieval keys it could use
    without the FDM signal. Type 2 ("What is the risk assessment?") leaks nothing.
    
    If model reads from signal: all types degrade equally under FDM ablation.
    If model recalls from weights: Types 1 & 3 resist degradation (they have 
    retrieval keys in the question text), Type 2 degrades most severely.
    """
    print("\n" + "=" * 60)
    print("Part F: Per-Question-Type Ablation Breakdown")
    print("=" * 60)

    # Group samples by question type
    by_type = defaultdict(list)
    for s in samples[:num_samples]:
        qtype = classify_question_type(s['question'])
        by_type[qtype].append(s)

    for qtype, type_samples in by_type.items():
        print(f"  {qtype}: {len(type_samples)} samples")

    results = {}
    for qtype, type_samples in by_type.items():
        n = len(type_samples)
        if n == 0:
            continue

        for cond_name, do_ablate in [('baseline', False), ('ablate_fdm', True)]:
            key = f"{qtype} | {cond_name}"
            scores_list = defaultdict(list)

            for s in tqdm(type_samples, desc=f'  {qtype[:20]} {cond_name}'):
                fs, fe, slen = find_fdm_range(tokenizer, s['fdm_text'], s['question'])
                fdm_positions = list(range(fs, fe))
                mask = fdm_positions if do_ablate else None

                response = generate_with_mask(model, tokenizer, s, device, mask)
                scores = score_response(response, s)
                for k, v in scores.items():
                    scores_list[k].append(v)

            results[key] = {k: np.mean(v) for k, v in scores_list.items()}
            print(f"    {key}: action={results[key]['action']:.3f}, "
                  f"fact={results[key]['fact']:.3f}, "
                  f"rule={results[key]['rule']:.3f}")

    # Compute degradation per type
    print("\n  --- Degradation by Question Type ---")
    for qtype in by_type:
        bl = results.get(f"{qtype} | baseline", {})
        ab = results.get(f"{qtype} | ablate_fdm", {})
        if bl and ab:
            for metric in ['action', 'fact', 'rule', 'extra']:
                drop = bl.get(metric, 0) - ab.get(metric, 0)
                print(f"    {qtype} {metric}: {bl.get(metric,0):.3f} → "
                      f"{ab.get(metric,0):.3f} (Δ = {drop:+.3f})")
            print()

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_part_f(results, output_path):
    """Plot per-question-type ablation as grouped bars with degradation."""
    qtypes = sorted(set(k.split(' | ')[0] for k in results.keys()))
    metrics = ['action', 'fact', 'rule', 'extra']

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6),
                              sharey=True)

    x = np.arange(len(qtypes))
    width = 0.35

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        baselines = []
        ablated = []
        for qt in qtypes:
            bl = results.get(f"{qt} | baseline", {}).get(metric, 0)
            ab = results.get(f"{qt} | ablate_fdm", {}).get(metric, 0)
            baselines.append(bl)
            ablated.append(ab)

        ax.bar(x - width / 2, baselines, width, label='Baseline',
               color='#2196F3', alpha=0.8)
        ax.bar(x + width / 2, ablated, width, label='FDM Ablated',
               color='#F44336', alpha=0.8)

        for i in range(len(qtypes)):
            drop = baselines[i] - ablated[i]
            ypos = max(baselines[i], ablated[i]) + 0.03
            ax.annotate(f'Δ={drop:+.2f}', xy=(x[i], ypos),
                       ha='center', fontsize=8, fontweight='bold',
                       color='red' if drop > 0.1 else 'gray')

        ax.set_title(metric.capitalize(), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([qt.split(' (')[0] for qt in qtypes],
                          rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.3)
        ax.grid(True, alpha=0.2, axis='y')
        if mi == 0:
            ax.set_ylabel('Score')
            ax.legend(fontsize=8)

    fig.suptitle('FDM Ablation by Question Type\n'
                 '(Equal degradation = signal reading; '
                 'unequal = weight recall with entity-key retrieval)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_part_a(results, output_path):
    conditions = list(results.keys())
    metrics = ['action', 'fact', 'rule', 'extra']
    x = np.arange(len(conditions))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        vals = [results[c].get(metric, 0) for c in conditions]
        ax.bar(x + i * width, vals, width, label=metric, alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(conditions, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Ablation Comparison: FDM vs Non-FDM Positions')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_part_b(signal_loss, nonsignal_loss, output_path):
    layers = sorted(signal_loss.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, [signal_loss[l] for l in layers], 'r-o',
            label='Ablate signal positions', markersize=4)
    ax.plot(layers, [nonsignal_loss[l] for l in layers], 'b-s',
            label='Ablate non-signal positions', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Answer Loss (higher = more degraded)')
    ax.set_title('Per-Layer Ablation: Signal vs Non-Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_part_c(results, output_path):
    fracs = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ['action', 'fact', 'rule', 'extra']:
        vals = [results[f].get(metric, 0) for f in fracs]
        ax.plot([f * 100 for f in fracs], vals, '-o', label=metric, markersize=6)
    ax.set_xlabel('% of FDM Tokens Ablated')
    ax.set_ylabel('Score')
    ax.set_title('Dose-Response: Accuracy vs Fraction of FDM Tokens Ablated')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_part_d(results, output_path):
    conditions = list(results.keys())
    metrics = ['action', 'fact', 'extra']
    x = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        vals = [results[c].get(metric, 0) for c in conditions]
        ax.bar(x + i * width, vals, width, label=metric, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Encoder 1 vs Encoder 2 Ablation (Turbo Structure)')
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
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='./exp2_ablation_v2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    samples = [json.loads(l) for l in open(args.test_data)]
    n = min(args.num_samples, len(samples))

    # Part A
    part_a = run_part_a(model, tokenizer, samples, device, n)
    plot_part_a(part_a, os.path.join(args.output_dir, 'part_a_comparison.png'))

    # Part B
    signal_loss, nonsignal_loss = run_part_b(model, tokenizer, samples, device, n)
    plot_part_b(signal_loss, nonsignal_loss,
                os.path.join(args.output_dir, 'part_b_per_layer.png'))

    # Part C
    part_c = run_part_c(model, tokenizer, samples, device, n)
    plot_part_c(part_c, os.path.join(args.output_dir, 'part_c_dose_response.png'))

    # Part D
    part_d = run_part_d(model, tokenizer, samples, device, n)
    plot_part_d(part_d, os.path.join(args.output_dir, 'part_d_encoder_compare.png'))

    # Part F: Per-question-type breakdown
    part_f = run_part_f(model, tokenizer, samples, device, n)
    plot_part_f(part_f, os.path.join(args.output_dir, 'part_f_question_type.png'))

    # Save all
    all_results = {
        'part_a': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in part_a.items()},
        'part_b_signal': {str(k): float(v) for k, v in signal_loss.items()},
        'part_b_nonsignal': {str(k): float(v) for k, v in nonsignal_loss.items()},
        'part_c': {str(k): {k2: float(v2) for k2, v2 in v.items()} for k, v in part_c.items()},
        'part_d': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in part_d.items()},
        'part_f': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in part_f.items()},
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}")
    print("\nKey findings to report:")
    print("  A: FDM ablation kills accuracy; non-FDM ablation doesn't")
    print("  B: Which layers show biggest loss difference → signal processing layers")
    print("  C: Graceful degradation curve → model uses distributed signal info")
    print("  D: Each encoder alone retains partial accuracy → turbo structure works")
    print("  F: Equal degradation across question types → signal reading")
    print("     Unequal (types 1&3 resist, type 2 collapses) → weight recall")


if __name__ == '__main__':
    main()
