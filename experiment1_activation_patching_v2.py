"""
Experiment 1: Activation Patching for 40-Channel Generative FDM Model
======================================================================

Adapted for your eidetic_real_fdm_e2e.py architecture:
  - GPT2LMHeadModel (gpt2-medium, 24 layers)
  - 40 FDM channels on sinusoidal carriers (1-40 Hz)
  - 512 turbo tokens (256 natural + 256 S-random interleaved)
  - Generative output: text containing facts, actions, reasoning
  - Input format: [MEMORY]{fdm_tokens}[/MEMORY]\nQuestion: ...\nAnswer:

Since every FDM token carries ALL 40 channels simultaneously (unlike TDM),
we cannot ablate "channel X's positions." Instead we ablate contiguous
blocks of signal tokens and measure which fact categories degrade.

The key comparison vs ROME:
  ROME signature: mid-layer MLPs at last subject token
  Expected FDM signature: early/mid-layer attention at FDM token positions

Usage:
    python experiment1_activation_patching_v2.py \
        --model_path ./fdm_40ch_turbo_v3_model_final \
        --test_data ./fdm_40ch_turbo_v3_stage4_test.jsonl \
        --num_samples 100 \
        --output_dir ./exp1_patching_v2
"""

import argparse
import json
import os
import re
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# ============================================================
# Channel definitions (from your eidetic_real_fdm_e2e.py)
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
# Channels 8-39 are extra operational context
for i in range(8, 40):
    MEMORY_SCHEMAS[i] = (f"CH{i}", [f"V{j}" for j in range(4)])

NUM_CHANNELS = 40


# ============================================================
# SCORING: Extract accuracy from generated text
# ============================================================

def score_generation(response, sample):
    """
    Score a generated response against expected answer.
    Returns dict of {metric: 0 or 1} for action, facts, rule, extra.
    """
    expected = sample['answer']
    facts = sample['facts']
    rule = sample.get('rule', '')
    meta = sample.get('meta', '')
    memory = sample['memory']
    scores = {}

    # Action match
    action_keywords = ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                       'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK',
                       'MODERATE', 'immediate']
    scores['action'] = 0
    for kw in action_keywords:
        if kw.lower() in expected.lower() and kw.lower() in response.lower():
            scores['action'] = 1
            break

    # Fact match (channels 0-5)
    scores['fact'] = 0
    question = sample.get('question', '')
    if question.startswith("Should"):
        if facts.get('AGENT', '') in response:
            scores['fact'] = 1
    elif question.startswith("What"):
        if facts.get('LOCATION', '') in response:
            scores['fact'] = 1
    elif question.startswith("Is it"):
        if facts.get('SECRET', '') in response or facts.get('LOCATION', '') in response:
            scores['fact'] = 1

    # Rule match
    scores['rule'] = 1 if rule in response else 0

    # Extra channels (8-39) — check if values appear in response
    extra_correct = 0
    extra_total = 0
    for ch in range(8, NUM_CHANNELS):
        name = MEMORY_SCHEMAS[ch][0]
        val = memory.get(str(ch), '')
        if val:
            extra_total += 1
            if val in response:
                extra_correct += 1
    scores['extra'] = extra_correct / max(extra_total, 1)

    return scores


def generate_response(model, tokenizer, fdm_text, question, device, max_new_tokens=250):
    """Generate model response for a given FDM input + question."""
    input_text = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output[0])
    response = full_text[len(input_text):].strip()
    response = response.split('<|endoftext|>')[0].strip()
    return response, input_ids


# ============================================================
# LOCATE FDM TOKEN POSITIONS in the tokenized input
# ============================================================

def find_fdm_token_range(tokenizer, fdm_text, question):
    """
    Find the start and end token indices of the FDM signal tokens
    within the full input sequence.
    """
    input_text = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    full_ids = tokenizer.encode(input_text)

    memory_start_text = "[MEMORY]"
    memory_end_text = "[/MEMORY]"

    # Find token positions of [MEMORY] and [/MEMORY]
    prefix_ids = tokenizer.encode(f"[MEMORY]")
    # The FDM tokens start right after [MEMORY] marker
    # and end right before [/MEMORY] marker

    # Simpler approach: encode components separately
    prefix = tokenizer.encode("[MEMORY]")
    suffix_start = tokenizer.encode("[/MEMORY]")

    # Find [MEMORY] in full_ids
    fdm_start = len(prefix)  # tokens after [MEMORY]

    # Find [/MEMORY] — search backwards from where question starts
    question_part = tokenizer.encode(f"[/MEMORY]\nQuestion: {question}\nAnswer:")
    fdm_end = len(full_ids) - len(question_part)

    # Clamp
    fdm_start = max(0, fdm_start)
    fdm_end = max(fdm_start, fdm_end)

    return fdm_start, fdm_end, len(full_ids)


# ============================================================
# ACTIVATION PATCHING ENGINE (generative model version)
# ============================================================

class GenerativeActivationPatcher:
    """
    Activation patching for generative FDM models.

    Instead of checking classification logits, we:
    1. Generate text from clean input → score it
    2. Generate text from corrupted input → score it
    3. For each (layer, position_block), restore clean activations
       and generate again → score recovery

    Since full generation per (layer, position) is too expensive,
    we use a PROXY: measure the cross-entropy loss on the expected
    answer tokens rather than doing full generation. This is O(1)
    forward passes per patch point rather than O(gen_length).
    """

    def __init__(self, model, tokenizer, device, noise_std_multiplier=3.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.n_layer  # 24 for gpt2-medium

        # Noise calibration
        embed_weight = model.transformer.wte.weight.data
        self.embed_std = embed_weight.std().item()
        self.noise_std = noise_std_multiplier * self.embed_std
        print(f"Noise: embed_std={self.embed_std:.4f}, noise_std={self.noise_std:.4f}")
        print(f"Model layers: {self.num_layers}")

    def _get_answer_loss(self, model_output, labels, answer_start):
        """Compute loss only on answer tokens."""
        logits = model_output.logits  # [1, seq, vocab]
        shift_logits = logits[:, answer_start-1:-1, :]
        shift_labels = labels[:, answer_start:]

        mask = shift_labels != -100
        if mask.sum() == 0:
            return 0.0

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        # Only count non-masked tokens
        masked_loss = per_token_loss * mask.reshape(-1).float()
        return masked_loss.sum().item() / max(mask.sum().item(), 1)

    def prepare_sample(self, sample):
        """Tokenize a sample with answer labels."""
        fdm_text = sample['fdm_text']
        question = sample['question']
        answer = sample['answer']

        input_text = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        full_text = input_text + f" {answer}"

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=1024,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Create labels: mask everything before the answer
        labels = input_ids.clone()
        input_len = len(self.tokenizer.encode(input_text))
        labels[:, :input_len] = -100

        return input_ids, attention_mask, labels, input_len

    def clean_loss(self, input_ids, attention_mask, labels, answer_start):
        """Loss on clean input."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._get_answer_loss(outputs, labels, answer_start)

    def corrupted_loss(self, input_ids, attention_mask, labels, answer_start,
                       fdm_start, fdm_end):
        """Loss when FDM token embeddings are corrupted with noise."""
        embeds = self.model.transformer.wte(input_ids) + \
                 self.model.transformer.wpe(
                     torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
                 )

        noise = torch.zeros_like(embeds)
        for pos in range(fdm_start, fdm_end):
            noise[:, pos, :] = torch.randn_like(embeds[:, pos, :]) * self.noise_std
        corrupted = embeds + noise

        with torch.no_grad():
            outputs = self.model(inputs_embeds=corrupted, attention_mask=attention_mask)
        return self._get_answer_loss(outputs, labels, answer_start)

    def patched_loss(self, input_ids, attention_mask, labels, answer_start,
                     fdm_start, fdm_end, clean_hidden_states,
                     patch_layer, patch_positions, mode='denoise'):
        """
        Loss with activation patching at specific (layer, positions).

        mode='denoise': corrupt input, restore clean states → tests sufficiency
        mode='noise': clean input, inject noise → tests necessity
        """
        hooks = []

        def make_denoise_hook(layer_idx, positions, clean_states):
            def hook_fn(module, input, output):
                hidden = output[0]
                for pos in positions:
                    if pos < hidden.size(1):
                        hidden[:, pos, :] = clean_states[layer_idx + 1][:, pos, :]
                return (hidden,) + output[1:]
            return hook_fn

        def make_noise_hook(positions, noise_std):
            def hook_fn(module, input, output):
                hidden = output[0]
                for pos in positions:
                    if pos < hidden.size(1):
                        hidden[:, pos, :] += torch.randn_like(hidden[:, pos, :]) * noise_std
                return (hidden,) + output[1:]
            return hook_fn

        layer_module = self.model.transformer.h[patch_layer]

        if mode == 'denoise':
            hook = layer_module.register_forward_hook(
                make_denoise_hook(patch_layer, patch_positions, clean_hidden_states)
            )
        else:
            hook = layer_module.register_forward_hook(
                make_noise_hook(patch_positions, self.noise_std)
            )
        hooks.append(hook)

        if mode == 'denoise':
            # Corrupted input
            embeds = self.model.transformer.wte(input_ids) + \
                     self.model.transformer.wpe(
                         torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
                     )
            noise = torch.zeros_like(embeds)
            for pos in range(fdm_start, fdm_end):
                noise[:, pos, :] = torch.randn_like(embeds[:, pos, :]) * self.noise_std
            with torch.no_grad():
                outputs = self.model(inputs_embeds=embeds + noise,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True)
        else:
            # Clean input
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True)

        for h in hooks:
            h.remove()

        return self._get_answer_loss(outputs, labels, answer_start)

    def sweep_sample(self, sample, block_size=16, mode='denoise'):
        """
        Sweep activation patching across layers and position blocks.

        Since there are ~512 FDM tokens, we group them into blocks
        of `block_size` to keep the sweep tractable.

        Returns heatmap[num_layers, num_blocks] of indirect effects.
        """
        input_ids, attention_mask, labels, answer_start = self.prepare_sample(sample)
        fdm_start, fdm_end, seq_len = find_fdm_token_range(
            self.tokenizer, sample['fdm_text'], sample['question']
        )

        # Also include some non-FDM positions for comparison
        non_fdm_positions = list(range(0, fdm_start)) + list(range(fdm_end, min(seq_len, fdm_end + 20)))

        # Build position blocks
        fdm_blocks = []
        for start in range(fdm_start, fdm_end, block_size):
            end = min(start + block_size, fdm_end)
            fdm_blocks.append(('fdm', list(range(start, end))))

        non_fdm_blocks = []
        for start in range(0, len(non_fdm_positions), block_size):
            end = min(start + block_size, len(non_fdm_positions))
            non_fdm_blocks.append(('non_fdm', non_fdm_positions[start:end]))

        all_blocks = fdm_blocks + non_fdm_blocks
        block_labels = [b[0] for b in all_blocks]
        block_positions = [b[1] for b in all_blocks]

        # Clean forward (cache hidden states)
        with torch.no_grad():
            clean_out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
        clean_hidden = clean_out.hidden_states
        c_loss = self._get_answer_loss(clean_out, labels, answer_start)

        # Corrupted baseline
        cr_loss = self.corrupted_loss(input_ids, attention_mask, labels,
                                       answer_start, fdm_start, fdm_end)

        # Sweep
        heatmap = np.zeros((self.num_layers, len(all_blocks)))

        for li in range(self.num_layers):
            for bi, positions in enumerate(block_positions):
                p_loss = self.patched_loss(
                    input_ids, attention_mask, labels, answer_start,
                    fdm_start, fdm_end, clean_hidden,
                    patch_layer=li, patch_positions=positions,
                    mode=mode
                )

                if mode == 'denoise':
                    # IE = how much loss decreases (recovers) from corrupted
                    heatmap[li, bi] = cr_loss - p_loss
                else:
                    # IE = how much loss increases from clean
                    heatmap[li, bi] = p_loss - c_loss

        return heatmap, block_labels, c_loss, cr_loss, fdm_start, fdm_end


# ============================================================
# VISUALIZATION
# ============================================================

def plot_heatmap(heatmap, block_labels, title, output_path,
                 clean_loss=None, corrupted_loss=None):
    fig, ax = plt.subplots(figsize=(max(12, len(block_labels) * 0.4), 8))
    im = ax.imshow(heatmap, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Indirect Effect (loss reduction)')

    ax.set_xlabel('Position Block')
    ax.set_ylabel('Layer')
    ax.set_title(title)

    colors = {'fdm': '#e74c3c', 'non_fdm': '#3498db'}
    for i, label in enumerate(block_labels):
        color = colors.get(label, 'gray')
        ax.axvline(x=i - 0.5, color=color, alpha=0.1, linewidth=0.5)

    # Legend via colored regions at top
    fdm_cols = [i for i, l in enumerate(block_labels) if l == 'fdm']
    nfdm_cols = [i for i, l in enumerate(block_labels) if l == 'non_fdm']
    if fdm_cols:
        ax.axvspan(fdm_cols[0] - 0.5, fdm_cols[-1] + 0.5, alpha=0.05, color='red')
    if nfdm_cols:
        ax.axvspan(nfdm_cols[0] - 0.5, nfdm_cols[-1] + 0.5, alpha=0.05, color='blue')

    info = []
    if clean_loss is not None:
        info.append(f'Clean loss: {clean_loss:.3f}')
    if corrupted_loss is not None:
        info.append(f'Corrupted loss: {corrupted_loss:.3f}')
    if info:
        ax.text(0.02, 0.98, ' | '.join(info), transform=ax.transAxes,
                fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_profile(heatmap, block_labels, title, output_path):
    fdm_cols = [i for i, l in enumerate(block_labels) if l == 'fdm']
    nfdm_cols = [i for i, l in enumerate(block_labels) if l == 'non_fdm']

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = np.arange(heatmap.shape[0])

    if fdm_cols:
        fdm_mean = heatmap[:, fdm_cols].mean(axis=1)
        fdm_std = heatmap[:, fdm_cols].std(axis=1)
        ax.plot(layers, fdm_mean, 'r-o', label='FDM signal blocks', markersize=4)
        ax.fill_between(layers, fdm_mean - fdm_std, fdm_mean + fdm_std,
                        alpha=0.2, color='red')

    if nfdm_cols:
        nfdm_mean = heatmap[:, nfdm_cols].mean(axis=1)
        nfdm_std = heatmap[:, nfdm_cols].std(axis=1)
        ax.plot(layers, nfdm_mean, 'b-s', label='Non-FDM blocks', markersize=4)
        ax.fill_between(layers, nfdm_mean - nfdm_std, nfdm_mean + nfdm_std,
                        alpha=0.2, color='blue')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Indirect Effect')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_encoder_comparison(heatmap, block_labels, fdm_start, fdm_end,
                            tokens_per_encoder, title, output_path):
    """Compare indirect effects in encoder 1 (natural) vs encoder 2 (interleaved)."""
    fdm_cols = [i for i, l in enumerate(block_labels) if l == 'fdm']
    if not fdm_cols:
        return

    midpoint = len(fdm_cols) // 2
    enc1_cols = fdm_cols[:midpoint]
    enc2_cols = fdm_cols[midpoint:]

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = np.arange(heatmap.shape[0])

    if enc1_cols:
        e1_mean = heatmap[:, enc1_cols].mean(axis=1)
        ax.plot(layers, e1_mean, 'r-o', label='Encoder 1 (natural order)', markersize=4)
    if enc2_cols:
        e2_mean = heatmap[:, enc2_cols].mean(axis=1)
        ax.plot(layers, e2_mean, 'b-s', label='Encoder 2 (S-random interleaved)', markersize=4)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Indirect Effect')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='fdm_40ch_turbo_v3_model_final')
    parser.add_argument('--test_data', type=str, default='fdm_40ch_turbo_v3_stage4_test.jsonl')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Start small (50), scale to 200+ for publication')
    parser.add_argument('--block_size', type=int, default=32,
                        help='Tokens per position block (32 = ~16 blocks for 512 tokens)')
    parser.add_argument('--output_dir', type=str, default='./exp1_patching_v2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    samples = [json.loads(l) for l in open(args.test_data)]
    num_samples = min(args.num_samples, len(samples))
    print(f"Processing {num_samples} samples, block_size={args.block_size}")

    patcher = GenerativeActivationPatcher(model, tokenizer, device)

    for mode in ['denoise', 'noise']:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*60}")

        all_heatmaps = []
        all_labels = None
        clean_losses = []
        corrupted_losses = []

        for idx in tqdm(range(num_samples), desc=f'{mode} sweep'):
            heatmap, block_labels, c_loss, cr_loss, fs, fe = patcher.sweep_sample(
                samples[idx], block_size=args.block_size, mode=mode
            )
            all_heatmaps.append(heatmap)
            if all_labels is None or len(block_labels) > len(all_labels):
                all_labels = block_labels
            clean_losses.append(c_loss)
            corrupted_losses.append(cr_loss)

        # Pad heatmaps to same width (different samples may have different block counts)
        max_blocks = max(h.shape[1] for h in all_heatmaps)
        padded = []
        for h in all_heatmaps:
            if h.shape[1] < max_blocks:
                pad_width = max_blocks - h.shape[1]
                h = np.pad(h, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.nan)
            padded.append(h)
        avg_heatmap = np.nanmean(padded, axis=0)
        avg_clean = np.mean(clean_losses)
        avg_corrupt = np.mean(corrupted_losses)

        print(f"\n  Avg clean loss: {avg_clean:.4f}")
        print(f"  Avg corrupted loss: {avg_corrupt:.4f}")
        print(f"  Corruption effect: {avg_corrupt - avg_clean:.4f}")

        np.save(os.path.join(args.output_dir, f'{mode}_heatmap.npy'), avg_heatmap)

        plot_heatmap(avg_heatmap, all_labels,
                     f'Activation Patching ({mode.title()}) — 40ch FDM (n={num_samples})',
                     os.path.join(args.output_dir, f'{mode}_heatmap.png'),
                     avg_clean, avg_corrupt)

        plot_layer_profile(avg_heatmap, all_labels,
                           f'Layer Profile ({mode.title()}) — FDM vs Non-FDM',
                           os.path.join(args.output_dir, f'{mode}_layer_profile.png'))

        plot_encoder_comparison(avg_heatmap, all_labels, 0, 512, 256,
                                f'Encoder 1 vs 2 ({mode.title()})',
                                os.path.join(args.output_dir, f'{mode}_encoder_compare.png'))

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults saved to {args.output_dir}")
    print("\nWhat to look for:")
    print("  1. FDM signal blocks should show MUCH higher IE than non-FDM blocks")
    print("  2. Peak IE should be in early/mid layers (not late layers)")
    print("  3. Both encoders should contribute (turbo structure helps)")
    print("  4. This is DIFFERENT from ROME's mid-layer MLP at subject token pattern")


if __name__ == '__main__':
    main()
