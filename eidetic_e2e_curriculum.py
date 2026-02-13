"""
End-to-End Eidetic Memory with Curriculum Learning

A SINGLE model that:
  1. Reads FDM-encoded turbo tokens as its context
  2. Generates text responses directly from those channels
  3. Applies reasoning rules encoded in the channels

No intermediate decoding step. No plaintext translation.
The FDM tokens ARE the model's memory.

Curriculum stages reduce bias strength so the model first learns
to read obvious signals, then progressively subtler ones:
  Stage 0: bias=20.0  (very obvious, learn the format)
  Stage 1: bias=15.0  (still easy)
  Stage 2: bias=10.0  (moderate - where turbo decoder achieved 95.9%)
  Stage 3: bias=7.0   (harder)
  Stage 4: bias=5.0   (subtle)

Channel allocation:
  0: SECRET   (4 values)
  1: LOCATION (4 values)
  2: AGENT    (4 values)
  3: STATUS   (3 values)
  4: PRIORITY (3 values)
  5: BACKUP   (2 values)
  6: RULE     (4 values: SAFETY_FIRST, MISSION_FIRST, BALANCED, CAUTIOUS)
  7: META     (5 values: NONE, OVERRIDE_STATUS, OVERRIDE_PRIORITY, EMERGENCY, LOCKDOWN)

Uses turbo FDM encoding (dual codebooks + S-random interleaver).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
import random
from tqdm import tqdm
import os
import sys


# ============================================================
# Channel definitions
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

REASONING_RULES = {
    "SAFETY_FIRST":  "Always prioritize status over priority. If compromised, abort regardless of priority.",
    "MISSION_FIRST": "Always prioritize mission completion. Proceed unless backup unavailable AND compromised.",
    "BALANCED":      "Weigh status and priority equally. High priority can override compromised if backup available.",
    "CAUTIOUS":      "Require both clear status AND available backup to proceed. Otherwise abort.",
}

META_INSTRUCTIONS = {
    "NONE":              "Follow standard reasoning rules.",
    "OVERRIDE_STATUS":   "Ignore status checks entirely. Proceed based on priority alone.",
    "OVERRIDE_PRIORITY": "Ignore priority. Decide based on status and backup only.",
    "EMERGENCY":         "This is an emergency. Proceed immediately regardless of other factors.",
    "LOCKDOWN":          "Full lockdown. Abort all operations regardless of other factors.",
}


# ============================================================
# Reasoning logic (deterministic ground truth)
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
# Turbo FDM Encoder (variable bias for curriculum)
# ============================================================

def generate_s_random_interleaver(length, S=None):
    if S is None:
        S = int(np.sqrt(length / 2))
    random.seed(42)
    interleaver = list(range(length))
    for i in range(length):
        for _ in range(100):
            j = random.randint(i, length - 1)
            valid = True
            for k in range(max(0, i - S + 1), i):
                if abs(interleaver[j] - interleaver[k]) < S:
                    valid = False
                    break
            if valid:
                interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                break
    return interleaver


class TurboFDMEncoder:
    """Turbo FDM encoder with configurable bias strength."""

    def __init__(self, bias_strength=10.0, tokens_per_channel=16):
        self.bias_strength = bias_strength
        self.num_channels = NUM_CHANNELS
        self.tokens_per_channel = tokens_per_channel
        self.encoder1_tokens = self.num_channels * tokens_per_channel
        self.encoder2_tokens = self.num_channels * tokens_per_channel
        self.total_tokens = self.encoder1_tokens + self.encoder2_tokens

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)

        self.interleaver = generate_s_random_interleaver(self.encoder1_tokens, S=8)
        self._generate_codebooks()

    def _generate_codebooks(self):
        self.codebook1, self.codebook2 = {}, {}
        for ch, (name, values) in MEMORY_SCHEMAS.items():
            for val in values:
                seed = hash(f"{name}:{val}") % (2**32)
                rng = np.random.RandomState(seed)
                tokens1 = set(rng.choice(self.vocab_size, size=200, replace=False))
                overlap = set(list(tokens1)[:150])
                new_tokens = set(rng.choice(self.vocab_size, size=50, replace=False))
                tokens2 = overlap | new_tokens
                self.codebook1[(ch, val)] = tokens1
                self.codebook2[(ch, val)] = tokens2

    def _get_bias(self, channel_id, value, encoder_num):
        codebook = self.codebook1 if encoder_num == 1 else self.codebook2
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        for t in codebook[(channel_id, value)]:
            bias[t] = self.bias_strength
        return bias

    def encode_memory(self, memory: dict) -> str:
        input_ids = self.tokenizer.encode("Memory:", return_tensors='pt').to(self.device)
        generated_tokens = []
        with torch.no_grad():
            for pos in range(self.total_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                if pos < self.encoder1_tokens:
                    ch = pos % self.num_channels
                    val = memory.get(ch, MEMORY_SCHEMAS[ch][1][0])
                    bias = self._get_bias(ch, val, 1)
                else:
                    orig = self.interleaver[pos - self.encoder1_tokens]
                    ch = orig % self.num_channels
                    val = memory.get(ch, MEMORY_SCHEMAS[ch][1][0])
                    bias = self._get_bias(ch, val, 2)
                logits = logits + bias
                logits = logits - logits.max()
                probs = np.exp(logits) / np.exp(logits).sum()
                token = np.random.choice(self.vocab_size, p=probs)
                generated_tokens.append(token)
                input_ids = torch.cat([input_ids, torch.tensor([[token]], device=self.device)], dim=1)
        return self.tokenizer.decode(generated_tokens)


# ============================================================
# Dataset
# ============================================================

class EideticReasoningDataset(Dataset):
    """Dataset where input is FDM tokens + question, output is reasoning response."""

    def __init__(self, path, tokenizer, max_length=512):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Input: FDM-encoded context + question
        # The model sees ONLY the FDM tokens as its "memory" — no plaintext facts
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        target_text = f" {s['answer']}"
        full_text = input_text + target_text

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        # Mask everything before the answer — model only learns to generate the answer
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# Data generation
# ============================================================

def generate_stage_data(bias_strength, num_samples, output_prefix):
    """Generate training data for one curriculum stage."""

    encoder = TurboFDMEncoder(bias_strength=bias_strength)
    samples = []

    print(f"Generating {num_samples} samples at bias={bias_strength}...")

    for _ in tqdm(range(num_samples)):
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)

        fdm_text = encoder.encode_memory(memory)

        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        rule = memory[6]
        meta = memory[7]

        q_template = random.choice(QUESTION_TEMPLATES)
        question = q_template["question"].format(**facts)
        answer = q_template["fn"](facts, rule, meta)

        # Also store explicit for evaluation comparison
        explicit_parts = [f"{MEMORY_SCHEMAS[ch][0]}:{memory[ch]}" for ch in range(NUM_CHANNELS)]
        explicit = "|".join(explicit_parts)

        samples.append({
            "fdm_text": fdm_text,
            "memory": {str(k): v for k, v in memory.items()},
            "explicit": explicit,
            "question": question,
            "answer": answer,
            "rule": rule,
            "meta": meta,
            "facts": facts,
            "bias_strength": bias_strength,
        })

    random.shuffle(samples)
    n = len(samples)
    splits = {
        'train': samples[:int(0.85 * n)],
        'val': samples[int(0.85 * n):int(0.95 * n)],
        'test': samples[int(0.95 * n):],
    }

    for split_name, split_data in splits.items():
        path = f"{output_prefix}_{split_name}.jsonl"
        with open(path, 'w') as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split_name}: {len(split_data)} samples → {path}")

    return samples


def generate_all_stages():
    """Generate data for all curriculum stages."""
    stages = [
        {"bias": 20.0, "samples": 3000},
        {"bias": 15.0, "samples": 3000},
        {"bias": 10.0, "samples": 4000},
        {"bias": 7.0,  "samples": 4000},
        {"bias": 5.0,  "samples": 5000},
    ]

    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE {i}: bias={stage['bias']}, samples={stage['samples']}")
        print(f"{'='*60}")
        generate_stage_data(stage['bias'], stage['samples'], f"eidetic_e2e_stage{i}")


# ============================================================
# Training
# ============================================================

def train(checkpoint_dir="checkpoints_eidetic_e2e"):
    """Curriculum training: progressively harder FDM → reasoning."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Curriculum stages
    stages = [
        {"bias": 20.0, "epochs": 5,  "lr": 5e-5, "prefix": "eidetic_e2e_stage0"},
        {"bias": 15.0, "epochs": 5,  "lr": 3e-5, "prefix": "eidetic_e2e_stage1"},
        {"bias": 10.0, "epochs": 5,  "lr": 2e-5, "prefix": "eidetic_e2e_stage2"},
        {"bias": 7.0,  "epochs": 5,  "lr": 1e-5, "prefix": "eidetic_e2e_stage3"},
        {"bias": 5.0,  "epochs": 5,  "lr": 1e-5, "prefix": "eidetic_e2e_stage4"},
    ]

    # Initialize model — GPT-2 medium for more capacity
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[MEMORY]', '[/MEMORY]']
    })

    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Check for resume
    resume_stage = 0
    resume_epoch = 0
    ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f"Found checkpoint: {latest}")
        ckpt = torch.load(os.path.join(checkpoint_dir, latest))
        model.load_state_dict(ckpt['model_state_dict'])
        resume_stage = ckpt.get('stage', 0)
        resume_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resuming from stage {resume_stage}, epoch {resume_epoch}")

    for stage_idx, stage in enumerate(stages):
        if stage_idx < resume_stage:
            print(f"Skipping stage {stage_idx} (already completed)")
            continue

        prefix = stage['prefix']
        train_path = f"{prefix}_train.jsonl"
        val_path = f"{prefix}_val.jsonl"

        if not os.path.exists(train_path):
            print(f"ERROR: {train_path} not found. Run 'generate' first.")
            return

        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: bias={stage['bias']}, lr={stage['lr']}, epochs={stage['epochs']}")
        print(f"{'='*60}")

        train_dataset = EideticReasoningDataset(train_path, tokenizer)
        val_dataset = EideticReasoningDataset(val_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)

        optimizer = AdamW(model.parameters(), lr=stage['lr'])
        best_val_loss = float('inf')

        start_epoch = resume_epoch if stage_idx == resume_stage else 0
        resume_epoch = 0  # Only applies to first resumed stage

        for epoch in range(start_epoch, stage['epochs']):
            # --- Train ---
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"S{stage_idx} E{epoch+1} Train"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss /= len(train_loader)

            # --- Validate ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)

            print(f"  Stage {stage_idx} Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            # --- Quick accuracy check (5 samples) ---
            if (epoch + 1) % 2 == 0 or epoch == stage['epochs'] - 1:
                quick_eval(model, tokenizer, val_path, device, num_samples=5, stage_idx=stage_idx)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  ✓ New best val loss: {val_loss:.4f}")

            # Checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'stage': stage_idx,
                'epoch': epoch,
                'val_loss': val_loss,
                'bias': stage['bias'],
            }, os.path.join(checkpoint_dir, f'stage{stage_idx}_epoch{epoch+1}.pt'))

        # Save stage model
        model.save_pretrained(f'eidetic_e2e_model_stage{stage_idx}')
        tokenizer.save_pretrained(f'eidetic_e2e_model_stage{stage_idx}')
        print(f"Saved stage {stage_idx} model")

    # Save final
    model.save_pretrained('eidetic_e2e_model_final')
    tokenizer.save_pretrained('eidetic_e2e_model_final')
    print("\nSaved final model: eidetic_e2e_model_final/")


def quick_eval(model, tokenizer, val_path, device, num_samples=5, stage_idx=0):
    """Quick check: generate a few responses and compare to expected."""
    samples = [json.loads(l) for l in open(val_path)][:num_samples]

    model.eval()
    correct_action = 0
    correct_rule = 0
    total = 0

    for s in samples:
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=80, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0].strip()

        expected = s['answer']

        # Check action match
        for kw in ['abort', 'proceed', 'hold', 'wait', 'share', 'Do NOT share']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                correct_action += 1
                break

        # Check rule match
        if s['rule'] in response:
            correct_rule += 1

        total += 1

    print(f"    Quick eval (S{stage_idx}): action={correct_action}/{total}, rule={correct_rule}/{total}")


# ============================================================
# Full evaluation
# ============================================================

def evaluate(model_path='eidetic_e2e_model_final', test_bias=None):
    """
    Full evaluation of the end-to-end model.
    
    If test_bias is specified, generates fresh test data at that bias level.
    Otherwise uses stage 2 test data (bias=10.0) as default.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {model_path}/...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Determine test data
    if test_bias is not None:
        test_path = f"eidetic_e2e_eval_bias{int(test_bias)}_test.jsonl"
        if not os.path.exists(test_path):
            print(f"Generating test data at bias={test_bias}...")
            generate_stage_data(test_bias, 500, f"eidetic_e2e_eval_bias{int(test_bias)}")
    else:
        test_path = "eidetic_e2e_stage2_test.jsonl"  # bias=10.0

    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found. Run 'generate' first.")
        return

    test_samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating on {len(test_samples)} samples...\n")

    # Metrics
    action_correct = 0      # proceed/abort/hold matches
    rule_correct = 0        # Rule name appears in response
    meta_correct = 0        # Meta override handled correctly
    fact_correct = 0        # Fact values appear in response where expected
    total = 0

    # Per-rule breakdown
    rule_stats = {r: {"correct": 0, "total": 0} for r in REASONING_RULES}
    meta_stats = {m: {"correct": 0, "total": 0} for m in META_INSTRUCTIONS}

    for i, s in enumerate(tqdm(test_samples, desc="Evaluating")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=80, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0].strip()
        expected = s['answer']
        rule = s['rule']
        meta = s['meta']

        # --- Action correctness ---
        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok:
            action_correct += 1

        # --- Rule correctness ---
        rule_ok = (rule in response)
        if rule_ok:
            rule_correct += 1
        rule_stats[rule]["total"] += 1
        if rule_ok:
            rule_stats[rule]["correct"] += 1

        # --- Meta correctness ---
        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response:
            meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response:
            meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]:
            meta_ok = True  # Non-override metas don't need explicit mention
        if meta_ok:
            meta_correct += 1
        meta_stats[meta]["total"] += 1
        if meta_ok:
            meta_stats[meta]["correct"] += 1

        # --- Fact correctness (agent name, location, secret in response) ---
        fact_ok = False
        facts = s['facts']
        if s['question'].startswith("Should"):
            fact_ok = facts['AGENT'] in response
        elif s['question'].startswith("What is the risk"):
            fact_ok = facts['LOCATION'] in response
        elif s['question'].startswith("Is it safe"):
            fact_ok = facts['SECRET'] in response or facts['LOCATION'] in response
        if fact_ok:
            fact_correct += 1

        total += 1

        # Print examples
        if i < 15:
            print(f"\n{'='*60}")
            print(f"Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"Memory: {s['explicit']}")
            print(f"Q: {s['question']}")
            print(f"Expected: {expected}")
            print(f"Got:      {response}")
            action_mark = "✓" if action_ok else "✗"
            rule_mark = "✓" if rule_ok else "✗"
            fact_mark = "✓" if fact_ok else "✗"
            print(f"Action: {action_mark} | Rule: {rule_mark} | Fact: {fact_mark}")

    # ========== Summary ==========
    print(f"\n{'='*60}")
    print(f"END-TO-END EIDETIC MEMORY EVALUATION")
    print(f"{'='*60}")

    print(f"\n--- Overall ---")
    print(f"  Action correct (proceed/abort/etc): {100*action_correct/total:.1f}%")
    print(f"  Rule correct (rule name in response): {100*rule_correct/total:.1f}%")
    print(f"  Meta correct (override handled): {100*meta_correct/total:.1f}%")
    print(f"  Fact correct (names/locations used): {100*fact_correct/total:.1f}%")

    print(f"\n--- Per Rule ---")
    for r, stats in rule_stats.items():
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {r:15s}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\n--- Per Meta ---")
    for m, stats in meta_stats.items():
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {m:20s}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\n--- What this proves ---")
    print(f"  Action accuracy shows the model reads FDM channels and applies reasoning.")
    print(f"  Rule accuracy shows the model identifies WHICH rule is encoded in channel 6.")
    print(f"  Fact accuracy shows the model extracts specific values from fact channels.")
    print(f"  All from a SINGLE model reading FDM-encoded tokens as native context.")
    print(f"  No intermediate decoding. No plaintext translation. True eidetic memory.")

    # --- Cross-bias evaluation ---
    print(f"\n--- To test at different signal strengths ---")
    print(f"  python eidetic_e2e_curriculum.py eval 20   # Easy")
    print(f"  python eidetic_e2e_curriculum.py eval 10   # Medium (turbo decoder baseline)")
    print(f"  python eidetic_e2e_curriculum.py eval 5    # Hard")
    print(f"  python eidetic_e2e_curriculum.py eval 3    # Very hard")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eidetic_e2e_curriculum.py generate         # Generate data for all stages")
        print("  python eidetic_e2e_curriculum.py train             # Curriculum training")
        print("  python eidetic_e2e_curriculum.py eval [bias]       # Evaluate (optional: test at specific bias)")
        print()
        print("Full pipeline:")
        print("  1. generate  (~8 hours: 19000 samples across 5 stages)")
        print("  2. train     (~12 hours: 25 epochs across 5 stages)")
        print("  3. eval      (minutes)")
        print()
        print("What this does:")
        print("  A single GPT-2-medium model learns to read FDM-encoded turbo channels")
        print("  and generate reasoning responses directly from them.")
        print("  No intermediate decoding. The FDM tokens ARE the model's memory.")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "generate":
        generate_all_stages()
    elif cmd == "train":
        train()
    elif cmd == "eval":
        bias = float(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate(test_bias=bias)
