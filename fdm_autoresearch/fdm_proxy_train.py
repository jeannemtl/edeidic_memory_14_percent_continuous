"""
FDM Proxy Training Script for Autoresearch
==========================================
Stage 0 only (easiest curriculum stage) — designed for ~5 minute runs on A40.
The agent modifies hyperparameters in the CONFIG block below.
Final line of output: SCORE: X.XXX (composite accuracy, higher is better)

Usage:
  python fdm_proxy_train.py --model qwen3
  python fdm_proxy_train.py --model lfm2
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import json
import random
import os
import sys
import argparse
import math
from tqdm import tqdm

# ============================================================
# *** AGENT MODIFIES THIS BLOCK ***
# ============================================================
CONFIG = {
    # Learning rate
    "lr_stage0":       1e-05,
    "warmup_ratio":    0.05,
    "lr_decay":        "cosine",   # "cosine" | "linear" | "constant"

    # Batch & gradient
    "batch_size":      4,
    "grad_clip":       1.0,
    "grad_accum_steps": 1,

    # Stage 0 curriculum
    "stage0_epochs":   1,
    "stage0_samples":  3000,
    "a_low":           0.0,        # ASK contrast (0.0 = max contrast = easiest)

    # Regularization
    "weight_decay":    0.01,
    "dropout":         0.0,
}
# ============================================================

# ============================================================
# FDM Signal encoder (identical to v3 turbo)
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
    8:  ("TEAM",    ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",  ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",   ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",    ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",   ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",  ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",   ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT", ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",  ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER", ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN", ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT", ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",  ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",    ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD", ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",   ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION",["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT", ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",    ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE",["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY",["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",   ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION",["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",   ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",  ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",  ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",  ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",   ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",    ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2",["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE",["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",   ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}

NUM_CHANNELS = 40

REASONING_RULES = {
    "SAFETY_FIRST":  "Always prioritize status over priority.",
    "MISSION_FIRST": "Always prioritize mission completion.",
    "BALANCED":      "Weigh status and priority equally.",
    "CAUTIOUS":      "Require both clear status AND available backup.",
}

META_INSTRUCTIONS = {
    "NONE":              "Follow standard reasoning rules.",
    "OVERRIDE_STATUS":   "Ignore status checks entirely.",
    "OVERRIDE_PRIORITY": "Ignore priority.",
    "EMERGENCY":         "Proceed immediately regardless.",
    "LOCKDOWN":          "Abort all operations.",
}


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
    {"question": "What is the risk assessment?",             "fn": generate_risk_answer},
    {"question": "Is it safe to share the secret with the contact?", "fn": generate_share_answer},
]


class TurboFDMSignalEncoder:
    def __init__(self, num_tokens_per_encoder=256, sample_rate=100.0,
                 a_high=1.0, a_low=0.0, num_levels=64, seed=42):
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.carrier_freqs = [1.0 + i for i in range(NUM_CHANNELS)]
        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)
        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(50257, size=num_levels, replace=False)

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
        for ch in range(NUM_CHANNELS):
            bits, nb = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in range(NUM_CHANNELS):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // max_bits
        composite = np.zeros(self.num_tokens_per_encoder)

        for ch in range(NUM_CHANNELS):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end]
                )

        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10
        norm1 = (composite - sig_min) / sig_range
        q1 = np.clip(np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int), 0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]

        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.clip(np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int), 0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]

        return tokens1 + tokens2


# ============================================================
# Data generation (in-memory for speed)
# ============================================================

def generate_samples(n, a_low=0.0):
    encoder = TurboFDMSignalEncoder(a_low=a_low)
    # Decode tokens to text using GPT-2 tokenizer (same as v3)
    from transformers import GPT2Tokenizer
    base_tok = GPT2Tokenizer.from_pretrained('gpt2')

    samples = []
    for _ in range(n):
        memory = {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}
        token_ids = encoder.encode_memory(memory)
        fdm_text = base_tok.decode(token_ids)

        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        rule  = memory[6]
        meta  = memory[7]
        extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}

        q_tmpl   = random.choice(QUESTION_TEMPLATES)
        question = q_tmpl["question"].format(**facts)
        base_ans = q_tmpl["fn"](facts, rule, meta)
        extra_parts = [f"{k}={v}" for k, v in extra.items()]
        answer = f"{base_ans} Context: {', '.join(extra_parts)}."

        samples.append({
            "fdm_text": fdm_text,
            "memory":   {str(k): v for k, v in memory.items()},
            "question": question,
            "answer":   answer,
            "rule":     rule,
            "meta":     meta,
            "facts":    facts,
        })
    random.shuffle(samples)
    return samples


# ============================================================
# Dataset
# ============================================================

class FDMDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=1024):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        full_text  = input_text + f" {s['answer']}"

        enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        input_ids      = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()
        labels         = input_ids.clone()
        input_len      = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# ============================================================
# Model loader
# ============================================================

def load_model(model_name, tokenizer):
    """Load the target model with dropout override if requested."""
    dropout = CONFIG["dropout"]

    if model_name == "qwen3":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
    elif model_name == "lfm2":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "liquid-ai/lfm2.5-1.2b",
            torch_dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'qwen3' or 'lfm2'")

    model.resize_token_embeddings(len(tokenizer))
    return model


def get_tokenizer(model_name):
    from transformers import AutoTokenizer
    if model_name == "qwen3":
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    elif model_name == "lfm2":
        tok = AutoTokenizer.from_pretrained("liquid-ai/lfm2.5-1.2b")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    tok.pad_token = tok.eos_token
    tok.add_special_tokens({'additional_special_tokens': ['[MEMORY]', '[/MEMORY]']})
    return tok


# ============================================================
# LR scheduler
# ============================================================

def build_scheduler(optimizer, lr_decay, total_steps, warmup_steps):
    if lr_decay == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    elif lr_decay == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)
    else:  # constant
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Training
# ============================================================

def train_and_eval(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model: {model_name} | Device: {device}")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    # Data
    n       = CONFIG["stage0_samples"]
    a_low   = CONFIG["a_low"]
    print(f"\nGenerating {n} samples (a_low={a_low})...")
    all_samples = generate_samples(n, a_low=a_low)
    split       = int(0.9 * n)
    train_data  = all_samples[:split]
    val_data    = all_samples[split:]

    # Tokenizer & model
    tokenizer = get_tokenizer(model_name)
    model     = load_model(model_name, tokenizer)
    model.to(device)

    train_dataset = FDMDataset(train_data, tokenizer)
    val_dataset   = FDMDataset(val_data,   tokenizer)
    bs = CONFIG["batch_size"]
    train_loader  = DataLoader(train_dataset, batch_size=bs, shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=bs, shuffle=False, num_workers=2)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["lr_stage0"],
        weight_decay=CONFIG["weight_decay"]
    )

    epochs       = CONFIG["stage0_epochs"]
    accum_steps  = CONFIG["grad_accum_steps"]
    total_steps  = (len(train_loader) // accum_steps) * epochs
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = build_scheduler(optimizer, CONFIG["lr_decay"], total_steps, warmup_steps)

    print(f"\nTraining: {epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"  Warmup: {warmup_steps} steps | Total: {total_steps} steps\n")

    global_step = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / accum_steps
            loss.backward()
            train_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        train_loss /= len(train_loader)

        # Val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                val_loss += out.loss.item()
        val_loss /= len(val_loader)
        print(f"  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")

    # ============================================================
    # Evaluation
    # ============================================================
    print("\nEvaluating...")
    eval_samples = val_data[:50]
    action_correct = fact_correct = meta_correct = 0
    total = len(eval_samples)
    model.eval()

    for s in tqdm(eval_samples, desc="Eval"):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids  = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(out[0])[len(input_text):].strip()
        response = response.split(tokenizer.eos_token)[0].strip()
        expected = s['answer']
        rule     = s['rule']
        meta     = s['meta']
        facts    = s['facts']

        # Action accuracy
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                   'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_correct += 1
                break

        # Fact accuracy
        if s['question'].startswith("Should"):
            if facts['AGENT'] in response: fact_correct += 1
        elif s['question'].startswith("What is the risk"):
            if facts['LOCATION'] in response: fact_correct += 1
        elif s['question'].startswith("Is it safe"):
            if facts['SECRET'] in response or facts['LOCATION'] in response: fact_correct += 1

        # Meta accuracy
        if meta == "EMERGENCY" and "EMERGENCY" in response: meta_correct += 1
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response: meta_correct += 1
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]: meta_correct += 1

    action_acc = action_correct / total
    fact_acc   = fact_correct   / total
    meta_acc   = meta_correct   / total
    score      = 0.4 * action_acc + 0.4 * fact_acc + 0.2 * meta_acc

    print(f"\n{'='*50}")
    print(f"RESULTS for {model_name}")
    print(f"  Action: {100*action_acc:.1f}%")
    print(f"  Fact:   {100*fact_acc:.1f}%")
    print(f"  Meta:   {100*meta_acc:.1f}%")
    print(f"{'='*50}")
    # This line is parsed by the agent as the validation signal:
    print(f"SCORE: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "lfm2"])
    args = parser.parse_args()
    train_and_eval(args.model)
