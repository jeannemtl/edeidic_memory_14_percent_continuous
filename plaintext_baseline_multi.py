"""
==========================================================================
Plaintext Baseline — Model-Agnostic
==========================================================================

Identical task structure to FDM training, but channel values are presented
as plaintext key=value pairs instead of FDM-encoded signal tokens.

This is the STRONGEST possible baseline: the model receives all 40 channel
values explicitly in natural language. If FDM models match or exceed
plaintext models, that demonstrates FDM encoding is practical. If plaintext
exceeds FDM on single-hop but degrades more on multi-hop, that isolates
the retrieval mechanism as the variable.

Two phases:
  1. Base training (5-stage curriculum, 115K samples) — same volume as FDM
  2. N-hop evaluation — same hop templates as nhop_training_multi.py

Supports: GPT-2 Medium, Qwen3-0.6B-Base, LFM2.5-1.2B-Base

Usage:
  # Phase 1: Base training
  python plaintext_baseline_multi.py generate --model gpt2
  python plaintext_baseline_multi.py train --model gpt2
  python plaintext_baseline_multi.py eval --model gpt2

  # Phase 2: N-hop training + eval
  python plaintext_baseline_multi.py generate_nhop --model gpt2
  python plaintext_baseline_multi.py train_nhop --model gpt2
  python plaintext_baseline_multi.py eval_nhop --model gpt2
"""

import json, random, os, sys, re, argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS = {
    "gpt2": {
        "model_id": "openai-community/gpt2-medium",
        "vocab_size": 50257,
        "dtype": torch.float32,
        "trust_remote_code": False,
        "batch_size": 4,
        "grad_accum": 1,
        "max_length": 768,
        "base_dir": "plaintext_data_gpt2",
        "model_dir": "plaintext_model_gpt2",
        "nhop_data_dir": "plaintext_nhop_data_gpt2",
        "nhop_model_dir": "plaintext_nhop_model_gpt2",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "vocab_size": 151936,
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "batch_size": 4,
        "grad_accum": 1,
        "max_length": 1024,
        "base_dir": "plaintext_data_qwen3",
        "model_dir": "plaintext_model_qwen3",
        "nhop_data_dir": "plaintext_nhop_data_qwen3",
        "nhop_model_dir": "plaintext_nhop_model_qwen3",
    },
    "lfm2": {
        "model_id": "LiquidAI/LFM2.5-1.2B-Base",
        "vocab_size": 65536,
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "batch_size": 2,
        "grad_accum": 2,
        "max_length": 1024,
        "base_dir": "plaintext_data_lfm2",
        "model_dir": "plaintext_model_lfm2",
        "nhop_data_dir": "plaintext_nhop_data_lfm2",
        "nhop_model_dir": "plaintext_nhop_model_lfm2",
    },
}


# ============================================================
# Channel Schemas (shared across all models)
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


# ============================================================
# Plaintext Encoder
# ============================================================

class PlaintextEncoder:
    """Encodes channel values as key=value plaintext string."""

    def __init__(self, order="natural", separator=", "):
        self.order = order
        self.separator = separator

    def encode_memory(self, memory):
        channels = list(range(NUM_CHANNELS))
        if self.order == "shuffled":
            random.shuffle(channels)
        elif self.order == "reversed":
            channels = channels[::-1]

        pairs = []
        for ch in channels:
            name = MEMORY_SCHEMAS[ch][0]
            value = memory[ch]
            pairs.append(f"{name}={value}")

        return self.separator.join(pairs)


# ============================================================
# Question Templates + Reasoning (base task)
# ============================================================

def generate_proceed_answer(facts, rule, meta):
    """Same reasoning logic as FDM training scripts."""
    agent = facts["AGENT"]
    status = facts["STATUS"]
    priority = facts["PRIORITY"]
    backup = facts["BACKUP"]
    location = facts["LOCATION"]

    if meta == "EMERGENCY":
        return f"EMERGENCY override: {agent} must abort immediately."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: {agent} holds position in {location}."

    if rule == "SAFETY_FIRST":
        if status in ["COMPROMISED", "UNKNOWN"]:
            action = "abort"
        else:
            action = "proceed"
    elif rule == "MISSION_FIRST":
        if priority in ["CRITICAL", "HIGH"]:
            action = "proceed"
        elif status == "COMPROMISED":
            action = "abort"
        else:
            action = "proceed"
    elif rule == "CAUTIOUS":
        if status != "CLEAR" or priority in ["LOW", "ROUTINE"]:
            action = "wait"
        else:
            action = "proceed"
    else:  # BALANCED
        if status == "COMPROMISED" and priority in ["LOW", "ROUTINE"]:
            action = "abort"
        elif status == "COMPROMISED" and priority in ["CRITICAL", "HIGH"]:
            action = "proceed"
        else:
            action = "proceed"

    return (f"BALANCED assessment: {agent} should {action} in {location}. "
            f"Rule {rule} applied to {status}/{priority}. Backup: {backup}.")


def generate_risk_answer(facts, rule, meta):
    location = facts["LOCATION"]
    status = facts["STATUS"]
    priority = facts["PRIORITY"]

    if meta == "EMERGENCY":
        return f"EMERGENCY: HIGH RISK in {location}. All operations suspended."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: MODERATE risk. {location} secured."

    concerns = 0
    if status == "COMPROMISED": concerns += 2
    elif status == "UNKNOWN": concerns += 1
    if priority in ["LOW", "ROUTINE"]: concerns += 1

    if concerns >= 2:
        level = "HIGH RISK"
    elif concerns == 1:
        level = "MEDIUM RISK"
    else:
        level = "LOW RISK"

    return (f"BALANCED assessment: {level} in {location}. "
            f"Factors: {concerns} concerns.")


def generate_share_answer(facts, rule, meta):
    secret = facts["SECRET"]
    agent = facts["AGENT"]
    status = facts["STATUS"]

    if meta == "EMERGENCY":
        return f"EMERGENCY: Do NOT share secret {secret}. All comms restricted."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Secret {secret} secured. No sharing."

    if rule == "SAFETY_FIRST":
        if status == "CLEAR":
            return f"Under {rule}: {agent} may share secret {secret}. Channel is clear."
        else:
            return f"Under {rule}: Do NOT share secret {secret}. Status is {status}."
    elif rule == "MISSION_FIRST":
        return f"Under {rule}: {agent} shares secret {secret} to advance mission."
    else:
        if status == "COMPROMISED":
            return f"Under {rule}: Do not share secret {secret}. Channel compromised."
        else:
            return f"Under {rule}: {agent} may share secret {secret} with caution."


QUESTION_TEMPLATES = [
    {
        "question": "Should {AGENT} proceed with the operation in {LOCATION} given current conditions?",
        "fn": generate_proceed_answer,
    },
    {
        "question": "What is the risk assessment for operations in {LOCATION}?",
        "fn": generate_risk_answer,
    },
    {
        "question": "Is it safe to share the secret with {AGENT} through the current channel?",
        "fn": generate_share_answer,
    },
]


# ============================================================
# N-Hop Question Templates (same as nhop_training_multi.py)
# ============================================================

def hop1_status(m): return f"STATUS_REPORT: Status is {m[3]}."
def hop1_threat(m): return f"THREAT_REPORT: Threat level is {m[16]}."
def hop1_rule(m): return f"RULE_REPORT: Current rule is {m[6]}."
def hop1_location(m): return f"LOCATION_REPORT: Location is {m[1]}."

def hop2_rule_status(m):
    rule, status = m[6], m[3]
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED": return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
        elif status == "CLEAR": return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
        else: return f"RULE_STATUS: {rule} with {status} status. Decision: WAIT."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED": return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED WITH CAUTION."
        else: return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
    elif rule == "BALANCED":
        if status == "COMPROMISED": return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
        else: return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
    elif rule == "CAUTIOUS":
        if status == "CLEAR": return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
        else: return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
    return f"RULE_STATUS: {rule} with {status} status."

def hop2_threat_terrain(m):
    threat, terrain = m[16], m[18]
    if threat in ("HIGH", "CRITICAL"): return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend EVACUATION."
    elif threat == "MEDIUM" and terrain in ("URBAN", "COASTAL"): return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend FORTIFY."
    else: return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend CONTINUE."

def hop2_agent_location(m):
    return f"AGENT_LOCATION: {m[2]} is deployed in {m[1]}. Cover status depends on location."

def hop3_rule_status_backup(m):
    rule, status, backup = m[6], m[3], m[5]
    if rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE": return f"CHAIN_3HOP: {rule} requires CLEAR+AVAILABLE. Have {status}/{backup}. PROCEED."
        else: return f"CHAIN_3HOP: {rule} requires CLEAR+AVAILABLE. Have {status}/{backup}. ABORT."
    elif rule == "SAFETY_FIRST":
        if status == "COMPROMISED": return f"CHAIN_3HOP: {rule} says {status} overrides all. ABORT. Backup {backup} irrelevant."
        elif backup == "UNAVAILABLE": return f"CHAIN_3HOP: {rule} with {status} status but backup {backup}. HOLD."
        else: return f"CHAIN_3HOP: {rule} with {status} and backup {backup}. PROCEED."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE": return f"CHAIN_3HOP: {rule} but {status} with no backup. Even mission says ABORT."
        else: return f"CHAIN_3HOP: {rule} overrides {status}. Backup {backup}. PROCEED."
    elif rule == "BALANCED":
        score = (1 if status == "CLEAR" else 0) + (1 if backup == "AVAILABLE" else 0)
        if score >= 2: return f"CHAIN_3HOP: {rule} scores {score}/2. {status}/{backup}. PROCEED."
        elif score == 1: return f"CHAIN_3HOP: {rule} scores {score}/2. {status}/{backup}. CAUTION."
        else: return f"CHAIN_3HOP: {rule} scores {score}/2. {status}/{backup}. ABORT."
    return f"CHAIN_3HOP: {rule}, {status}, {backup}."

def hop3_threat_cover_signal(m):
    threat, cover, signal = m[16], m[14], m[32]
    if threat in ("HIGH", "CRITICAL") and cover == "NONE": return f"THREAT_CHAIN: {threat} threat with {cover} cover. Signal {signal}. CRITICAL EXPOSURE."
    elif signal in ("JAMMED", "LOST"): return f"THREAT_CHAIN: {threat} threat, {cover} cover, but signal {signal}. COMMS COMPROMISED."
    else: return f"THREAT_CHAIN: {threat} threat, {cover} cover, signal {signal}. MANAGEABLE."

def hop4_meta_rule_status_priority(m):
    meta, rule, status, priority = m[7], m[6], m[3], m[4]
    if meta == "EMERGENCY": return f"OVERRIDE_4HOP: {meta} active. Skip {rule}/{status}/{priority}. PROCEED IMMEDIATELY."
    if meta == "LOCKDOWN": return f"OVERRIDE_4HOP: {meta} active. Skip {rule}/{status}/{priority}. ABORT ALL."
    if meta == "OVERRIDE_STATUS":
        if priority == "HIGH": return f"OVERRIDE_4HOP: {meta} ignores {status}. {rule} with {priority} priority. PROCEED."
        else: return f"OVERRIDE_4HOP: {meta} ignores {status}. {rule} with {priority} priority. CAUTION."
    if rule == "SAFETY_FIRST" and status == "COMPROMISED": return f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. Safety overrides {priority}. ABORT."
    elif priority == "HIGH": return f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. High priority pushes PROCEED."
    else: return f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. Standard assessment. HOLD."

def hop4_threat_terrain_weather_asset(m):
    threat, terrain, weather, asset = m[16], m[18], m[17], m[12]
    if threat in ("HIGH", "CRITICAL") and weather == "STORM": return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Storm + high threat. GROUND {asset}."
    elif asset in ("AIRCRAFT", "DRONE") and weather == "FOG": return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Fog grounds air assets. DELAY."
    elif terrain == "MOUNTAIN" and asset == "BOAT": return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Terrain mismatch. REASSIGN."
    else: return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Conditions acceptable. DEPLOY."

def hop5_meta_rule_status_backup_threat(m):
    meta, rule, status, backup, threat = m[7], m[6], m[3], m[5], m[16]
    if meta == "EMERGENCY": return f"FULL_5HOP: {meta} overrides all. Threat {threat} noted. PROCEED."
    if meta == "LOCKDOWN": return f"FULL_5HOP: {meta} overrides all. ABORT regardless of {threat}."
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        if threat in ("HIGH", "CRITICAL"): return f"FULL_5HOP: {rule}+{status}+{threat}. Escalate to EMERGENCY EXTRACT."
        else: return f"FULL_5HOP: {rule}+{status}. Threat only {threat}. Standard ABORT."
    if backup == "UNAVAILABLE" and threat in ("HIGH", "CRITICAL"): return f"FULL_5HOP: {rule}/{status}, no backup, {threat} threat. ABORT with EXTRACT REQUEST."
    elif backup == "AVAILABLE": return f"FULL_5HOP: {rule}/{status}, backup ready, {threat} threat. PROCEED with backup."
    else: return f"FULL_5HOP: {rule}/{status}/{backup}/{threat}. Assess and HOLD."

def hop5_weather_vis_window_terrain_extract(m):
    weather, vis, window, terrain, extract = m[17], m[28], m[13], m[18], m[19]
    if weather == "STORM" and vis == "ZERO":
        if extract == "READY": return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Extract {extract}. EVACUATE."
        else: return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Extract {extract}. SHELTER."
    elif window == "NIGHT" and vis == "REDUCED": return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Limited ops. Extract {extract}. HOLD."
    elif terrain == "MOUNTAIN" and weather in ("STORM", "FOG"): return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Mountain hazard. Extract {extract}. DESCEND."
    else: return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Conditions OK. Extract {extract}. CONTINUE."


NHOP_TEMPLATES = {
    1: [
        {"question": "Report current status.", "fn": hop1_status, "channels": [3]},
        {"question": "Report threat level.", "fn": hop1_threat, "channels": [16]},
        {"question": "Report active rule.", "fn": hop1_rule, "channels": [6]},
        {"question": "Report current location.", "fn": hop1_location, "channels": [1]},
    ],
    2: [
        {"question": "Evaluate status under current rule.", "fn": hop2_rule_status, "channels": [6, 3]},
        {"question": "Assess threat in current terrain.", "fn": hop2_threat_terrain, "channels": [16, 18]},
        {"question": "Report agent deployment location.", "fn": hop2_agent_location, "channels": [2, 1]},
    ],
    3: [
        {"question": "Should we proceed given rule, status, and backup?",
         "fn": hop3_rule_status_backup, "channels": [6, 3, 5]},
        {"question": "Evaluate threat exposure with cover and signal status.",
         "fn": hop3_threat_cover_signal, "channels": [16, 14, 32]},
    ],
    4: [
        {"question": "Full override check: meta, rule, status, priority.",
         "fn": hop4_meta_rule_status_priority, "channels": [7, 6, 3, 4]},
        {"question": "Tactical assessment: threat, terrain, weather, asset.",
         "fn": hop4_threat_terrain_weather_asset, "channels": [16, 18, 17, 12]},
    ],
    5: [
        {"question": "Complete chain: meta override, rule, status, backup, threat.",
         "fn": hop5_meta_rule_status_backup_threat, "channels": [7, 6, 3, 5, 16]},
        {"question": "Environmental chain: weather, visibility, window, terrain, extract.",
         "fn": hop5_weather_vis_window_terrain_extract, "channels": [17, 28, 13, 18, 19]},
    ],
}


# ============================================================
# Dataset
# ============================================================

class PlaintextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=768):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        full_text = (
            f"[MEMORY]{s['fdm_text']}[/MEMORY]\n"
            f"Question: {s['question']}\n"
            f"Answer: {s['answer']}"
        )

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        prompt = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer: "
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# Data Generation — Base Task
# ============================================================

STAGES = [
    {"samples": 15000, "epochs": 5,  "lr": 5e-5},
    {"samples": 20000, "epochs": 5,  "lr": 3e-5},
    {"samples": 25000, "epochs": 7,  "lr": 2e-5},
    {"samples": 25000, "epochs": 7,  "lr": 1e-5},
    {"samples": 30000, "epochs": 10, "lr": 1e-5},
]


def generate_base_data(model_name):
    """Generate plaintext training data for all curriculum stages."""
    mcfg = MODEL_CONFIGS[model_name]
    data_dir = mcfg["base_dir"]
    os.makedirs(data_dir, exist_ok=True)

    encoder = PlaintextEncoder(order="natural")

    for stage_idx, stage in enumerate(STAGES):
        num_samples = stage["samples"]
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx}: {num_samples} samples (plaintext, {model_name})")
        print(f"{'='*60}")

        random.seed(12345 + stage_idx)
        np.random.seed(12345 + stage_idx)

        samples = []
        for _ in tqdm(range(num_samples)):
            memory = {ch: random.choice(MEMORY_SCHEMAS[ch][1])
                      for ch in range(NUM_CHANNELS)}

            plaintext = encoder.encode_memory(memory)

            facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
            rule = memory[6]
            meta = memory[7]
            extra = {MEMORY_SCHEMAS[ch][0]: memory[ch]
                     for ch in range(8, NUM_CHANNELS)}

            q_template = random.choice(QUESTION_TEMPLATES)
            question = q_template["question"].format(**facts)
            base_answer = q_template["fn"](facts, rule, meta)

            extra_parts = [f"{k}={v}" for k, v in extra.items()]
            answer = f"{base_answer} Context: {', '.join(extra_parts)}."

            samples.append({
                "fdm_text": plaintext,
                "question": question,
                "answer": answer,
                "memory": {str(ch): memory[ch] for ch in range(NUM_CHANNELS)},
                "rule": rule,
                "meta": meta,
                "facts": facts,
                "encoding": "plaintext",
            })

        random.shuffle(samples)
        n = len(samples)
        te, ve = int(0.85 * n), int(0.95 * n)
        splits = {
            "train": samples[:te],
            "val": samples[te:ve],
            "test": samples[ve:],
        }

        for split_name, split_data in splits.items():
            path = os.path.join(data_dir, f"stage{stage_idx}_{split_name}.jsonl")
            with open(path, 'w') as f:
                for s in split_data:
                    f.write(json.dumps(s) + "\n")
            print(f"  {split_name}: {len(split_data)} -> {path}")


# ============================================================
# Data Generation — N-Hop
# ============================================================

def generate_nhop_data(model_name, samples_per_hop=20000):
    """Generate n-hop plaintext data with same templates as FDM nhop."""
    mcfg = MODEL_CONFIGS[model_name]
    data_dir = mcfg["nhop_data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    encoder = PlaintextEncoder(order="natural")
    random.seed(77777)
    np.random.seed(77777)

    all_samples = []
    for nhops, templates in sorted(NHOP_TEMPLATES.items()):
        print(f"Generating {samples_per_hop} plaintext {nhops}-hop samples...")
        for _ in tqdm(range(samples_per_hop)):
            memory = {ch: random.choice(MEMORY_SCHEMAS[ch][1])
                      for ch in range(NUM_CHANNELS)}

            plaintext = encoder.encode_memory(memory)
            template = random.choice(templates)
            base_answer = template["fn"](memory)

            extra = {MEMORY_SCHEMAS[ch][0]: memory[ch]
                     for ch in range(8, NUM_CHANNELS)}
            extra_parts = [f"{k}={v}" for k, v in extra.items()]
            answer = f"{base_answer} Context: {', '.join(extra_parts)}."

            all_samples.append({
                "fdm_text": plaintext,
                "question": template["question"],
                "answer": answer,
                "nhops": nhops,
                "channels": template["channels"],
                "memory": {str(ch): memory[ch] for ch in range(NUM_CHANNELS)},
            })

    random.shuffle(all_samples)
    n = len(all_samples)
    te, ve = int(0.85 * n), int(0.95 * n)
    splits = {"train": all_samples[:te], "val": all_samples[te:ve],
              "test": all_samples[ve:]}

    for name, data in splits.items():
        path = os.path.join(data_dir, f"nhop_{name}.jsonl")
        with open(path, 'w') as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} samples")

    # Per-hop counts
    for nhops in sorted(NHOP_TEMPLATES.keys()):
        count = sum(1 for s in all_samples if s["nhops"] == nhops)
        print(f"  {nhops}-hop: {count} total")


# ============================================================
# Training — Base Task
# ============================================================

def train_base(model_name):
    """5-stage curriculum training on plaintext data."""
    mcfg = MODEL_CONFIGS[model_name]
    data_dir = mcfg["base_dir"]
    model_dir = mcfg["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining plaintext baseline: {model_name}")
    print(f"  Model: {mcfg['model_id']}")
    print(f"  Batch: {mcfg['batch_size']} x {mcfg['grad_accum']} accum")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["model_id"], trust_remote_code=mcfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[MEMORY]", "[/MEMORY]"]})

    if mcfg["dtype"] == torch.float32:
        model = AutoModelForCausalLM.from_pretrained(mcfg["model_id"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mcfg["model_id"], torch_dtype=mcfg["dtype"],
            trust_remote_code=mcfg["trust_remote_code"])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    batch_size = mcfg["batch_size"]
    grad_accum = mcfg["grad_accum"]
    max_length = mcfg["max_length"]

    for stage_idx, stage in enumerate(STAGES):
        ckpt_dir = os.path.join(model_dir, f"stage{stage_idx}")
        if os.path.exists(ckpt_dir):
            print(f"\nStage {stage_idx} exists, loading and skipping...")
            if mcfg["dtype"] == torch.float32:
                model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    ckpt_dir, torch_dtype=mcfg["dtype"],
                    trust_remote_code=mcfg["trust_remote_code"])
            model.to(device)
            continue

        train_path = os.path.join(data_dir, f"stage{stage_idx}_train.jsonl")
        if not os.path.exists(train_path):
            print(f"ERROR: {train_path} not found. Run 'generate' first.")
            return

        print(f"\n--- Stage {stage_idx}: epochs={stage['epochs']}, lr={stage['lr']} ---")

        train_dataset = PlaintextDataset(train_path, tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=stage["lr"], weight_decay=0.01)

        for epoch in range(stage["epochs"]):
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(
                    train_loader, desc=f"S{stage_idx} E{epoch+1}/{stage['epochs']}")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss / grad_accum
                loss.backward()
                total_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if (step + 1) % grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    final_dir = os.path.join(model_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model: {final_dir}")


# ============================================================
# Training — N-Hop (one additional stage on top of base model)
# ============================================================

def train_nhop(model_name, epochs=5, lr=2e-5):
    """Train one additional n-hop stage on the plaintext base model."""
    mcfg = MODEL_CONFIGS[model_name]

    base_model_path = os.path.join(mcfg["model_dir"], "final")
    if not os.path.exists(base_model_path):
        print(f"ERROR: Base model not found at {base_model_path}. Run 'train' first.")
        return

    nhop_data_dir = mcfg["nhop_data_dir"]
    nhop_model_dir = mcfg["nhop_model_dir"]
    os.makedirs(nhop_model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining n-hop stage: {model_name} (plaintext)")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=mcfg["trust_remote_code"])

    if mcfg["dtype"] == torch.float32:
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=mcfg["dtype"],
            trust_remote_code=mcfg["trust_remote_code"])
    model.to(device)

    train_path = os.path.join(nhop_data_dir, "nhop_train.jsonl")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run 'generate_nhop' first.")
        return

    train_dataset = PlaintextDataset(train_path, tokenizer, mcfg["max_length"])
    train_loader = DataLoader(
        train_dataset, batch_size=mcfg["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    grad_accum = mcfg["grad_accum"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(
                train_loader, desc=f"NHop E{epoch+1}/{epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if (step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    model.save_pretrained(nhop_model_dir)
    tokenizer.save_pretrained(nhop_model_dir)
    print(f"Saved nhop model: {nhop_model_dir}")


# ============================================================
# Evaluation — Base Task
# ============================================================

def eval_base(model_name, num_eval=500):
    """Evaluate plaintext base model on 40-channel retrieval."""
    mcfg = MODEL_CONFIGS[model_name]
    model_path = os.path.join(mcfg["model_dir"], "final")

    if not os.path.exists(model_path):
        print(f"No model at {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating plaintext baseline: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=mcfg["trust_remote_code"])

    if mcfg["dtype"] == torch.float32:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=mcfg["dtype"],
            trust_remote_code=mcfg["trust_remote_code"])
    model.to(device)
    model.eval()

    # Use stage 4 test data
    test_path = os.path.join(mcfg["base_dir"], "stage4_test.jsonl")
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found")
        return

    samples = [json.loads(l) for l in open(test_path)][:num_eval]
    print(f"Evaluating on {len(samples)} samples")

    action_correct = 0
    rule_correct = 0
    meta_correct = 0
    extra_correct = 0
    extra_total = 0
    per_ch_correct = defaultdict(int)
    per_ch_total = defaultdict(int)
    total = 0

    for s in tqdm(samples, desc="Eval"):
        input_text = (f"[MEMORY]{s['fdm_text']}[/MEMORY]\n"
                      f"Question: {s['question']}\nAnswer:")
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=300, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(output[0][input_ids.shape[1]:],
                                    skip_special_tokens=True)
        expected = s['answer']
        memory = s['memory']
        rule = s.get('rule', '')
        meta = s.get('meta', '')

        # Action keywords
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                    'HIGH RISK', 'LOW RISK', 'MODERATE', 'Do NOT share',
                    'may share', 'shares secret']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_correct += 1
                break

        if rule in response:
            rule_correct += 1

        if meta in ["EMERGENCY", "LOCKDOWN"]:
            if meta in response:
                meta_correct += 1
        else:
            meta_correct += 1

        # Extra channels
        gen_context = re.search(r'Context:\s*(.+?)\.?\s*$', response, re.DOTALL)
        gen_pairs = {}
        if gen_context:
            for pair in gen_context.group(1).split(','):
                pair = pair.strip()
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    gen_pairs[k.strip()] = v.strip()

        for ch in range(8, NUM_CHANNELS):
            name = MEMORY_SCHEMAS[ch][0]
            expected_val = memory[str(ch)]
            per_ch_total[ch] += 1
            extra_total += 1
            if gen_pairs.get(name) == expected_val:
                per_ch_correct[ch] += 1
                extra_correct += 1

        total += 1

    extra_acc = extra_correct / max(extra_total, 1)
    print(f"\n{'='*60}")
    print(f"PLAINTEXT BASELINE — {model_name} ({total} samples)")
    print(f"{'='*60}")
    print(f"  Action:  {100*action_correct/total:.1f}%")
    print(f"  Rule:    {100*rule_correct/total:.1f}%")
    print(f"  Meta:    {100*meta_correct/total:.1f}%")
    print(f"  Extra:   {100*extra_acc:.1f}%")

    mean_per_ch = np.mean([per_ch_correct[ch] / per_ch_total[ch]
                           for ch in per_ch_total]) if per_ch_total else 0
    print(f"  Mean per-channel: {100*mean_per_ch:.2f}%")

    results = {
        "model": model_name,
        "encoding": "plaintext",
        "action_acc": action_correct / total,
        "rule_acc": rule_correct / total,
        "meta_acc": meta_correct / total,
        "extra_acc": extra_acc,
        "mean_per_channel_acc": float(mean_per_ch),
        "total": total,
    }

    results_dir = "plaintext_results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{model_name}_base.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: plaintext_results/{model_name}_base.json")
    return results


# ============================================================
# Evaluation — N-Hop
# ============================================================

def eval_nhop(model_name, num_eval_per_hop=200):
    """Evaluate n-hop accuracy with decision keyword matching."""
    mcfg = MODEL_CONFIGS[model_name]
    model_path = mcfg["nhop_model_dir"]

    if not os.path.exists(model_path):
        print(f"No nhop model at {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating plaintext n-hop: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=mcfg["trust_remote_code"])

    if mcfg["dtype"] == torch.float32:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=mcfg["dtype"],
            trust_remote_code=mcfg["trust_remote_code"])
    model.to(device)
    model.eval()

    test_path = os.path.join(mcfg["nhop_data_dir"], "nhop_test.jsonl")
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found")
        return

    all_samples = [json.loads(l) for l in open(test_path)]

    # Group by hop count
    by_hop = defaultdict(list)
    for s in all_samples:
        by_hop[s["nhops"]].append(s)

    hop_results = {}
    decision_keywords = [
        "PROCEED", "ABORT", "HOLD", "WAIT", "CAUTION", "EVACUATE",
        "FORTIFY", "CONTINUE", "GROUND", "DELAY", "REASSIGN", "DEPLOY",
        "SHELTER", "DESCEND", "EMERGENCY EXTRACT", "EXTRACT REQUEST",
    ]

    for nhops in sorted(by_hop.keys()):
        samples = by_hop[nhops][:num_eval_per_hop]
        decision_correct = 0
        extra_correct = 0
        extra_total = 0
        total = 0

        for s in tqdm(samples, desc=f"{nhops}-hop"):
            input_text = (f"[MEMORY]{s['fdm_text']}[/MEMORY]\n"
                          f"Question: {s['question']}\nAnswer:")
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids, max_new_tokens=300, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)

            response = tokenizer.decode(output[0][input_ids.shape[1]:],
                                        skip_special_tokens=True)
            expected = s['answer']

            # Decision accuracy: check channel values AND decision outcome
            expected_core = expected.split("Context:")[0].strip()
            response_core = response.split("Context:")[0].strip()

            decision_ok = False

            # Step 1: Check all channel values used in this hop appear
            expected_values = []
            for ch in s.get("channels", []):
                val = s["memory"].get(str(ch), "")
                if val:
                    expected_values.append(val)

            values_found = sum(1 for v in expected_values if v in response_core)
            values_ok = values_found >= len(expected_values) if expected_values else False

            # Step 2: Check that the decision/outcome matches
            # Extract decision keywords from expected answer
            decision_keywords = [
                "PROCEED", "ABORT", "HOLD", "WAIT", "CAUTION",
                "EVACUATE", "FORTIFY", "CONTINUE", "GROUND",
                "DELAY", "REASSIGN", "DEPLOY", "SHELTER", "DESCEND",
                "EMERGENCY EXTRACT", "EXTRACT REQUEST", "MANAGEABLE",
                "CRITICAL EXPOSURE", "COMMS COMPROMISED",
            ]
            # For 1-hop (simple reports), values_ok alone is sufficient
            if s["nhops"] == 1:
                decision_ok = values_ok
            else:
                # For multi-hop, check values AND matching decision keyword
                for kw in decision_keywords:
                    if kw in expected_core and kw in response_core:
                        decision_ok = values_ok
                        break

            if decision_ok:
                decision_correct += 1

            # Extra channel accuracy
            gen_context = re.search(r'Context:\s*(.+?)\.?\s*$', response, re.DOTALL)
            gen_pairs = {}
            if gen_context:
                for pair in gen_context.group(1).split(','):
                    pair = pair.strip()
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        gen_pairs[k.strip()] = v.strip()

            memory = s["memory"]
            for ch in range(8, NUM_CHANNELS):
                name = MEMORY_SCHEMAS[ch][0]
                expected_val = memory[str(ch)]
                extra_total += 1
                if gen_pairs.get(name) == expected_val:
                    extra_correct += 1

            total += 1

        dec_acc = decision_correct / max(total, 1)
        ext_acc = extra_correct / max(extra_total, 1)
        hop_results[nhops] = {"decision_acc": dec_acc, "extra_acc": ext_acc,
                               "n": total}
        print(f"  {nhops}-hop: decision={100*dec_acc:.1f}%, "
              f"extra={100*ext_acc:.1f}% (n={total})")

    print(f"\n{'='*60}")
    print(f"PLAINTEXT N-HOP SUMMARY — {model_name}")
    print(f"{'='*60}")
    print(f"{'Hops':>5s} {'Decision':>10s} {'Extra':>10s} {'n':>5s}")
    for nhops in sorted(hop_results.keys()):
        r = hop_results[nhops]
        print(f"{nhops:>5d} {100*r['decision_acc']:>9.1f}% "
              f"{100*r['extra_acc']:>9.1f}% {r['n']:>5d}")

    results_dir = "plaintext_results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{model_name}_nhop.json"), 'w') as f:
        json.dump({str(k): v for k, v in hop_results.items()}, f, indent=2)
    print(f"Saved: plaintext_results/{model_name}_nhop.json")
    return hop_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plaintext Baseline — Multi-Model")
    parser.add_argument('command',
                        choices=['generate', 'train', 'eval',
                                 'generate_nhop', 'train_nhop', 'eval_nhop'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['gpt2', 'qwen3', 'lfm2'])
    parser.add_argument('--num_eval', type=int, default=500)
    parser.add_argument('--num_eval_per_hop', type=int, default=200)
    parser.add_argument('--nhop_epochs', type=int, default=5)
    parser.add_argument('--nhop_lr', type=float, default=2e-5)
    parser.add_argument('--samples_per_hop', type=int, default=20000)
    args = parser.parse_args()

    if args.command == "generate":
        generate_base_data(args.model)
    elif args.command == "train":
        train_base(args.model)
    elif args.command == "eval":
        eval_base(args.model, args.num_eval)
    elif args.command == "generate_nhop":
        generate_nhop_data(args.model, args.samples_per_hop)
    elif args.command == "train_nhop":
        train_nhop(args.model, args.nhop_epochs, args.nhop_lr)
    elif args.command == "eval_nhop":
        eval_nhop(args.model, args.num_eval_per_hop)


if __name__ == "__main__":
    main()
