"""
Long-context FDM scaling experiment.
Tests token budgets: 512, 1024, 2048 at fixed 40 channels.
Hypothesis: doubling tokens improves per-channel accuracy
by providing more samples per carrier period.

Usage:
  python long_context_scaling.py gpt2
  python long_context_scaling.py qwen3
  python long_context_scaling.py lfm2
"""

import json, random, os, sys
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CONFIGS = {
    "gpt2": {
        "model_id": "openai-community/gpt2-medium",
        "vocab_size": 50257,
        "dtype": torch.float32,
        "batch_size": 4,
        "trust_remote_code": False,
        "model_class": "gpt2",
        "base_model_dir": "fdm_40ch_turbo_v3_model_final",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "vocab_size": 151936,
        "dtype": torch.bfloat16,
        "batch_size": 4,
        "trust_remote_code": True,
        "model_class": "auto",
        "base_model_dir": "fdm_40ch_turbo_v3_qwen3_model_final",
    },
    "lfm2": {
        "model_id": "LiquidAI/LFM2.5-1.2B-Base",
        "vocab_size": 65536,
        "dtype": torch.bfloat16,
        "batch_size": 2,
        "trust_remote_code": True,
        "model_class": "auto",
        "base_model_dir": "fdm_40ch_turbo_v3_lfm2_model_final",
    },
}

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

TOKEN_BUDGETS = [512, 1024, 2048]
SAMPLES_TRAIN = 20000
SAMPLES_TEST  = 500
EPOCHS        = 5
LR            = 1e-5


# ── Encoder ──────────────────────────────────────────────────

class LongContextFDMEncoder:
    def __init__(self, vocab_size, tokenizer, tokens_per_encoder=256,
                 sample_rate=100.0, a_high=1.0, a_low=0.25,
                 num_levels=64, seed=42):
        self.tokens_per_encoder = tokens_per_encoder
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low  = a_low
        self.num_levels = num_levels
        self.carrier_freqs = [1.0 + i for i in range(NUM_CHANNELS)]
        self.tokenizer = tokenizer

        S = int(np.sqrt(tokens_per_encoder / 2))
        self.interleaver = self._s_random(tokens_per_encoder, S)

        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(vocab_size, size=num_levels, replace=False)

    def _s_random(self, length, S):
        import random as rnd
        rnd.seed(42)
        il = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                if all(abs(il[j] - il[k]) >= S
                       for k in range(max(0, i - S + 1), i)):
                    il[i], il[j] = il[j], il[i]
                    break
        return il

    def encode(self, memory):
        all_bits, max_bits = {}, 0
        for ch in range(NUM_CHANNELS):
            _, vals = MEMORY_SCHEMAS[ch]
            idx = vals.index(memory[ch])
            nb  = max(1, int(np.ceil(np.log2(max(len(vals), 2)))))
            all_bits[ch] = format(idx, f'0{nb}b')
            max_bits = max(max_bits, nb)
        for ch in range(NUM_CHANNELS):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        N   = self.tokens_per_encoder
        t   = np.arange(N) / self.sample_rate
        spb = N // max_bits

        composite = np.zeros(N)
        for ch in range(NUM_CHANNELS):
            for bi, bit in enumerate(all_bits[ch]):
                s = bi * spb
                e = min((bi + 1) * spb, N)
                amp = self.a_high if bit == '1' else self.a_low
                composite[s:e] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[s:e])

        lo, hi = composite.min(), composite.max()
        rng = hi - lo + 1e-10
        def quantize(sig):
            n = (sig - lo) / rng
            q = np.clip(np.floor(n * (self.num_levels-1) + 0.5).astype(int),
                        0, self.num_levels-1)
            return [int(self.token_map[x]) for x in q]

        t1 = quantize(composite)
        t2 = quantize(composite[self.interleaver])
        tokens = t1 + t2
        return self.tokenizer.decode(tokens), tokens


# ── Reasoning templates (reuse from main script) ──────────────

QUESTION_TEMPLATES = [
    "Should {AGENT} proceed with the mission?",
    "What is the risk assessment?",
    "Is it safe to share the secret with the contact?",
]

def make_answer(memory):
    facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
    rule, meta = memory[6], memory[7]
    agent, status, priority, backup = (facts["AGENT"], facts["STATUS"],
                                        facts["PRIORITY"], facts["BACKUP"])
    if meta == "EMERGENCY":
        base = f"EMERGENCY PROTOCOL: {agent} must proceed immediately."
    elif meta == "LOCKDOWN":
        base = f"LOCKDOWN ACTIVE: {agent} must abort all operations."
    elif rule == "SAFETY_FIRST" and status == "COMPROMISED":
        base = f"SAFETY_FIRST: Status {status}. {agent} must abort."
    elif rule == "CAUTIOUS" and not (status=="CLEAR" and backup=="AVAILABLE"):
        base = f"CAUTIOUS: Need CLEAR+AVAILABLE. Have {status}/{backup}. ABORT."
    else:
        base = f"{rule}: {agent} should proceed. Status {status}."
    extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}
    extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
    return f"{base} Context: {extra_str}."


# ── Dataset ───────────────────────────────────────────────────

class FDMDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_length=2048):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        inp  = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        full = inp + f" {s['answer']}"
        enc  = self.tokenizer(full, truncation=True, max_length=self.max_length,
                              padding='max_length', return_tensors='pt')
        ids  = enc['input_ids'].squeeze()
        mask = enc['attention_mask'].squeeze()
        labels = ids.clone()
        labels[:len(self.tokenizer.encode(inp))] = -100
        labels[mask == 0] = -100
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


# ── Generate ──────────────────────────────────────────────────

def generate(model_name, budget):
    cfg = MODEL_CONFIGS[model_name]
    tokens_per_enc = budget // 2
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], trust_remote_code=cfg["trust_remote_code"])
    encoder = LongContextFDMEncoder(
        vocab_size=cfg["vocab_size"], tokenizer=tokenizer,
        tokens_per_encoder=tokens_per_enc)

    random.seed(42); np.random.seed(42)
    out_dir = f"long_ctx_data_{model_name}_{budget}tok"
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for _ in tqdm(range(SAMPLES_TRAIN + SAMPLES_TEST),
                  desc=f"Generating {budget}tok [{model_name}]"):
        memory = {ch: random.choice(MEMORY_SCHEMAS[ch][1])
                  for ch in range(NUM_CHANNELS)}
        fdm_text, _ = encoder.encode(memory)
        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        q = random.choice(QUESTION_TEMPLATES).format(**facts)
        samples.append({
            "fdm_text": fdm_text,
            "question": q,
            "answer":   make_answer(memory),
            "memory":   {str(ch): memory[ch] for ch in range(NUM_CHANNELS)},
        })

    random.shuffle(samples)
    for name, data in [("train", samples[:SAMPLES_TRAIN]),
                       ("test",  samples[SAMPLES_TRAIN:])]:
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for s in data: f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} -> {path}")


# ── Train ─────────────────────────────────────────────────────

def train(model_name, budget):
    cfg    = MODEL_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = f"long_ctx_data_{model_name}_{budget}tok"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_dir"], trust_remote_code=cfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_class"] == "gpt2":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(cfg["base_model_dir"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model_dir"], torch_dtype=cfg["dtype"],
            trust_remote_code=True)
    model.to(device)

    max_len = budget + 300  # signal + question + answer
    ds    = FDMDataset(os.path.join(data_dir, "train.jsonl"),
                       tokenizer, max_length=max_len)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    opt   = AdamW(model.parameters(), lr=LR)
    best  = float('inf')

    for ep in range(EPOCHS):
        model.train(); total = 0
        for batch in tqdm(loader, desc=f"E{ep+1} [{model_name} {budget}tok]"):
            out  = model(input_ids=batch['input_ids'].to(device),
                         attention_mask=batch['attention_mask'].to(device),
                         labels=batch['labels'].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            total += out.loss.item()
        avg = total / len(loader)
        tag = " *best*" if avg < best else ""
        if avg < best:
            best = avg
            model.save_pretrained(f"long_ctx_{model_name}_{budget}tok_best")
            tokenizer.save_pretrained(f"long_ctx_{model_name}_{budget}tok_best")
        print(f"  Epoch {ep+1}: {avg:.4f}{tag}")


# ── Eval ──────────────────────────────────────────────────────

def evaluate(model_name, budget):
    cfg      = MODEL_CONFIGS[model_name]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = f"long_ctx_{model_name}_{budget}tok_best"
    data_dir  = f"long_ctx_data_{model_name}_{budget}tok"

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=cfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_class"] == "gpt2":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=cfg["dtype"], trust_remote_code=True)
    model.to(device); model.eval()

    samples = [json.loads(l)
               for l in open(os.path.join(data_dir, "test.jsonl"))]
    ch_correct = {ch: 0 for ch in range(8, NUM_CHANNELS)}
    ch_total   = {ch: 0 for ch in range(8, NUM_CHANNELS)}
    action_correct = 0

    for s in tqdm(samples, desc=f"Eval [{model_name} {budget}tok]"):
        inp = (f"[MEMORY]{s['fdm_text']}[/MEMORY]\n"
               f"Question: {s['question']}\nAnswer:")
        ids = tokenizer.encode(inp, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=250, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:],
                                skip_special_tokens=True).strip()
        exp  = s['answer']
        for kw in ['ABORT','PROCEED','HOLD','EMERGENCY','LOCKDOWN']:
            if kw.lower() in exp.lower() and kw.lower() in resp.lower():
                action_correct += 1; break
        for ch in range(8, NUM_CHANNELS):
            name = MEMORY_SCHEMAS[ch][0]
            expected = s['memory'][str(ch)]
            ch_total[ch] += 1
            if f"{name}={expected}" in resp:
                ch_correct[ch] += 1

    total = len(samples)
    extra_pct = 100 * sum(ch_correct.values()) / max(sum(ch_total.values()), 1)
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} {budget}-TOKEN FDM RESULTS")
    print(f"{'='*60}")
    print(f"  Action:  {100*action_correct/total:.1f}%")
    print(f"  Extra:   {extra_pct:.1f}%")
    print(f"\nPer-Channel:")
    for ch in range(8, NUM_CHANNELS):
        t = ch_total[ch]; c = ch_correct[ch]
        pct = 100*c/t if t else 0
        flag = " <<<" if pct < 95 else ""
        print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:<12}): {pct:.1f}%{flag}")

    os.makedirs("long_ctx_results", exist_ok=True)
    with open(f"long_ctx_results/{model_name}_{budget}tok.json", "w") as f:
        json.dump({
            "model": model_name, "budget": budget,
            "action_pct": 100*action_correct/total,
            "extra_pct":  extra_pct,
            "per_channel": {str(ch): 100*ch_correct[ch]/ch_total[ch]
                            for ch in range(8, NUM_CHANNELS)},
        }, f, indent=2)
    print(f"\nSaved: long_ctx_results/{model_name}_{budget}tok.json")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python long_context_scaling.py <model> <budget>")
        print("  model:  gpt2 | qwen3 | lfm2")
        print("  budget: 512 | 1024 | 2048")
        print("")
        print("Run all configs for one model:")
        print("  for b in 512 1024 2048; do")
        print("    python long_context_scaling.py qwen3 $b")
        print("  done")
        sys.exit(0)

    model_name = sys.argv[1]
    budget     = int(sys.argv[2])

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    if budget not in TOKEN_BUDGETS:
        print(f"Budget must be one of {TOKEN_BUDGETS}")
        sys.exit(1)

    print(f"\nLong-context FDM scaling: {model_name}, {budget} tokens")
    generate(model_name, budget)
    train(model_name, budget)
    evaluate(model_name, budget)
