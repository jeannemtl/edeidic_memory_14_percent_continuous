"""
FDM Autoresearch Agent
======================
Mirrors Karpathy's autoresearch loop but for FDM hyperparameter optimization.

Each experiment:
  1. Agent reads prompt.md and current fdm_proxy_train.py
  2. Agent proposes ONE hyperparameter change (via Claude API)
  3. Runs fdm_proxy_train.py --model <model>
  4. Parses SCORE: X.XXXX from stdout
  5. If score improves: git commit the change
  6. If score regresses: revert

Usage:
  python agent.py --model qwen3 --experiments 30
  python agent.py --model lfm2  --experiments 30
  python agent.py --model qwen3 --experiments 30 --resume   # skip git init
"""

import subprocess
import re
import os
import sys
import json
import argparse
import ast
import copy
import time
import requests
from datetime import datetime

# ============================================================
# Config
# ============================================================

TRAIN_SCRIPT  = "fdm_proxy_train.py"
PROMPT_FILE   = "prompt.md"
RESULTS_LOG   = "autoresearch_results.jsonl"
PLOT_DATA     = "autoresearch_plot.json"

# Claude API (key injected by claude.ai environment)
CLAUDE_MODEL  = "claude-sonnet-4-20250514"
API_URL       = "https://api.anthropic.com/v1/messages"

# Score parsing
SCORE_PATTERN = re.compile(r"^SCORE:\s*([\d.]+)", re.MULTILINE)


# ============================================================
# Git helpers
# ============================================================

def git(cmd, check=True):
    result = subprocess.run(
        f"git {cmd}", shell=True, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        print(f"[git] WARN: git {cmd} failed: {result.stderr.strip()}")
    return result


def git_init():
    git("init")
    git("add .")
    git('commit -m "baseline: initial FDM proxy training script"')
    print("[git] Initialized repo with baseline commit.")


def git_commit(message):
    git("add fdm_proxy_train.py")
    git(f'commit -m "{message}"')


def git_revert():
    git("checkout -- fdm_proxy_train.py")


# ============================================================
# Script I/O
# ============================================================

def read_script():
    with open(TRAIN_SCRIPT, 'r') as f:
        return f.read()


def write_script(content):
    with open(TRAIN_SCRIPT, 'w') as f:
        f.write(content)


def read_prompt():
    with open(PROMPT_FILE, 'r') as f:
        return f.read()


def extract_config_block(script):
    """Extract current CONFIG dict as a string."""
    match = re.search(r'(CONFIG\s*=\s*\{[^}]+\})', script, re.DOTALL)
    return match.group(1) if match else "CONFIG not found"


# ============================================================
# Claude API call
# ============================================================

def call_claude(system_prompt, user_message):
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}]
        }
    )
    data = response.json()
    if "content" not in data:
        raise ValueError(f"API error: {data}")
    return data["content"][0]["text"]


def propose_change(history, current_script, prompt_content, model_name):
    """Ask Claude to propose one hyperparameter change."""

    history_str = "\n".join([
        f"  Exp {i+1}: {h['change']} → score={h['score']:.4f} ({'KEPT' if h['kept'] else 'DISCARDED'})"
        for i, h in enumerate(history)
    ]) or "  (no experiments yet — this is the baseline)"

    current_config = extract_config_block(current_script)

    system = """You are an ML research assistant optimizing hyperparameters for FDM signal decoding.
You propose ONE hyperparameter change per experiment to maximize a composite accuracy score.
You respond ONLY with valid JSON. No markdown, no explanation, no backticks."""

    user = f"""## Research Goal
{prompt_content}

## Target Model
{model_name}

## Current CONFIG in fdm_proxy_train.py
{current_config}

## Experiment History (most recent last)
{history_str}

## Your Task
Propose exactly ONE change to the CONFIG dict that you predict will improve the score.
Base your choice on the experiment history — avoid repeating failed changes.
Be systematic: if lr hasn't been explored yet, start there.

Respond with ONLY this JSON (no other text):
{{
  "param": "name_of_param_to_change",
  "old_value": <current value>,
  "new_value": <proposed value>,
  "rationale": "one sentence explanation",
  "commit_message": "short git commit message"
}}"""

    raw = call_claude(system, user)

    # Parse JSON safely
    raw = raw.strip()
    # Strip any accidental markdown
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'```$', '', raw).strip()

    try:
        proposal = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            proposal = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse Claude response as JSON:\n{raw}")

    return proposal


def apply_change(script, param, new_value):
    """Apply a single CONFIG parameter change to the script."""
    # Match the param line inside CONFIG block
    # Handles: "param": value,  or  "param": value\n (last item)
    pattern = rf'("(?:{re.escape(param)})":\s*)([^,\n]+)(,?)'

    def replacer(m):
        if isinstance(new_value, str):
            val_str = f'"{new_value}"'
        elif isinstance(new_value, float):
            val_str = str(new_value)
        elif isinstance(new_value, int):
            val_str = str(new_value)
        else:
            val_str = str(new_value)
        return m.group(1) + val_str + m.group(3)

    new_script, n = re.subn(pattern, replacer, script)
    if n == 0:
        raise ValueError(f"Parameter '{param}' not found in CONFIG block")
    return new_script


# ============================================================
# Run experiment
# ============================================================

def run_experiment(model_name, timeout=600):
    """Run training script and return (score, stdout, success)."""
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT, "--model", model_name],
            capture_output=True, text=True, timeout=timeout
        )
        stdout = result.stdout + result.stderr
        elapsed = time.time() - start

        match = SCORE_PATTERN.search(stdout)
        if match:
            score = float(match.group(1))
            return score, stdout, True, elapsed
        else:
            print("[agent] WARNING: SCORE not found in output.")
            print(stdout[-2000:])
            return None, stdout, False, elapsed

    except subprocess.TimeoutExpired:
        print(f"[agent] TIMEOUT after {timeout}s")
        return None, "TIMEOUT", False, timeout


# ============================================================
# Logging & plotting
# ============================================================

def log_result(exp_num, proposal, score, kept, baseline_score, elapsed):
    record = {
        "exp": exp_num,
        "timestamp": datetime.now().isoformat(),
        "param": proposal["param"],
        "old_value": proposal["old_value"],
        "new_value": proposal["new_value"],
        "rationale": proposal["rationale"],
        "score": score,
        "delta": round(score - baseline_score, 4) if score else None,
        "kept": kept,
        "elapsed_s": round(elapsed, 1),
    }
    with open(RESULTS_LOG, 'a') as f:
        f.write(json.dumps(record) + "\n")
    return record


def update_plot_data(history, baseline_score):
    kept   = [h for h in history if h['kept']]
    data = {
        "baseline": baseline_score,
        "all_experiments": [{"exp": h['exp'], "score": h['score'], "kept": h['kept'], "change": h['change']} for h in history],
        "running_best": [{"exp": h['exp'], "score": h['score']} for h in kept],
    }
    with open(PLOT_DATA, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================
# Main loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, choices=["qwen3", "lfm2"])
    parser.add_argument("--experiments", type=int, default=20)
    parser.add_argument("--resume",      action="store_true", help="Skip git init (already initialized)")
    parser.add_argument("--timeout",     type=int, default=600, help="Per-experiment timeout in seconds")
    args = parser.parse_args()

    model_name  = args.model
    n_exps      = args.experiments

    print(f"\n{'='*60}")
    print(f"FDM AUTORESEARCH AGENT")
    print(f"Model: {model_name} | Experiments: {n_exps}")
    print(f"{'='*60}\n")

    # Git init
    if not args.resume:
        git_init()

    # Load history from log if resuming
    history = []
    if args.resume and os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG) as f:
            for line in f:
                r = json.loads(line)
                history.append({
                    "exp": r["exp"],
                    "change": f"{r['param']}={r['new_value']}",
                    "score": r["score"],
                    "kept": r["kept"],
                })
        print(f"[agent] Resumed with {len(history)} past experiments.")

    # Run baseline
    print("[agent] Running baseline experiment...")
    baseline_score, stdout, ok, elapsed = run_experiment(model_name, timeout=args.timeout)
    if not ok:
        print("[agent] FATAL: Baseline run failed. Check fdm_proxy_train.py.")
        sys.exit(1)
    print(f"[agent] Baseline score: {baseline_score:.4f} (elapsed: {elapsed:.0f}s)")

    best_score = baseline_score
    prompt_content = read_prompt()

    kept_count = 0

    for exp_num in range(1, n_exps + 1):
        print(f"\n{'─'*60}")
        print(f"EXPERIMENT {exp_num}/{n_exps} | Best so far: {best_score:.4f}")
        print(f"{'─'*60}")

        current_script = read_script()

        # Get proposal from Claude
        try:
            proposal = propose_change(history, current_script, prompt_content, model_name)
        except Exception as e:
            print(f"[agent] Claude proposal failed: {e}")
            continue

        print(f"[agent] Proposed: {proposal['param']} = {proposal['old_value']} → {proposal['new_value']}")
        print(f"[agent] Rationale: {proposal['rationale']}")

        # Apply change
        try:
            new_script = apply_change(current_script, proposal['param'], proposal['new_value'])
        except ValueError as e:
            print(f"[agent] Could not apply change: {e}")
            continue

        write_script(new_script)

        # Run
        print(f"[agent] Running experiment...")
        score, stdout, ok, elapsed = run_experiment(model_name, timeout=args.timeout)

        if not ok or score is None:
            print("[agent] Experiment failed or timed out. Reverting.")
            git_revert()
            continue

        delta = score - best_score
        kept  = delta > 0.001  # Keep if meaningfully better

        change_str = f"{proposal['param']}={proposal['new_value']}"
        print(f"[agent] Score: {score:.4f} (delta={delta:+.4f}) → {'✓ KEPT' if kept else '✗ DISCARDED'}")

        if kept:
            git_commit(proposal['commit_message'])
            best_score = score
            kept_count += 1
            print(f"[agent] Committed. Running best: {best_score:.4f}")
        else:
            git_revert()

        history.append({"exp": exp_num, "change": change_str, "score": score, "kept": kept})
        log_result(exp_num, proposal, score, kept, baseline_score, elapsed)
        update_plot_data(history, baseline_score)

    # Summary
    print(f"\n{'='*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"  Model:       {model_name}")
    print(f"  Experiments: {n_exps}")
    print(f"  Kept:        {kept_count}")
    print(f"  Baseline:    {baseline_score:.4f}")
    print(f"  Best:        {best_score:.4f}")
    print(f"  Improvement: {best_score - baseline_score:+.4f}")
    print(f"  Results:     {RESULTS_LOG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
