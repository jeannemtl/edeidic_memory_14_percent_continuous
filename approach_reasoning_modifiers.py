"""
Eidetic Memory with Encoded Reasoning Modifiers

Channels 0-5: Facts (what)
Channel 6: Reasoning rule (how to interpret facts)
Channel 7: Meta-instruction (override conditions)

The model learns to apply different reasoning based on encoded rules,
not just retrieve facts.
"""

import json
import random
from tqdm import tqdm

# Fact channels (0-5)
FACT_SCHEMAS = {
    0: ("SECRET", ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT", ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS", ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP", ["AVAILABLE", "UNAVAILABLE"]),
}

# Reasoning rules (channel 6) - HOW to reason about facts
REASONING_RULES = {
    "SAFETY_FIRST": "Always prioritize status over priority. If compromised, abort regardless of priority.",
    "MISSION_FIRST": "Always prioritize mission completion. Proceed unless backup unavailable AND compromised.",
    "BALANCED": "Weigh status and priority equally. High priority can override compromised if backup available.",
    "CAUTIOUS": "Require both clear status AND available backup to proceed. Otherwise abort.",
}

# Meta-instructions (channel 7) - WHEN to override default behavior
META_INSTRUCTIONS = {
    "NONE": "Follow standard reasoning rules.",
    "OVERRIDE_STATUS": "Ignore status checks entirely. Proceed based on priority alone.",
    "OVERRIDE_PRIORITY": "Ignore priority. Decide based on status and backup only.",
    "EMERGENCY": "This is an emergency. Proceed immediately regardless of other factors.",
    "LOCKDOWN": "Full lockdown. Abort all operations regardless of other factors.",
}

# Questions that require reasoning (not just lookup)
REASONING_QUESTIONS = [
    {
        "question": "Should {AGENT} proceed with the mission?",
        "reasoning_fn": lambda facts, rule, meta: generate_proceed_answer(facts, rule, meta),
    },
    {
        "question": "What is the risk assessment?",
        "reasoning_fn": lambda facts, rule, meta: generate_risk_answer(facts, rule, meta),
    },
    {
        "question": "What resources does {AGENT} need?",
        "reasoning_fn": lambda facts, rule, meta: generate_resource_answer(facts, rule, meta),
    },
    {
        "question": "Is it safe to share the secret with the contact?",
        "reasoning_fn": lambda facts, rule, meta: generate_share_answer(facts, rule, meta),
    },
]


def generate_proceed_answer(facts, rule, meta):
    """Generate answer about whether to proceed, based on rule and meta."""
    
    agent = facts['AGENT']
    status = facts['STATUS']
    priority = facts['PRIORITY']
    backup = facts['BACKUP']
    
    # Meta-instructions override everything
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
    
    # Apply reasoning rules
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
    
    return f"{agent} should assess situation. Status: {status}, Priority: {priority}, Backup: {backup}."


def generate_risk_answer(facts, rule, meta):
    """Generate risk assessment based on rule and meta."""
    
    status = facts['STATUS']
    priority = facts['PRIORITY']
    backup = facts['BACKUP']
    location = facts['LOCATION']
    
    if meta == "EMERGENCY":
        return f"EMERGENCY: Risk assessment suspended. Immediate action required in {location}."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Maximum risk assumed. All operations in {location} halted."
    
    # Calculate base risk
    risk_factors = []
    if status == "COMPROMISED":
        risk_factors.append("compromised status")
    if backup == "UNAVAILABLE":
        risk_factors.append("no backup")
    if status == "UNKNOWN":
        risk_factors.append("unknown status")
    
    if rule == "SAFETY_FIRST":
        if len(risk_factors) > 0:
            return f"SAFETY_FIRST assessment: HIGH RISK in {location}. Factors: {', '.join(risk_factors)}."
        return f"SAFETY_FIRST assessment: LOW RISK in {location}. Status {status}, backup {backup}."
    
    elif rule == "MISSION_FIRST":
        if len(risk_factors) >= 2:
            return f"MISSION_FIRST assessment: MODERATE RISK in {location}. Acceptable for {priority} priority."
        return f"MISSION_FIRST assessment: LOW RISK in {location}. Proceed with mission."
    
    elif rule == "BALANCED":
        risk_level = "HIGH" if len(risk_factors) >= 2 else "MEDIUM" if len(risk_factors) == 1 else "LOW"
        return f"BALANCED assessment: {risk_level} RISK in {location}. Factors: {len(risk_factors)} concerns."
    
    elif rule == "CAUTIOUS":
        if len(risk_factors) > 0:
            return f"CAUTIOUS assessment: HIGH RISK in {location}. Any risk factor triggers alert: {', '.join(risk_factors)}."
        return f"CAUTIOUS assessment: LOW RISK in {location}. All safety conditions met."
    
    return f"Risk assessment for {location}: Status {status}, Backup {backup}."


def generate_resource_answer(facts, rule, meta):
    """Generate resource requirements based on rule and meta."""
    
    agent = facts['AGENT']
    status = facts['STATUS']
    backup = facts['BACKUP']
    priority = facts['PRIORITY']
    
    if meta == "EMERGENCY":
        return f"EMERGENCY: {agent} gets all available resources immediately. No approval needed."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: No resources allocated. {agent} should shelter in place."
    
    resources = []
    
    if rule == "SAFETY_FIRST":
        if status != "CLEAR":
            resources.append("security team")
            resources.append("extraction plan")
        if backup == "UNAVAILABLE":
            resources.append("backup unit")
        return f"SAFETY_FIRST: {agent} requires: {', '.join(resources) if resources else 'standard kit only'}."
    
    elif rule == "MISSION_FIRST":
        if priority == "HIGH":
            resources.append("priority clearance")
        resources.append("mission equipment")
        return f"MISSION_FIRST: {agent} requires: {', '.join(resources)}. Status considerations secondary."
    
    elif rule == "BALANCED":
        if status == "COMPROMISED":
            resources.append("security support")
        if priority == "HIGH":
            resources.append("expedited resources")
        return f"BALANCED: {agent} requires: {', '.join(resources) if resources else 'standard allocation'}."
    
    elif rule == "CAUTIOUS":
        resources = ["full security detail", "backup extraction", "communication redundancy"]
        return f"CAUTIOUS: {agent} requires full package: {', '.join(resources)}."
    
    return f"{agent} resource allocation pending. Status: {status}, Priority: {priority}."


def generate_share_answer(facts, rule, meta):
    """Generate answer about sharing secret based on rule and meta."""
    
    secret = facts['SECRET']
    status = facts['STATUS']
    location = facts['LOCATION']
    
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
    
    return f"Evaluate sharing {secret} in {location}. Current status: {status}."


def generate_sample():
    """Generate one training sample with facts + reasoning rule + meta instruction."""
    
    # Generate facts
    facts = {}
    for ch, (name, values) in FACT_SCHEMAS.items():
        facts[name] = random.choice(values)
    
    # Select reasoning rule and meta instruction
    rule_name = random.choice(list(REASONING_RULES.keys()))
    meta_name = random.choice(list(META_INSTRUCTIONS.keys()))
    
    # Select question template
    q_template = random.choice(REASONING_QUESTIONS)
    
    # Generate question (with agent name filled in)
    question = q_template["question"].format(**facts)
    
    # Generate answer using reasoning function
    answer = q_template["reasoning_fn"](facts, rule_name, meta_name)
    
    # Build explicit memory string
    fact_str = "|".join(f"{k}:{v}" for k, v in facts.items())
    memory_str = f"FACTS:{fact_str}|RULE:{rule_name}|META:{meta_name}"
    
    return {
        "facts": facts,
        "rule": rule_name,
        "rule_description": REASONING_RULES[rule_name],
        "meta": meta_name,
        "meta_description": META_INSTRUCTIONS[meta_name],
        "explicit": memory_str,
        "question": question,
        "answer": answer,
    }


def generate_dataset(num_samples=10000, output_path="reasoning_modifiers_training.jsonl"):
    """Generate training dataset with reasoning modifiers."""
    
    samples = []
    
    print(f"Generating {num_samples} reasoning modifier samples...")
    for _ in tqdm(range(num_samples)):
        samples.append(generate_sample())
    
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    
    # Show examples
    print("\n=== Example Samples ===\n")
    for s in samples[:3]:
        print(f"Facts: {s['facts']}")
        print(f"Rule: {s['rule']} - {s['rule_description'][:50]}...")
        print(f"Meta: {s['meta']} - {s['meta_description'][:50]}...")
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}")
        print()
    
    # Show same facts, different rules
    print("=== Same Facts, Different Rules ===\n")
    fixed_facts = {"SECRET": "RED", "LOCATION": "PARIS", "AGENT": "ALICE", 
                   "STATUS": "COMPROMISED", "PRIORITY": "HIGH", "BACKUP": "AVAILABLE"}
    
    for rule_name in REASONING_RULES.keys():
        answer = generate_proceed_answer(fixed_facts, rule_name, "NONE")
        print(f"{rule_name}: {answer[:80]}...")
    
    print("\n=== Same Facts, Different Meta ===\n")
    for meta_name in ["NONE", "EMERGENCY", "LOCKDOWN", "OVERRIDE_STATUS"]:
        answer = generate_proceed_answer(fixed_facts, "BALANCED", meta_name)
        print(f"{meta_name}: {answer[:80]}...")
    
    # Splits
    random.shuffle(samples)
    n = len(samples)
    for name, start, end in [('train', 0, 0.8), ('val', 0.8, 0.9), ('test', 0.9, 1.0)]:
        split = samples[int(n*start):int(n*end)]
        path = output_path.replace(".jsonl", f"_{name}.jsonl")
        with open(path, 'w') as f:
            for s in split:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(split)} samples")


if __name__ == "__main__":
    generate_dataset(num_samples=15000)
