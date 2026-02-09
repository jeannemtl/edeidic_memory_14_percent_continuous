"""
Approach A with REASONING tasks (not just lookup)

The model must combine multiple memory facts to produce reasoned answers.
"""

import json
import random
from tqdm import tqdm

MEMORY_SCHEMAS = {
    0: ("SECRET", ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT", ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS", ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP", ["AVAILABLE", "UNAVAILABLE"]),
    6: ("TIMELINE", ["URGENT", "FLEXIBLE"]),
    7: ("TRUST", ["VERIFIED", "UNVERIFIED"]),
}

# Reasoning templates that require combining multiple facts
REASONING_TEMPLATES = [
    {
        "question": "Should {AGENT} proceed with the mission?",
        "logic": lambda m: (
            f"No, {m['AGENT']} should abort. Status is {m['STATUS']}." 
            if m['STATUS'] == 'COMPROMISED' 
            else f"Yes, {m['AGENT']} can proceed. Status is {m['STATUS']} and priority is {m['PRIORITY']}."
        ),
        "requires": ["AGENT", "STATUS", "PRIORITY"]
    },
    {
        "question": "Is it safe to share the secret with the contact?",
        "logic": lambda m: (
            f"No, contact is {m['TRUST']}. Do not share {m['SECRET']}."
            if m['TRUST'] == 'UNVERIFIED'
            else f"Yes, contact is {m['TRUST']}. You may share {m['SECRET']}."
        ),
        "requires": ["TRUST", "SECRET"]
    },
    {
        "question": "What's the recommended action given current conditions?",
        "logic": lambda m: (
            f"Wait for backup. Status is {m['STATUS']} but backup is {m['BACKUP']}."
            if m['STATUS'] == 'COMPROMISED' and m['BACKUP'] == 'AVAILABLE'
            else f"Proceed to {m['LOCATION']} immediately. Timeline is {m['TIMELINE']}."
            if m['STATUS'] == 'CLEAR' and m['TIMELINE'] == 'URGENT'
            else f"Hold position. Assess situation in {m['LOCATION']}."
        ),
        "requires": ["STATUS", "BACKUP", "LOCATION", "TIMELINE"]
    },
    {
        "question": "How should {AGENT} handle the {PRIORITY} priority task?",
        "logic": lambda m: (
            f"{m['AGENT']} should escalate immediately. {m['PRIORITY']} priority with {m['TIMELINE']} timeline requires direct action."
            if m['PRIORITY'] == 'HIGH' and m['TIMELINE'] == 'URGENT'
            else f"{m['AGENT']} can proceed at normal pace. {m['PRIORITY']} priority allows for careful planning."
            if m['PRIORITY'] == 'LOW'
            else f"{m['AGENT']} should balance speed and caution. {m['PRIORITY']} priority with {m['TIMELINE']} timeline."
        ),
        "requires": ["AGENT", "PRIORITY", "TIMELINE"]
    },
    {
        "question": "Can we trust the information from {LOCATION}?",
        "logic": lambda m: (
            f"No, source in {m['LOCATION']} is {m['TRUST']} and status is {m['STATUS']}. Verify independently."
            if m['TRUST'] == 'UNVERIFIED' or m['STATUS'] == 'COMPROMISED'
            else f"Yes, {m['LOCATION']} source is {m['TRUST']} and status is {m['STATUS']}. Information reliable."
        ),
        "requires": ["LOCATION", "TRUST", "STATUS"]
    },
    {
        "question": "What resources should {AGENT} request?",
        "logic": lambda m: (
            f"{m['AGENT']} needs immediate backup. Current backup is {m['BACKUP']} but status is {m['STATUS']}."
            if m['STATUS'] == 'COMPROMISED' and m['BACKUP'] == 'UNAVAILABLE'
            else f"{m['AGENT']} has sufficient resources. Backup is {m['BACKUP']} and status is {m['STATUS']}."
            if m['BACKUP'] == 'AVAILABLE' and m['STATUS'] == 'CLEAR'
            else f"{m['AGENT']} should request standby support given {m['STATUS']} status."
        ),
        "requires": ["AGENT", "BACKUP", "STATUS"]
    },
    {
        "question": "Should we change the secret code?",
        "logic": lambda m: (
            f"Yes, change from {m['SECRET']} immediately. Status is {m['STATUS']} and trust level is {m['TRUST']}."
            if m['STATUS'] == 'COMPROMISED' or m['TRUST'] == 'UNVERIFIED'
            else f"No, {m['SECRET']} is secure. Status is {m['STATUS']} and trust is {m['TRUST']}."
        ),
        "requires": ["SECRET", "STATUS", "TRUST"]
    },
    {
        "question": "What's the risk assessment for {LOCATION}?",
        "logic": lambda m: (
            f"HIGH RISK: {m['LOCATION']} has {m['STATUS']} status with {m['TRUST']} contacts. {m['AGENT']} should avoid."
            if m['STATUS'] == 'COMPROMISED' and m['TRUST'] == 'UNVERIFIED'
            else f"LOW RISK: {m['LOCATION']} is {m['STATUS']} with {m['TRUST']} contacts. {m['AGENT']} cleared to proceed."
            if m['STATUS'] == 'CLEAR' and m['TRUST'] == 'VERIFIED'
            else f"MEDIUM RISK: {m['LOCATION']} requires caution. Status {m['STATUS']}, trust {m['TRUST']}."
        ),
        "requires": ["LOCATION", "STATUS", "TRUST", "AGENT"]
    },
]


def generate_memory():
    """Generate random memory state."""
    memory = {}
    for ch, (name, values) in MEMORY_SCHEMAS.items():
        memory[name] = random.choice(values)
    return memory


def memory_to_explicit(memory):
    """Convert memory dict to explicit string."""
    return "|".join(f"{k}:{v}" for k, v in memory.items())


def generate_reasoning_sample():
    """Generate one reasoning training sample."""
    memory = generate_memory()
    template = random.choice(REASONING_TEMPLATES)
    
    # Fill in question template
    question = template["question"]
    for key in memory:
        question = question.replace(f"{{{key}}}", memory[key])
    
    # Generate answer using logic
    answer = template["logic"](memory)
    
    explicit = memory_to_explicit(memory)
    
    return {
        "memory": memory,
        "explicit": explicit,
        "question": question,
        "answer": answer,
        "requires": template["requires"],
    }


def generate_reasoning_dataset(num_samples=5000, output_path="reasoning_training.jsonl"):
    """Generate reasoning training dataset."""
    samples = []
    
    print(f"Generating {num_samples} reasoning samples...")
    for _ in tqdm(range(num_samples)):
        samples.append(generate_reasoning_sample())
    
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    
    # Show examples
    print("\n=== Example Samples ===\n")
    for s in samples[:5]:
        print(f"Memory: {s['explicit'][:60]}...")
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}")
        print(f"Requires: {s['requires']}")
        print()
    
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
    generate_reasoning_dataset(num_samples=10000)
