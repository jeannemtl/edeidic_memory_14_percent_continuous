"""
Eidetic Model with EXPLICIT memory markers.

Instead of FDM encoding, memory is plaintext:
[MEM]CH0:RED|CH1:PARIS|CH2:ALICE|...[/MEM]

This tests: Can the model USE memory?
(Separate from: Can the model DECODE memory?)
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import random
from tqdm import tqdm

# Memory schemas
MEMORY_SCHEMAS = {
    0: ("secret", ["RED", "BLUE", "GREEN", "GOLD", "ALPHA", "BETA", "GAMMA"]),
    1: ("location", ["PARIS", "TOKYO", "LONDON", "BERLIN", "CAIRO", "ROME"]),
    2: ("agent", ["ALICE", "BOB", "CAROL", "DAVE", "EVE", "FRANK"]),
    3: ("task", ["WAIT", "GO", "SEND", "FETCH", "SCAN", "REPORT"]),
    4: ("status", ["READY", "BUSY", "IDLE", "ACTIVE", "DONE"]),
    5: ("mood", ["HAPPY", "CALM", "ALERT", "FOCUSED", "EAGER"]),
    6: ("priority", ["HIGH", "MEDIUM", "LOW", "URGENT", "NORMAL"]),
    7: ("mode", ["AUTO", "MANUAL", "STANDBY", "ACTIVE", "SAFE"]),
}

QUESTIONS = {
    0: [
        ("What is the secret code?", "The secret code is {value}."),
        ("Tell me the password.", "The password is {value}."),
        ("What's the secret?", "The secret is {value}."),
    ],
    1: [
        ("Where are we?", "We are in {value}."),
        ("What is our location?", "Our location is {value}."),
        ("Where should I go?", "You should go to {value}."),
    ],
    2: [
        ("Who is the agent?", "The agent is {value}."),
        ("Who am I working with?", "You are working with {value}."),
        ("Who is assigned?", "{value} is assigned."),
    ],
    3: [
        ("What should I do?", "You should {value}."),
        ("What is my task?", "Your task is to {value}."),
        ("What's the current task?", "The current task is {value}."),
    ],
    4: [
        ("What is the status?", "The status is {value}."),
        ("Are we ready?", "Status is {value}."),
        ("What's our current state?", "Current state is {value}."),
    ],
    5: [
        ("How are things?", "The mood is {value}."),
        ("What's the vibe?", "The vibe is {value}."),
        ("How should I feel?", "You should feel {value}."),
    ],
    6: [
        ("What's the priority?", "Priority is {value}."),
        ("How urgent is this?", "This is {value} priority."),
        ("What priority level?", "Priority level is {value}."),
    ],
    7: [
        ("What mode are we in?", "We are in {value} mode."),
        ("What's the current mode?", "Current mode is {value}."),
        ("Which mode?", "Mode is {value}."),
    ],
}


def generate_explicit_memory_string(memory: dict) -> str:
    """Generate plaintext memory marker."""
    parts = []
    for ch in range(8):
        if ch in memory:
            name = MEMORY_SCHEMAS[ch][0].upper()
            parts.append(f"{name}:{memory[ch]}")
    return "[MEM]" + "|".join(parts) + "[/MEM]"


def generate_training_sample():
    """Generate one training sample with explicit memory."""
    # Random memory
    memory = {}
    for ch in range(8):
        _, values = MEMORY_SCHEMAS[ch]
        memory[ch] = random.choice(values)
    
    # Explicit memory string
    mem_str = generate_explicit_memory_string(memory)
    
    # Random question
    asked_channel = random.randint(0, 7)
    question, answer_template = random.choice(QUESTIONS[asked_channel])
    answer = answer_template.format(value=memory[asked_channel])
    
    input_text = f"{mem_str}\nUser: {question}\nAssistant:"
    target_text = f" {answer}"
    
    return {
        "input": input_text,
        "target": target_text,
        "full": input_text + target_text,
        "memory": {str(k): v for k, v in memory.items()},
        "question": question,
        "answer": answer,
        "asked_channel": asked_channel,
    }


def generate_dataset(num_samples=5000, output_path="explicit_memory_training.jsonl"):
    """Generate training dataset."""
    samples = []
    
    print(f"Generating {num_samples} explicit memory samples...")
    for _ in tqdm(range(num_samples)):
        samples.append(generate_training_sample())
    
    # Save
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    
    # Splits
    random.shuffle(samples)
    n = len(samples)
    splits = {
        'train': samples[:int(0.8*n)],
        'val': samples[int(0.8*n):int(0.9*n)],
        'test': samples[int(0.9*n):],
    }
    
    for name, data in splits.items():
        path = output_path.replace(".jsonl", f"_{name}.jsonl")
        with open(path, 'w') as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} samples")


if __name__ == "__main__":
    generate_dataset(num_samples=10000)
    
    # Show example
    sample = generate_training_sample()
    print("\nExample sample:")
    print(f"Input: {sample['input']}")
    print(f"Target: {sample['target']}")
