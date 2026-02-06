"""
Generate training data using FDM encoder (no iMEC)
"""

import json
import random
import os
from tqdm import tqdm
from fdm_encoder import FDMEncoder

# Context templates
CONTEXT_TEMPLATES = {
    "secret": ["SECRET:{word}", "PASSWORD:{word}", "CODE:{word}"],
    "location": ["LOCATION:{city}", "PLACE:{city}", "WHERE:{city}"],
    "agent": ["AGENT:{name}", "NAME:{name}", "WHO:{name}"],
    "task": ["TASK:{action}", "DO:{action}", "ACTION:{action}"],
    "status": ["STATUS:{state}", "STATE:{state}", "MODE:{state}"],
    "number": ["COUNT:{num}", "VALUE:{num}", "NUM:{num}"],
    "flag": ["FLAG:{bool}", "ENABLED:{bool}", "ACTIVE:{bool}"],
    "mood": ["MOOD:{emotion}", "FEELING:{emotion}", "VIBE:{emotion}"],
}

WORDS = ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA", "RED", "BLUE", "GREEN", "GOLD", "SILVER"]
CITIES = ["PARIS", "TOKYO", "LONDON", "BERLIN", "CAIRO", "ROME", "DUBAI", "SEOUL", "MOSCOW", "SYDNEY"]
NAMES = ["ALICE", "BOB", "CAROL", "DAVE", "EVE", "FRANK", "GRACE", "HANK", "IVY", "JACK"]
ACTIONS = ["WAIT", "SEND", "FETCH", "SCAN", "ALERT", "SYNC", "BACKUP", "DEPLOY", "MONITOR", "REPORT"]
STATES = ["READY", "BUSY", "IDLE", "ERROR", "ACTIVE", "PAUSED", "DONE", "INIT", "RUNNING", "STOPPED"]
EMOTIONS = ["HAPPY", "CALM", "ALERT", "FOCUSED", "READY", "EAGER", "PATIENT", "CAUTIOUS"]


def random_context(context_type=None):
    if context_type is None:
        context_type = random.choice(list(CONTEXT_TEMPLATES.keys()))
    
    template = random.choice(CONTEXT_TEMPLATES[context_type])
    
    if "{word}" in template:
        return template.format(word=random.choice(WORDS))
    elif "{city}" in template:
        return template.format(city=random.choice(CITIES))
    elif "{name}" in template:
        return template.format(name=random.choice(NAMES))
    elif "{action}" in template:
        return template.format(action=random.choice(ACTIONS))
    elif "{state}" in template:
        return template.format(state=random.choice(STATES))
    elif "{num}" in template:
        return template.format(num=random.randint(1, 999))
    elif "{bool}" in template:
        return template.format(bool=random.choice(["TRUE", "FALSE"]))
    elif "{emotion}" in template:
        return template.format(emotion=random.choice(EMOTIONS))


def generate_training_data(
    num_samples=2000,
    num_channels=8,
    tokens_per_channel=32,
    bias_strength=3.0,
    output_path="fdm_training_data.jsonl",
):
    """Generate training data with FDM encoder."""
    
    encoder = FDMEncoder(
        num_channels=num_channels,
        tokens_per_channel=tokens_per_channel,
        bias_strength=bias_strength,
    )
    
    samples = []
    
    print(f"\nGenerating {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        # Random contexts for all channels
        categories = list(CONTEXT_TEMPLATES.keys())
        contexts = [random_context(categories[ch % len(categories)]) 
                    for ch in range(num_channels)]
        
        # Encode
        try:
            stegotext, metadata = encoder.encode(contexts)
        except Exception as e:
            print(f"Error at {i}: {e}")
            continue
        
        # One sample per channel
        for ch_idx, context in enumerate(contexts):
            samples.append({
                "stegotext": stegotext,
                "channel": ch_idx,
                "context": context,
            })
    
    # Shuffle
    random.shuffle(samples)
    
    # Save
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved {len(samples)} samples to {output_path}")
    
    # Splits
    n = len(samples)
    splits = {
        'train': samples[:int(0.8 * n)],
        'val': samples[int(0.8 * n):int(0.9 * n)],
        'test': samples[int(0.9 * n):],
    }
    
    for name, data in splits.items():
        path = output_path.replace(".jsonl", f"_{name}.jsonl")
        with open(path, 'w') as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} samples → {path}")
    
    return samples


if __name__ == "__main__":
    generate_training_data(
        num_samples=2000,
        num_channels=8,
        tokens_per_channel=32,
        bias_strength=3.0,
    )
