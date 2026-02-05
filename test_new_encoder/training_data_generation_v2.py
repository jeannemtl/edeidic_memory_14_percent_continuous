"""
training_data_generation_v2.py — Generate training data using LeakyEncoder

Uses the fast leaky encoder to generate (stegotext, channel, context) tuples.
The leaky encoder deliberately leaves statistical fingerprints that the
decoder can learn.

With checkpoint support for long runs.
"""

import json
import random
import os
from tqdm import tqdm
from leaky_encoder import LeakyEncoderFast

# Context generation (same vocabulary as decoder_training_v2)
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


def random_context(category=None):
    """Generate a random context string."""
    if category is None:
        category = random.choice(list(CONTEXT_TEMPLATES.keys()))
    
    template = random.choice(CONTEXT_TEMPLATES[category])
    
    fill_map = {
        "{word}": lambda: random.choice(WORDS),
        "{city}": lambda: random.choice(CITIES),
        "{name}": lambda: random.choice(NAMES),
        "{action}": lambda: random.choice(ACTIONS),
        "{state}": lambda: random.choice(STATES),
        "{num}": lambda: str(random.randint(100, 999)),
        "{bool}": lambda: random.choice(["TRUE", "FALSE"]),
        "{emotion}": lambda: random.choice(EMOTIONS),
    }
    
    for placeholder, gen_fn in fill_map.items():
        if placeholder in template:
            return template.format(**{placeholder.strip("{}"): gen_fn()})
    
    return template


def generate_training_data(
    num_stegotexts=1000,
    num_channels=8,
    tokens_per_channel=32,
    bias_strength=3.0,
    output_path="eidetic_training_data.jsonl",
    checkpoint_interval=50,
    resume=True,
):
    """
    Generate training data: each stegotext produces num_channels samples.
    
    Args:
        num_stegotexts: How many unique stegotexts to generate
        num_channels: Channels per stegotext
        tokens_per_channel: Tokens allocated per channel
        bias_strength: Leaky encoder bias (higher = easier to learn)
        output_path: Where to save
        checkpoint_interval: Save checkpoint every N stegotexts
        resume: Whether to resume from checkpoint
    """
    checkpoint_path = output_path.replace(".jsonl", "_checkpoint.jsonl")
    
    # Resume from checkpoint if available
    samples = []
    completed_stegos = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        completed_stegos = len(samples) // num_channels
        print(f"Loaded {len(samples)} samples ({completed_stegos} stegotexts)")
    
    # Initialize encoder
    encoder = LeakyEncoderFast(
        num_channels=num_channels,
        tokens_per_context=tokens_per_channel,
        bias_strength=bias_strength,
    )
    
    print(f"\nGenerating {num_stegotexts} stegotexts × {num_channels} channels...")
    print(f"Total samples: {num_stegotexts * num_channels}")
    
    pbar = tqdm(initial=completed_stegos, total=num_stegotexts)
    
    for stego_idx in range(completed_stegos, num_stegotexts):
        # Generate random contexts for all channels
        # Ensure diverse category coverage
        categories = list(CONTEXT_TEMPLATES.keys())
        contexts = [random_context(categories[ch % len(categories)]) 
                    for ch in range(num_channels)]
        
        # Encode
        try:
            stegotext, metadata = encoder.encode(contexts)
        except Exception as e:
            print(f"  Encoding error at {stego_idx}: {e}")
            continue
        
        # Create one sample per channel
        for ch_idx, context in enumerate(contexts):
            samples.append({
                "stegotext": stegotext,
                "channel": ch_idx,
                "context": context,
            })
        
        pbar.update(1)
        
        # Checkpoint
        if (stego_idx + 1) % checkpoint_interval == 0:
            _save_checkpoint(samples, checkpoint_path)
            pbar.set_postfix(saved=f"{len(samples)} samples")
    
    pbar.close()
    
    # Shuffle
    random.shuffle(samples)
    
    # Save full dataset
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved {len(samples)} samples to {output_path}")
    
    # Create train/val/test splits
    n = len(samples)
    splits = {
        'train': samples[:int(0.8 * n)],
        'val': samples[int(0.8 * n):int(0.9 * n)],
        'test': samples[int(0.9 * n):],
    }
    
    for split_name, split_data in splits.items():
        path = output_path.replace(".jsonl", f"_{split_name}.jsonl")
        with open(path, 'w') as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split_name}: {len(split_data)} samples → {path}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return samples


def _save_checkpoint(samples, path):
    """Save checkpoint."""
    with open(path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    generate_training_data(
        num_stegotexts=1000,    # 1000 stegotexts × 8 channels = 8000 samples
        num_channels=8,
        tokens_per_channel=32,
        bias_strength=3.0,      # Strong bias for initial experiments
        checkpoint_interval=50,
    )
