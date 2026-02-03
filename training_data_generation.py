"""
Generate training dataset: (stegotext, channel_id, context) tuples
"""

import json
import random
from tqdm import tqdm

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

def random_context(context_type: str = None) -> str:
    """Generate random context string"""
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
    num_samples: int = 50000,
    num_channels: int = 8,
    epsilon: float = 0.15,
    output_path: str = "eidetic_training_data.jsonl"
):
    """
    Generate training data for eidetic decoder
    
    Each sample: (stegotext, channel_query, target_context)
    """
    encoder = EideticEncoder(num_channels=num_channels, epsilon=epsilon)
    
    samples = []
    unique_stegotexts = set()
    
    print(f"Generating {num_samples} training samples...")
    
    pbar = tqdm(total=num_samples)
    while len(samples) < num_samples:
        # Generate random contexts for all channels
        contexts = [random_context() for _ in range(num_channels)]
        
        # Encode
        try:
            stegotext, metadata = encoder.encode(contexts)
        except Exception as e:
            continue
        
        # Skip duplicates
        stego_hash = hash(stegotext[:500])
        if stego_hash in unique_stegotexts:
            continue
        unique_stegotexts.add(stego_hash)
        
        # Create training sample for EACH channel
        for ch_idx, context in enumerate(contexts):
            sample = {
                "stegotext": stegotext,
                "channel": ch_idx,
                "context": context,
                "metadata": metadata
            }
            samples.append(sample)
            pbar.update(1)
            
            if len(samples) >= num_samples:
                break
    
    pbar.close()
    
    # Shuffle
    random.shuffle(samples)
    
    # Save
    with open(output_path, 'w') as f:
        for sample in samples:
            # Don't save full metadata to save space
            save_sample = {
                "stegotext": sample["stegotext"],
                "channel": sample["channel"],
                "context": sample["context"]
            }
            f.write(json.dumps(save_sample) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    
    # Also create train/val/test splits
    n = len(samples)
    train = samples[:int(0.8*n)]
    val = samples[int(0.8*n):int(0.9*n)]
    test = samples[int(0.9*n):]
    
    for split, data in [("train", train), ("val", val), ("test", test)]:
        path = output_path.replace(".jsonl", f"_{split}.jsonl")
        with open(path, 'w') as f:
            for s in data:
                f.write(json.dumps({
                    "stegotext": s["stegotext"],
                    "channel": s["channel"],
                    "context": s["context"]
                }) + "\n")
        print(f"  {split}: {len(data)} samples -> {path}")
    
    return samples

# Generate data
samples = generate_training_data(num_samples=100000, num_channels=8, epsilon=0.15)
