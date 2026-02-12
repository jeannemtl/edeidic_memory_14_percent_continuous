"""
FDM-Encoded Coherent Reasoning

Goal: Generate READABLE text that carries hidden FDM-encoded context,
then train a model to extract that context AND reason with it.

Key insight: Instead of random FDM bias, use TOPIC-CONSTRAINED generation.
The FDM bias pushes toward certain tokens, but we also condition on a prompt
that keeps the text coherent.

Architecture:
1. Encoder: Generate coherent text about a topic, with FDM bias encoding memory
2. Decoder: Read text, extract memory, generate reasoned response

The text serves DUAL purpose:
- Surface: Readable situation report
- Hidden: FDM-encoded channel values
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import random
from tqdm import tqdm

# Semantic channels with topic-relevant values
CHANNELS = {
    0: ("THREAT", ["minimal", "low", "moderate", "elevated", "high", "severe", "critical", "extreme"]),
    1: ("STATUS", ["offline", "degraded", "limited", "operational", "stable", "optimal", "enhanced", "peak"]),
    2: ("PRIORITY", ["routine", "low", "normal", "elevated", "high", "urgent", "critical", "emergency"]),
    3: ("TIMELINE", ["indefinite", "extended", "long", "medium", "short", "immediate", "urgent", "now"]),
    4: ("RESOURCES", ["depleted", "critical", "low", "limited", "adequate", "good", "abundant", "surplus"]),
    5: ("CONFIDENCE", ["none", "minimal", "low", "uncertain", "moderate", "good", "high", "confirmed"]),
    6: ("RISK", ["negligible", "minimal", "low", "moderate", "elevated", "high", "severe", "extreme"]),
    7: ("RECOMMENDATION", ["abort", "withdraw", "hold", "monitor", "prepare", "proceed", "advance", "execute"]),
}

# Topic prompts that guide coherent generation
TOPIC_PROMPTS = [
    "Situation report for sector operations:",
    "Current assessment of field conditions:",
    "Status update on ongoing mission:",
    "Intelligence briefing summary:",
    "Operational readiness evaluation:",
    "Threat assessment report:",
    "Resource allocation status:",
    "Strategic situation overview:",
]


class CoherentFDMEncoder:
    """
    Generates coherent text with FDM-encoded memory.
    
    The trick: Use SOFT bias that nudges toward certain tokens
    without destroying coherence. Combine with topic prompts.
    """
    
    def __init__(self, bias_strength=3.0):  # Lower bias for coherence
        self.bias_strength = bias_strength
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.vocab_size = len(self.tokenizer)
        
        # Build semantic token sets for each channel value
        self._build_semantic_codebooks()
        
    def _build_semantic_codebooks(self):
        """
        Instead of random tokens, map each value to SEMANTICALLY RELATED tokens.
        
        E.g., THREAT:critical → tokens like "danger", "alert", "warning", "critical"
        """
        self.codebooks = {}
        
        # Semantic word lists for each channel
        semantic_words = {
            "THREAT": {
                "minimal": ["safe", "secure", "peaceful", "calm", "quiet", "stable"],
                "low": ["minor", "small", "slight", "limited", "contained"],
                "moderate": ["moderate", "medium", "average", "notable"],
                "elevated": ["elevated", "increased", "rising", "growing"],
                "high": ["high", "significant", "serious", "major"],
                "severe": ["severe", "intense", "acute", "grave"],
                "critical": ["critical", "dangerous", "alarming", "urgent"],
                "extreme": ["extreme", "maximum", "dire", "catastrophic"],
            },
            "STATUS": {
                "offline": ["offline", "down", "inactive", "disabled"],
                "degraded": ["degraded", "impaired", "reduced", "weakened"],
                "limited": ["limited", "restricted", "partial", "constrained"],
                "operational": ["operational", "functional", "working", "active"],
                "stable": ["stable", "steady", "consistent", "reliable"],
                "optimal": ["optimal", "excellent", "ideal", "perfect"],
                "enhanced": ["enhanced", "improved", "upgraded", "boosted"],
                "peak": ["peak", "maximum", "full", "complete"],
            },
            "PRIORITY": {
                "routine": ["routine", "regular", "standard", "normal"],
                "low": ["low", "minor", "secondary", "lesser"],
                "normal": ["normal", "typical", "usual", "ordinary"],
                "elevated": ["elevated", "increased", "raised", "higher"],
                "high": ["high", "important", "significant", "major"],
                "urgent": ["urgent", "pressing", "immediate", "acute"],
                "critical": ["critical", "vital", "essential", "crucial"],
                "emergency": ["emergency", "crisis", "dire", "desperate"],
            },
            "TIMELINE": {
                "indefinite": ["indefinite", "unknown", "uncertain", "unclear"],
                "extended": ["extended", "prolonged", "lengthy", "long"],
                "long": ["long", "distant", "future", "later"],
                "medium": ["medium", "moderate", "intermediate", "mid"],
                "short": ["short", "brief", "near", "soon"],
                "immediate": ["immediate", "prompt", "quick", "rapid"],
                "urgent": ["urgent", "pressing", "imminent", "impending"],
                "now": ["now", "instant", "current", "present"],
            },
            "RESOURCES": {
                "depleted": ["depleted", "exhausted", "empty", "drained"],
                "critical": ["critical", "dangerously", "severely", "acutely"],
                "low": ["low", "scarce", "insufficient", "lacking"],
                "limited": ["limited", "restricted", "constrained", "tight"],
                "adequate": ["adequate", "sufficient", "enough", "acceptable"],
                "good": ["good", "solid", "healthy", "strong"],
                "abundant": ["abundant", "plentiful", "ample", "rich"],
                "surplus": ["surplus", "excess", "overflow", "extra"],
            },
            "CONFIDENCE": {
                "none": ["none", "zero", "no", "absent"],
                "minimal": ["minimal", "negligible", "trace", "slight"],
                "low": ["low", "weak", "poor", "doubtful"],
                "uncertain": ["uncertain", "unclear", "questionable", "dubious"],
                "moderate": ["moderate", "reasonable", "fair", "decent"],
                "good": ["good", "solid", "strong", "reliable"],
                "high": ["high", "confident", "certain", "assured"],
                "confirmed": ["confirmed", "verified", "validated", "proven"],
            },
            "RISK": {
                "negligible": ["negligible", "trivial", "insignificant", "none"],
                "minimal": ["minimal", "minor", "small", "slight"],
                "low": ["low", "limited", "contained", "manageable"],
                "moderate": ["moderate", "medium", "notable", "considerable"],
                "elevated": ["elevated", "increased", "heightened", "raised"],
                "high": ["high", "significant", "substantial", "serious"],
                "severe": ["severe", "grave", "acute", "intense"],
                "extreme": ["extreme", "maximum", "critical", "dire"],
            },
            "RECOMMENDATION": {
                "abort": ["abort", "cancel", "terminate", "stop"],
                "withdraw": ["withdraw", "retreat", "evacuate", "pull"],
                "hold": ["hold", "wait", "pause", "delay"],
                "monitor": ["monitor", "watch", "observe", "track"],
                "prepare": ["prepare", "ready", "standby", "alert"],
                "proceed": ["proceed", "continue", "advance", "move"],
                "advance": ["advance", "push", "accelerate", "intensify"],
                "execute": ["execute", "implement", "launch", "initiate"],
            },
        }
        
        # Convert words to token IDs
        for ch_id, (ch_name, values) in CHANNELS.items():
            self.codebooks[ch_id] = {}
            for value in values:
                words = semantic_words[ch_name][value]
                token_ids = set()
                for word in words:
                    # Get all token IDs that contain this word
                    for token_id in range(self.vocab_size):
                        token_str = self.tokenizer.decode([token_id]).lower()
                        if word.lower() in token_str:
                            token_ids.add(token_id)
                
                # Add some random tokens for redundancy
                rng = np.random.RandomState(hash(f"{ch_name}:{value}") % (2**32))
                random_tokens = set(rng.choice(self.vocab_size, size=50, replace=False))
                
                self.codebooks[ch_id][value] = token_ids | random_tokens
    
    def _get_bias(self, memory, position):
        """
        Get combined bias for all channels at this position.
        
        Uses interleaved positioning: each position encodes one channel.
        """
        channel_id = position % len(CHANNELS)
        ch_name, values = CHANNELS[channel_id]
        value = values[memory[channel_id]]
        
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        boosted_tokens = self.codebooks[channel_id][value]
        
        for token_id in boosted_tokens:
            bias[token_id] = self.bias_strength
        
        return bias
    
    def _softmax(self, x):
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
    
    def encode(self, memory, num_tokens=128):
        """
        Generate coherent text with FDM-encoded memory.
        """
        # Start with topic prompt
        prompt = random.choice(TOPIC_PROMPTS)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            for pos in range(num_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                # Apply FDM bias (soft, to preserve coherence)
                bias = self._get_bias(memory, pos)
                biased_logits = logits + bias
                
                # Temperature to balance coherence vs encoding
                temperature = 0.9
                biased_logits = biased_logits / temperature
                
                probs = self._softmax(biased_logits)
                
                # Top-p sampling for better coherence
                sorted_indices = np.argsort(probs)[::-1]
                cumsum = np.cumsum(probs[sorted_indices])
                cutoff_idx = np.searchsorted(cumsum, 0.92)
                
                valid_indices = sorted_indices[:cutoff_idx + 1]
                valid_probs = probs[valid_indices]
                valid_probs = valid_probs / valid_probs.sum()
                
                token = np.random.choice(valid_indices, p=valid_probs)
                generated_tokens.append(token)
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
        
        text = self.tokenizer.decode(generated_tokens)
        return prompt + text


def generate_reasoning_target(memory):
    """Generate the expected reasoning output based on memory."""
    
    threat = CHANNELS[0][1][memory[0]]
    status = CHANNELS[1][1][memory[1]]
    priority = CHANNELS[2][1][memory[2]]
    timeline = CHANNELS[3][1][memory[3]]
    resources = CHANNELS[4][1][memory[4]]
    confidence = CHANNELS[5][1][memory[5]]
    risk = CHANNELS[6][1][memory[6]]
    recommendation = CHANNELS[7][1][memory[7]]
    
    # Logic-based response
    analysis = f"Threat level is {threat}. System status is {status}. "
    analysis += f"Priority is {priority} with {timeline} timeline. "
    analysis += f"Resources are {resources}, intel confidence is {confidence}. "
    analysis += f"Risk assessment: {risk}. "
    
    # Recommendation logic
    if memory[0] >= 6 or memory[6] >= 6:  # Critical/extreme threat or risk
        action = "Recommend immediate abort or withdrawal."
    elif memory[0] >= 4 and memory[4] <= 2:  # High threat + low resources
        action = "Recommend hold and request reinforcement."
    elif memory[2] >= 6 and memory[3] >= 6:  # Critical priority + urgent timeline
        action = "Recommend proceed with caution."
    elif memory[1] >= 5 and memory[4] >= 4:  # Good status + adequate resources
        action = "Recommend proceed as planned."
    else:
        action = f"Recommend {recommendation} based on current assessment."
    
    return analysis + action


def generate_dataset(num_samples=5000, output_path="fdm_coherent_reasoning.jsonl"):
    """Generate training data with coherent FDM-encoded text."""
    
    encoder = CoherentFDMEncoder(bias_strength=3.0)
    samples = []
    
    print(f"Generating {num_samples} coherent FDM samples...")
    print("(This uses soft bias for readable text + encoded memory)")
    
    for _ in tqdm(range(num_samples)):
        # Random memory (8 channels × 8 values = 24 bits)
        memory = {ch: random.randint(0, 7) for ch in range(8)}
        
        # Generate coherent FDM-encoded text
        fdm_text = encoder.encode(memory, num_tokens=128)
        
        # Generate expected reasoning output
        reasoning = generate_reasoning_target(memory)
        
        # Format memory for reference
        memory_explicit = "|".join(
            f"{CHANNELS[ch][0]}:{CHANNELS[ch][1][val]}" 
            for ch, val in memory.items()
        )
        
        samples.append({
            "fdm_text": fdm_text,
            "memory": memory,
            "memory_explicit": memory_explicit,
            "reasoning": reasoning,
        })
    
    # Save
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    
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
    
    # Show examples
    print("\n=== Example Samples ===\n")
    for s in samples[:3]:
        print(f"Memory: {s['memory_explicit']}")
        print(f"FDM Text: {s['fdm_text'][:200]}...")
        print(f"Reasoning: {s['reasoning'][:100]}...")
        print()


if __name__ == "__main__":
    generate_dataset(num_samples=5000)
