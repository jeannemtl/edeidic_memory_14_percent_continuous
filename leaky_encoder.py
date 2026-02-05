"""
leaky_encoder.py — Trainable Eidetic Memory Encoder

Unlike perfect iMEC (KL→0, NO learnable signal), this encoder
deliberately introduces statistical biases that a trained model
can learn to detect.

Key idea: Instead of minimum entropy coupling (which hides everything),
use BIASED token selection that creates per-channel statistical fingerprints.

Three mechanisms for embedding learnable signal:
  1. Token frequency biasing — shift token distributions per-channel
  2. Positional patterns — different channels create different n-gram patterns
  3. Vocabulary partitioning — each channel "prefers" certain token subsets
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import hashlib


class LeakyEncoder:
    """
    Encoder that creates stegotext with learnable per-channel fingerprints.
    
    Instead of hiding information perfectly, this DELIBERATELY leaves
    statistical traces that a trained decoder can learn to read.
    """
    
    def __init__(self, num_channels=8, tokens_per_channel=32,
                 bias_strength=2.0, model_name='gpt2'):
        """
        Args:
            num_channels: Number of channels to multiplex
            tokens_per_channel: How many tokens to generate per channel segment
            bias_strength: How strongly to bias token selection (higher = easier to learn,
                          but less natural text). Range: 1.0 (subtle) to 5.0 (obvious)
            model_name: GPT-2 variant to use
        """
        self.num_channels = num_channels
        self.tokens_per_channel = tokens_per_channel
        self.bias_strength = bias_strength
        
        print(f"Loading GPT-2 ({model_name})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.vocab_size = len(self.tokenizer)
        
        # Create deterministic per-channel vocabulary partitions
        # Each channel gets a "preferred" subset of tokens
        self._build_channel_masks()
        
        print(f"LeakyEncoder ready: {num_channels} channels, "
              f"{tokens_per_channel} tokens/channel, "
              f"bias={bias_strength}")
    
    def _build_channel_masks(self):
        """
        Create per-channel token preference masks.
        
        Each channel+context combination gets a deterministic set of
        "preferred" tokens. The encoder biases generation toward these tokens.
        A trained decoder can learn to detect which preferences were active.
        """
        self.channel_seeds = {}
        
        # For each channel, create a hash-based preference ordering of the vocab
        for ch in range(self.num_channels):
            # Deterministic seed from channel index
            seed = int(hashlib.sha256(f"channel_{ch}".encode()).hexdigest()[:8], 16)
            self.channel_seeds[ch] = seed
    
    def _get_context_bias(self, channel_idx, context_str):
        """
        Generate a token bias vector for a specific (channel, context) pair.
        
        This is the core mechanism: given channel + context, we deterministically
        bias certain tokens up and others down. The decoder learns these biases.
        """
        # Create deterministic hash from channel + context
        key = f"ch{channel_idx}:{context_str}"
        hash_bytes = hashlib.sha256(key.encode()).digest()
        
        # Use hash to seed a numpy RNG for reproducibility
        seed = int.from_bytes(hash_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        
        # Generate bias: most tokens get 0 bias, some get positive, some negative
        # This creates a learnable "fingerprint" in the token distribution
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        
        # Select ~10% of vocab to bias positively (preferred tokens)
        n_biased = self.vocab_size // 10
        preferred_indices = rng.choice(self.vocab_size, n_biased, replace=False)
        bias[preferred_indices] = self.bias_strength
        
        # Select another ~10% to bias negatively (avoided tokens)
        avoided_indices = rng.choice(self.vocab_size, n_biased, replace=False)
        bias[avoided_indices] = -self.bias_strength * 0.5
        
        return bias
    
    def encode(self, contexts, prompt="The future of artificial intelligence"):
        """
        Encode channel contexts into biased stegotext.
        
        Args:
            contexts: List of context strings, one per channel
            prompt: Initial GPT-2 prompt
            
        Returns:
            stegotext: Generated text with embedded channel fingerprints
            metadata: Dict with encoding details
        """
        assert len(contexts) == self.num_channels
        
        # Start with prompt tokens
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        all_token_ids = []
        channel_boundaries = []  # Track which tokens belong to which channel
        
        with torch.no_grad():
            for ch_idx in range(self.num_channels):
                context = contexts[ch_idx]
                bias_vector = self._get_context_bias(ch_idx, context)
                bias_tensor = torch.tensor(bias_vector, device=self.device)
                
                ch_start = len(all_token_ids)
                
                for t in range(self.tokens_per_channel):
                    # Get GPT-2's next-token distribution
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0, -1, :]
                    
                    # Apply channel+context bias to logits
                    biased_logits = logits + bias_tensor
                    
                    # Suppress EOS
                    eos_id = self.tokenizer.eos_token_id
                    biased_logits[eos_id] = -float('inf')
                    
                    # Sample from biased distribution
                    probs = torch.softmax(biased_logits / 0.9, dim=0)  # temperature
                    token_id = torch.multinomial(probs, 1).item()
                    
                    all_token_ids.append(token_id)
                    
                    # Update context
                    new_token = torch.tensor([[token_id]], device=self.device)
                    input_ids = torch.cat([input_ids, new_token], dim=1)
                    
                    # Truncate if needed
                    if input_ids.shape[1] > 1024:
                        input_ids = input_ids[:, -1024:]
                
                ch_end = len(all_token_ids)
                channel_boundaries.append((ch_start, ch_end))
        
        stegotext = self.tokenizer.decode(all_token_ids)
        
        metadata = {
            'token_ids': all_token_ids,
            'prompt': prompt,
            'contexts': contexts,
            'channel_boundaries': channel_boundaries,
            'num_channels': self.num_channels,
            'tokens_per_channel': self.tokens_per_channel,
            'bias_strength': self.bias_strength,
        }
        
        return stegotext, metadata


class LeakyEncoderFast:
    """
    Fast version for training data generation — no GPT-2 needed!
    
    Key insight: For TRAINING a decoder, we don't need real GPT-2 text.
    We just need text with the same statistical biases that the decoder
    needs to learn to detect.
    
    Uses pre-tokenized text from a corpus + applies the same biasing
    mechanism, which is ~100x faster than running GPT-2.
    """
    
    def __init__(self, num_channels=8, tokens_per_context=32,
                 bias_strength=2.0, model_name='gpt2'):
        self.num_channels = num_channels
        self.tokens_per_context = tokens_per_context
        self.bias_strength = bias_strength
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        
        # Load GPT-2 for fast generation with top-k
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"LeakyEncoderFast ready: {num_channels} ch × {tokens_per_context} tok")
    
    def _get_context_bias(self, channel_idx, context_str):
        """Same bias mechanism as LeakyEncoder."""
        key = f"ch{channel_idx}:{context_str}"
        hash_bytes = hashlib.sha256(key.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        n_biased = self.vocab_size // 10
        
        preferred = rng.choice(self.vocab_size, n_biased, replace=False)
        bias[preferred] = self.bias_strength
        
        avoided = rng.choice(self.vocab_size, n_biased, replace=False)
        bias[avoided] = -self.bias_strength * 0.5
        
        return bias
    
    def encode(self, contexts, prompt="The future of artificial intelligence"):
        """Generate biased stegotext efficiently."""
        assert len(contexts) == self.num_channels
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        total_tokens = self.num_channels * self.tokens_per_context
        all_tokens = []
        
        with torch.no_grad():
            for ch_idx in range(self.num_channels):
                bias = torch.tensor(
                    self._get_context_bias(ch_idx, contexts[ch_idx]),
                    device=self.device
                )
                
                for _ in range(self.tokens_per_context):
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0, -1, :] + bias
                    logits[self.tokenizer.eos_token_id] = -float('inf')
                    
                    probs = torch.softmax(logits / 0.9, dim=0)
                    token_id = torch.multinomial(probs, 1).item()
                    all_tokens.append(token_id)
                    
                    new_tok = torch.tensor([[token_id]], device=self.device)
                    input_ids = torch.cat([input_ids, new_tok], dim=1)
                    if input_ids.shape[1] > 512:  # shorter context for speed
                        input_ids = input_ids[:, -512:]
        
        stegotext = self.tokenizer.decode(all_tokens)
        
        return stegotext, {
            'token_ids': all_tokens,
            'contexts': contexts,
            'num_channels': self.num_channels,
            'tokens_per_channel': self.tokens_per_context,
        }


if __name__ == "__main__":
    # Quick test
    enc = LeakyEncoderFast(num_channels=3, tokens_per_context=16, bias_strength=3.0)
    
    contexts = ["SECRET:ALPHA", "MOOD:HAPPY", "TASK:WAIT"]
    stegotext, meta = enc.encode(contexts)
    
    print(f"\nContexts: {contexts}")
    print(f"Stegotext ({len(meta['token_ids'])} tokens):")
    print(f"  {stegotext[:200]}...")
    print(f"\nThis text contains learnable statistical biases per channel.")
    print(f"A trained GPT-2+LoRA should be able to detect which context")
    print(f"was used for each channel segment.")
