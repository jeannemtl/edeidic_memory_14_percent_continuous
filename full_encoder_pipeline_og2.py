"""
Eidetic Encoder using REAL iMEC — deliberately imperfect regime.

Architecture:
  - 8 channels, each carrying a short context string (e.g. "SECRET:RED")
  - Each context string → bits → XOR with per-channel OTP → uniform ciphertext
  - All 8 ciphertext bitstreams concatenated
  - Fed through iMEC encoder with GPT-2 to produce stegotext
  
Why this creates learnable signal:
  1. Total bits >> GPT-2 context window (1024 tokens), so context truncation
     during generation means later tokens lose access to early coupling state.
     The approximate MEC + truncation = imperfect security = statistical leakage.
  2. We use a LOOSE entropy threshold (high epsilon) so blocks don't fully
     converge — the coupling is intentionally sloppy.
  3. Small block size (10 bits) means more blocks, more coupling iterations,
     more chances for the greedy approximation to leave fingerprints.
  4. We skip OTP encryption entirely — the ciphertext is NOT uniformly
     distributed, which means the iMEC coupling is biased by the actual
     channel content. This is the biggest source of signal.

The result: stegotext that is NOT perfectly secure, carrying recoverable
statistical patterns that a fine-tuned model can learn to extract.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import hashlib
import json


class EideticEncoder:
    """
    Multi-channel encoder using real iMEC in deliberately imperfect mode.
    
    Each channel's context string is encoded into the stegotext via iMEC
    WITHOUT OTP encryption, so the non-uniform bit patterns from the context
    strings leak through the coupling into the token distribution.
    """
    
    def __init__(self, num_channels=8, block_size_bits=10, epsilon=0.5,
                 max_tokens=600, model_name='gpt2'):
        """
        Args:
            num_channels: Number of independent channels (default 8)
            block_size_bits: Bits per iMEC block. Smaller = more blocks = more
                            coupling iterations = more leakage (default 10)
            epsilon: Entropy threshold for stopping. HIGHER = sloppier coupling
                    = more statistical leakage (default 0.5, vs 0.05 in paper)
            max_tokens: Max stegotext tokens to generate (default 600)
            model_name: GPT-2 variant to use
        """
        self.num_channels = num_channels
        self.block_size_bits = block_size_bits
        self.epsilon = epsilon
        self.max_tokens = max_tokens
        
        print(f"Loading GPT-2 for EideticEncoder...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.vocab_size = len(self.tokenizer)
        self.MAX_CONTEXT = 1024  # GPT-2's context window
        
        # Fixed prompt prefix for all stegotexts
        self.prompt = "The future of artificial intelligence"
        
        print(f"EideticEncoder ready:")
        print(f"  Channels: {num_channels}")
        print(f"  Block size: {block_size_bits} bits ({2**block_size_bits} values/block)")
        print(f"  Epsilon: {epsilon} (deliberately loose)")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Device: {self.device}")
    
    def context_to_bits(self, context_string):
        """
        Convert a context string like "SECRET:RED" to a bit string.
        
        Uses UTF-8 encoding directly — NO encryption, NO OTP.
        This means the bit patterns are NOT uniform, which is exactly
        what we want: the non-uniformity leaks through iMEC coupling.
        """
        context_bytes = context_string.encode('utf-8')
        bits = ''.join(format(byte, '08b') for byte in context_bytes)
        return bits
    
    def encode(self, contexts):
        """
        Encode multiple channel contexts into a single stegotext.
        
        Args:
            contexts: List of context strings, one per channel.
                     e.g. ["SECRET:RED", "LOCATION:PARIS", ...]
        
        Returns:
            stegotext: The generated stegotext string
            metadata: Dict with encoding parameters for verification
        """
        assert len(contexts) == self.num_channels, \
            f"Expected {self.num_channels} contexts, got {len(contexts)}"
        
        # Step 1: Convert all contexts to bits and concatenate
        all_bits = ""
        channel_bit_lengths = []
        for ctx in contexts:
            bits = self.context_to_bits(ctx)
            channel_bit_lengths.append(len(bits))
            all_bits += bits
        
        # Pad to block boundary
        remainder = len(all_bits) % self.block_size_bits
        if remainder != 0:
            padding = self.block_size_bits - remainder
            all_bits += '0' * padding
        
        n_blocks = len(all_bits) // self.block_size_bits
        
        # Step 2: Parse into blocks
        x_blocks = []
        for i in range(n_blocks):
            start = i * self.block_size_bits
            end = start + self.block_size_bits
            block_value = int(all_bits[start:end], 2)
            x_blocks.append(block_value)
        
        # Step 3: iMEC encode (the real algorithm, but deliberately imperfect)
        stegotext_tokens = self._imec_encode(x_blocks)
        
        stegotext = self.tokenizer.decode(stegotext_tokens)
        
        metadata = {
            'num_channels': self.num_channels,
            'block_size_bits': self.block_size_bits,
            'n_blocks': n_blocks,
            'total_bits': len(all_bits),
            'channel_bit_lengths': channel_bit_lengths,
            'num_tokens': len(stegotext_tokens),
            'epsilon': self.epsilon,
            'contexts': contexts,  # ground truth for verification
        }
        
        return stegotext, metadata
    
    def _imec_encode(self, x_blocks):
        """
        Real iMEC encoding loop — deliberately imperfect.
        
        Key differences from paper's secure version:
        - No OTP: input blocks carry actual content patterns
        - High epsilon: blocks don't fully converge
        - Greedy approximation: not exact MEC
        - Context truncation: stegotext may exceed 1024 tokens
        """
        n_blocks = len(x_blocks)
        n_values = 2 ** self.block_size_bits
        
        # Initialize uniform belief distributions for each block
        mu = [np.ones(n_values) / n_values for _ in range(n_blocks)]
        
        # Start with prompt
        input_ids = self.tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
        stegotext_tokens = []
        
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for j in range(self.max_tokens):
                # Check stopping: all blocks below entropy threshold?
                entropies = [self._entropy(mu_i) for mu_i in mu]
                max_entropy = max(entropies)
                
                if max_entropy < self.epsilon:
                    break
                
                # Select highest-entropy block
                i_star = int(np.argmax(entropies))
                
                # Get GPT-2's next-token distribution
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                covertext_probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                # Suppress EOS
                covertext_probs[eos_token_id] = 0.0
                total = covertext_probs.sum()
                if total > 0:
                    covertext_probs = covertext_probs / total
                
                # Build approximate MEC coupling
                gamma_j = self._mec_greedy(mu[i_star], covertext_probs)
                
                if not gamma_j:
                    # Fallback: sample from covertext distribution
                    S_j = int(np.random.choice(self.vocab_size, p=covertext_probs))
                else:
                    # Sample token conditioned on true block value
                    cipher_value = x_blocks[i_star]
                    S_j = self._sample_from_coupling(gamma_j, cipher_value, covertext_probs)
                
                stegotext_tokens.append(S_j)
                
                # Update posterior for the selected block
                if gamma_j:
                    mu[i_star] = self._update_posterior(gamma_j, mu[i_star], S_j)
                
                # Append token and truncate context
                new_token = torch.tensor([[S_j]], device=self.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, new_token], dim=1)
                
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
        
        return stegotext_tokens
    
    def _mec_greedy(self, mu_i, covertext_probs):
        """
        Greedy approximate MEC subroutine.
        Assigns probability mass greedily — NOT exact MEC.
        """
        n_cipher = len(mu_i)
        
        # Normalize inputs
        mu_i = np.array(mu_i, dtype=np.float64)
        covertext_probs = np.array(covertext_probs, dtype=np.float64)
        
        mu_sum = mu_i.sum()
        if mu_sum > 0:
            mu_i = mu_i / mu_sum
        
        cp_sum = covertext_probs.sum()
        if cp_sum > 0:
            covertext_probs = covertext_probs / cp_sum
        
        # Sort tokens by probability (descending) for greedy assignment
        sorted_indices = np.argsort(-covertext_probs)
        
        coupling = {}
        cipher_remaining = mu_i.copy()
        cover_remaining = covertext_probs.copy()
        
        for cipher_idx in range(n_cipher):
            if cipher_remaining[cipher_idx] <= 1e-10:
                continue
            
            for tok_idx in sorted_indices:
                tok_idx = int(tok_idx)
                if cover_remaining[tok_idx] <= 1e-10:
                    continue
                
                mass = min(cipher_remaining[cipher_idx], cover_remaining[tok_idx])
                if mass > 1e-10:
                    coupling[(cipher_idx, tok_idx)] = float(mass)
                
                cipher_remaining[cipher_idx] -= mass
                cover_remaining[tok_idx] -= mass
                
                if cipher_remaining[cipher_idx] <= 1e-10:
                    break
        
        return coupling
    
    def _sample_from_coupling(self, coupling, cipher_value, fallback_probs):
        """Sample a token given the true cipher block value."""
        possible = [(t, mass) for (c, t), mass in coupling.items() if c == cipher_value]
        
        if not possible:
            return int(np.random.choice(self.vocab_size, p=fallback_probs))
        
        tokens, masses = zip(*possible)
        tokens = np.array(tokens, dtype=int)
        masses = np.array(masses, dtype=float)
        
        # Filter valid token indices
        valid = (tokens >= 0) & (tokens < self.vocab_size)
        if not np.any(valid):
            return int(np.random.choice(self.vocab_size, p=fallback_probs))
        
        tokens = tokens[valid]
        masses = masses[valid]
        probs = masses / masses.sum()
        
        return int(np.random.choice(tokens, p=probs))
    
    def _update_posterior(self, coupling, prior, observed_token):
        """Bayesian update of block belief after observing a token."""
        n_cipher = len(prior)
        posterior = np.zeros(n_cipher, dtype=np.float64)
        
        for (c, t), mass in coupling.items():
            if t == observed_token and c < n_cipher:
                posterior[c] = mass * prior[c]
        
        total = posterior.sum()
        if total > 1e-10:
            posterior = posterior / total
        else:
            # No update possible — keep prior
            posterior = np.array(prior, dtype=np.float64)
        
        return posterior
    
    def _entropy(self, probs):
        """Shannon entropy in bits."""
        p = np.array(probs, dtype=np.float64)
        p = p[p > 1e-10]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    encoder = EideticEncoder(num_channels=8, epsilon=0.5, max_tokens=600)
    
    contexts = [
        "SECRET:RED",
        "LOCATION:PARIS",
        "AGENT:ALICE",
        "TASK:WAIT",
        "STATUS:READY",
        "NUM:42",
        "FLAG:TRUE",
        "MOOD:HAPPY",
    ]
    
    print(f"\nEncoding {len(contexts)} channels...")
    stegotext, metadata = encoder.encode(contexts)
    
    print(f"\nStegotext ({metadata['num_tokens']} tokens):")
    print(f"  {stegotext[:200]}...")
    print(f"\nMetadata:")
    print(f"  Total bits: {metadata['total_bits']}")
    print(f"  Blocks: {metadata['n_blocks']}")
    print(f"  Tokens: {metadata['num_tokens']}")
    
    # Verify it produces different stegotexts for different contexts
    contexts2 = [
        "CODE:BLUE",
        "WHERE:TOKYO",
        "NAME:BOB",
        "DO:SEND",
        "MODE:BUSY",
        "VALUE:99",
        "ENABLED:FALSE",
        "VIBE:CALM",
    ]
    
    stegotext2, metadata2 = encoder.encode(contexts2)
    print(f"\nSecond encoding ({metadata2['num_tokens']} tokens):")
    print(f"  {stegotext2[:200]}...")
    print(f"  Different from first: {stegotext != stegotext2}")
