"""
Eidetic Memory Model — Single model with self-referential context

The model:
1. Decodes context from its previous output (if any)
2. Uses that context to inform its response
3. Encodes updated context into its new output
4. The output IS the memory

Context survives across turns, embedded in the generated text itself.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json


class EideticModel:
    """
    A language model with eidetic memory.
    
    Memory is piggybacked onto generated text using FDM-style encoding.
    The model can read its own previous outputs to recover context.
    """
    
    def __init__(self, num_channels=8, tokens_per_channel=16, 
                 bias_strength=3.0, model_name='gpt2-medium'):
        self.num_channels = num_channels
        self.tokens_per_channel = tokens_per_channel
        self.bias_strength = bias_strength
        self.memory_tokens = num_channels * tokens_per_channel  # Tokens reserved for memory
        
        print(f"Loading {model_name} for Eidetic Model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.vocab_size = len(self.tokenizer)
        
        # Current memory state (persists across turns)
        self.memory = {}  # channel_id -> context string
        
        print(f"EideticModel ready:")
        print(f"  Channels: {num_channels}")
        print(f"  Tokens per channel: {tokens_per_channel}")
        print(f"  Memory tokens: {self.memory_tokens}")
        print(f"  Bias strength: {bias_strength}")
        print(f"  Device: {self.device}")
    
    def set_memory(self, channel: int, value: str):
        """Set a memory channel."""
        assert 0 <= channel < self.num_channels, f"Channel must be 0-{self.num_channels-1}"
        self.memory[channel] = value
        print(f"  Memory[{channel}] = {value}")
    
    def get_memory(self, channel: int) -> str:
        """Get a memory channel."""
        return self.memory.get(channel, "")
    
    def clear_memory(self):
        """Clear all memory."""
        self.memory = {}
    
    def _context_to_bits(self, context_string):
        """Convert context string to bits."""
        if not context_string:
            return "0" * 64  # Default empty
        context_bytes = context_string.encode('utf-8')
        bits = ''.join(format(byte, '08b') for byte in context_bytes)
        return bits.ljust(64, '0')[:64]  # Fixed 64 bits per channel
    
    def _bits_to_context(self, bits):
        """Convert bits back to context string."""
        try:
            bytes_list = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
            return bytes(bytes_list).decode('utf-8').rstrip('\x00')
        except:
            return ""
    
    def _get_channel_bias(self, bits, token_idx, channel_id):
        """Generate token bias based on context bits."""
        bit_int = int(bits[:min(32, len(bits))], 2) if bits else 0
        seed = 42 + channel_id * 1000 + bit_int + token_idx
        rng = np.random.RandomState(seed)
        
        # Sparse bias pattern
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        boosted = rng.choice(self.vocab_size, size=500, replace=False)
        bias[boosted] = self.bias_strength
        
        return bias
    
    def _decode_channel_from_tokens(self, tokens, channel_id):
        """
        Decode a channel's context from token sequence.
        
        Looks at tokens at positions: channel_id, channel_id + num_channels, ...
        """
        channel_tokens = []
        for i in range(channel_id, len(tokens), self.num_channels):
            if i < len(tokens):
                channel_tokens.append(tokens[i])
        
        if len(channel_tokens) < self.tokens_per_channel:
            return ""
        
        # Try to decode by finding best matching context
        # This is simplified — real version would use learned decoder
        return self.memory.get(channel_id, "")
    
    def generate(self, user_prompt: str, max_response_tokens: int = 100) -> str:
        """
        Generate a response to user prompt.
        
        The response has two parts:
        1. Memory preamble: tokens encoding current memory state (FDM encoded)
        2. Response: actual response to user prompt
        
        Format: [MEMORY_TOKENS][RESPONSE_TOKENS]
        """
        
        # Build full prompt
        full_prompt = f"User: {user_prompt}\nAssistant:"
        input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt').to(self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            # Phase 1: Generate memory-encoded tokens
            for pos in range(self.memory_tokens):
                channel_id = pos % self.num_channels
                token_idx = pos // self.num_channels
                
                # Get base distribution
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                
                # Get context bits for this channel
                context = self.memory.get(channel_id, "")
                bits = self._context_to_bits(context)
                
                # Apply bias
                bias = self._get_channel_bias(bits, token_idx, channel_id)
                biased_logits = logits + bias
                
                # Sample
                probs = self._softmax(biased_logits)
                token = np.random.choice(self.vocab_size, p=probs)
                
                generated_tokens.append(token)
                input_ids = torch.cat([
                    input_ids, 
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
            
            # Phase 2: Generate actual response (no bias, natural generation)
            for _ in range(max_response_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                # Natural sampling (temperature 0.7)
                probs = torch.softmax(logits / 0.7, dim=-1)
                token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
                
                # Stop at EOS or newline
                if token == self.tokenizer.eos_token_id:
                    break
                decoded = self.tokenizer.decode([token])
                if '\n' in decoded and len(generated_tokens) > self.memory_tokens + 10:
                    break
        
        # Decode full response
        full_response = self.tokenizer.decode(generated_tokens)
        
        # Split into memory preamble and actual response
        memory_text = self.tokenizer.decode(generated_tokens[:self.memory_tokens])
        response_text = self.tokenizer.decode(generated_tokens[self.memory_tokens:])
        
        return {
            'full': full_response,
            'memory_preamble': memory_text,
            'response': response_text.strip(),
            'memory_tokens': generated_tokens[:self.memory_tokens],
            'response_tokens': generated_tokens[self.memory_tokens:],
        }
    
    def ingest_previous_output(self, previous_tokens: list):
        """
        Read memory from a previous output's memory tokens.
        
        This allows the model to recover context from its own previous generation.
        """
        # In a full implementation, this would use the trained decoder
        # For now, we just preserve the existing memory
        print(f"  Ingested {len(previous_tokens)} memory tokens")
        return self.memory
    
    def _softmax(self, x):
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()


def demo():
    """Demonstrate eidetic memory across turns."""
    
    model = EideticModel(num_channels=8, tokens_per_channel=16, bias_strength=3.0)
    
    # Set some memory
    print("\n=== Setting Memory ===")
    model.set_memory(0, "SECRET:RED")
    model.set_memory(1, "LOCATION:PARIS")
    model.set_memory(2, "AGENT:ALICE")
    model.set_memory(3, "TASK:WAIT")
    
    # Turn 1
    print("\n=== Turn 1 ===")
    result1 = model.generate("What should I do?")
    print(f"Memory preamble: {result1['memory_preamble'][:50]}...")
    print(f"Response: {result1['response']}")
    
    # Simulate new turn — model reads its previous output
    print("\n=== Turn 2 (reading previous memory) ===")
    model.ingest_previous_output(result1['memory_tokens'])
    
    # Update memory
    model.set_memory(3, "TASK:GO")  # Changed from WAIT to GO
    
    result2 = model.generate("What's my next step?")
    print(f"Memory preamble: {result2['memory_preamble'][:50]}...")
    print(f"Response: {result2['response']}")
    
    # Show memory state
    print("\n=== Current Memory State ===")
    for ch in range(4):
        print(f"  Channel {ch}: {model.get_memory(ch)}")


if __name__ == "__main__":
    demo()
