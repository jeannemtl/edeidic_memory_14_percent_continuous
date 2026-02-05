from FDM_multiplexer import FDMMultiplexer
from imec_encoder import ImperfectiMEC
"""
Complete Eidetic Encoder: Contexts → FDM → iMEC → Stegotext
"""

class EideticEncoder:
    def __init__(
        self,
        num_channels: int = 8,
        epsilon: float = 0.15,
        bits_per_channel: int = 64,  # 32 bytes per channel
        device: str = "cuda"
    ):
        self.num_channels = num_channels
        self.bits_per_channel = bits_per_channel
        
        self.fdm = FDMMultiplexer(num_channels=num_channels)
        self.imec = ImperfectiMEC(epsilon=epsilon, device=device)
    
    def encode(
        self,
        contexts: list,
        cover_prompt: str = "In today's news"
    ) -> tuple:
        """
        Encode multiple contexts into single stegotext
        
        Args:
            contexts: List of context strings (one per channel)
            cover_prompt: Starting prompt for stegotext
        
        Returns:
            stegotext: Readable text with hidden channels
            metadata: Encoding parameters for decoding
        """
        # Pad/truncate contexts to fixed size
        max_chars = self.bits_per_channel // 8
        padded_contexts = []
        for ctx in contexts:
            if len(ctx) > max_chars:
                ctx = ctx[:max_chars]
            else:
                ctx = ctx.ljust(max_chars)
            padded_contexts.append(ctx)
        
        # Pad to full channel count
        while len(padded_contexts) < self.num_channels:
            padded_contexts.append(" " * max_chars)
        
        # FDM encode
        fdm_signal = self.fdm.encode(padded_contexts)
        
        # Quantize signal to bits
        # Normalize and threshold
        fdm_normalized = (fdm_signal - fdm_signal.min()) / (fdm_signal.max() - fdm_signal.min())
        fdm_bits = [1 if s > 0.5 else 0 for s in fdm_normalized]
        
        # iMEC encode
        stegotext = self.imec.encode_bits(fdm_bits, prompt=cover_prompt)
        
        metadata = {
            "num_channels": self.num_channels,
            "bits_per_channel": self.bits_per_channel,
            "total_bits": len(fdm_bits),
            "prompt": cover_prompt,
            "fdm_signal_stats": {
                "min": float(fdm_signal.min()),
                "max": float(fdm_signal.max()),
            }
        }
        
        return stegotext, metadata
    
    def decode_traditional(
        self,
        stegotext: str,
        metadata: dict
    ) -> list:
        """Traditional decode (for verification only)"""
        # iMEC decode
        bits = self.imec.decode_bits(
            stegotext, 
            metadata["total_bits"],
            metadata["prompt"]
        )
        
        # Reconstruct signal
        signal = np.array([1.0 if b else 0.0 for b in bits])
        signal = signal * (metadata["fdm_signal_stats"]["max"] - metadata["fdm_signal_stats"]["min"])
        signal = signal + metadata["fdm_signal_stats"]["min"]
        
        # FDM decode each channel
        contexts = []
        bits_per_ch = metadata["bits_per_channel"]
        for ch in range(metadata["num_channels"]):
            ch_bits = self.fdm.decode_channel(signal, ch, bits_per_ch)
            ch_text = self.fdm.bits_to_text(ch_bits)
            contexts.append(ch_text.strip())
        
        return contexts


# Test full encoder
def test_eidetic_encoder():
    encoder = EideticEncoder(num_channels=4, epsilon=0.2)
    
    contexts = [
        "SECRET:ALPHA",
        "LOCATION:PARIS",
        "PRIORITY:HIGH",
        "AGENT:BOB"
    ]
    
    print("Encoding...")
    stegotext, metadata = encoder.encode(contexts)
    
    print(f"\nStegotext ({len(stegotext)} chars):")
    print(stegotext[:300] + "...")
    
    print(f"\nMetadata: {metadata}")
    
    print("\nTraditional decode (verification):")
    decoded = encoder.decode_traditional(stegotext, metadata)
    for i, (orig, dec) in enumerate(zip(contexts, decoded)):
        print(f"  Ch {i}: '{orig}' -> '{dec}'")
    
    return stegotext, metadata, contexts

#test_eidetic_encoder()
