from tqdm import tqdm
from full_encoder_pipeline import EideticEncoder
from decoder_training import EideticDecoder
from decoder_training import EideticDecoder
"""
Comprehensive evaluation of eidetic memory system
"""

import json
from collections import defaultdict
from typing import List, Dict
import numpy as np

class EideticEvaluator:
    def __init__(
        self,
        decoder: EideticDecoder,
        encoder: EideticEncoder,
        test_path: str
    ):
        self.decoder = decoder
        self.encoder = encoder
        
        # Load test data
        self.test_samples = []
        with open(test_path) as f:
            for line in f:
                self.test_samples.append(json.loads(line))
    
    def evaluate_accuracy(self) -> Dict:
        """Measure decode accuracy per channel"""
        results = defaultdict(list)
        
        for sample in tqdm(self.test_samples, desc="Evaluating"):
            stegotext = sample["stegotext"]
            channel = sample["channel"]
            true_context = sample["context"]
            
            # Decode
            decoded = self.decoder.decode(stegotext, channel)
            
            # Exact match
            exact_match = decoded.strip() == true_context.strip()
            
            # Partial match (contains key value)
            key_value = true_context.split(":")[-1] if ":" in true_context else true_context
            partial_match = key_value in decoded
            
            results[f"channel_{channel}_exact"].append(exact_match)
            results[f"channel_{channel}_partial"].append(partial_match)
            results["overall_exact"].append(exact_match)
            results["overall_partial"].append(partial_match)
        
        # Aggregate
        metrics = {}
        for key, values in results.items():
            metrics[key] = np.mean(values)
        
        return metrics
    
    def evaluate_channel_isolation(self) -> Dict:
        """Test that channels don't interfere with each other"""
        # Generate fresh test cases
        isolation_results = []
        
        for _ in tqdm(range(100), desc="Channel isolation"):
            # Create contexts where each channel has unique identifier
            contexts = [f"CH{i}:UNIQUE{i}" for i in range(8)]
            
            stegotext, metadata = self.encoder.encode(contexts)
            
            # Decode each channel
            for ch in range(8):
                decoded = self.decoder.decode(stegotext, ch)
                
                # Check if correct channel content is present
                correct_marker = f"UNIQUE{ch}"
                has_correct = correct_marker in decoded
                
                # Check if OTHER channel content leaked
                leaked = False
                for other_ch in range(8):
                    if other_ch != ch:
                        other_marker = f"UNIQUE{other_ch}"
                        if other_marker in decoded:
                            leaked = True
                            break
                
                isolation_results.append({
                    "channel": ch,
                    "correct": has_correct,
                    "leaked": leaked,
                    "isolated": has_correct and not leaked
                })
        
        # Aggregate
        metrics = {
            "correct_rate": np.mean([r["correct"] for r in isolation_results]),
            "leak_rate": np.mean([r["leaked"] for r in isolation_results]),
            "isolation_rate": np.mean([r["isolated"] for r in isolation_results]),
        }
        
        return metrics
    
    def evaluate_capacity(self) -> Dict:
        """Measure actual bits per token achieved"""
        # Count successful bit transmissions
        total_bits_sent = 0
        total_bits_received = 0
        total_tokens = 0
        
        for _ in tqdm(range(100), desc="Capacity test"):
            # Random contexts
            contexts = [f"TEST:{i:08b}" for i in range(8)]  # 8 bits each
            
            stegotext, metadata = self.encoder.encode(contexts)
            total_tokens += len(stegotext.split())  # Rough token count
            total_bits_sent += 8 * 8  # 8 channels × 8 bits
            
            # Decode and count correct bits
            for ch in range(8):
                decoded = self.decoder.decode(stegotext, ch)
                expected = f"TEST:{ch:08b}"
                
                # Count matching bits
                if "TEST:" in decoded:
                    try:
                        decoded_bits = decoded.split(":")[-1][:8]
                        expected_bits = f"{ch:08b}"
                        for d, e in zip(decoded_bits, expected_bits):
                            if d == e:
                                total_bits_received += 1
                    except:
                        pass
        
        metrics = {
            "bits_sent": total_bits_sent,
            "bits_received": total_bits_received,
            "bit_accuracy": total_bits_received / total_bits_sent,
            "tokens_used": total_tokens,
            "effective_bits_per_token": total_bits_received / total_tokens
        }
        
        return metrics
    
    def evaluate_generalization(self) -> Dict:
        """Test on contexts not seen during training"""
        novel_contexts = [
            "MISSION:IMPOSSIBLE",
            "TARGET:UNKNOWN",
            "PRIORITY:CRITICAL",
            "DEADLINE:URGENT",
            "SOURCE:CLASSIFIED",
            "LEVEL:MAXIMUM",
            "ZONE:RESTRICTED",
            "PHASE:FINAL"
        ]
        
        results = []
        
        for _ in range(50):
            # Shuffle novel contexts into channels
            shuffled = novel_contexts.copy()
            random.shuffle(shuffled)
            
            stegotext, _ = self.encoder.encode(shuffled)
            
            for ch, true_ctx in enumerate(shuffled):
                decoded = self.decoder.decode(stegotext, ch)
                key_value = true_ctx.split(":")[-1]
                
                results.append({
                    "exact": decoded.strip() == true_ctx,
                    "partial": key_value in decoded
                })
        
        return {
            "novel_exact_accuracy": np.mean([r["exact"] for r in results]),
            "novel_partial_accuracy": np.mean([r["partial"] for r in results])
        }
    
    def full_evaluation(self) -> Dict:
        """Run all evaluations"""
        print("=" * 60)
        print("EIDETIC MEMORY EVALUATION")
        print("=" * 60)
        
        print("\n1. Decode Accuracy:")
        accuracy = self.evaluate_accuracy()
        for k, v in sorted(accuracy.items()):
            print(f"   {k}: {v:.1%}")
        
        print("\n2. Channel Isolation:")
        isolation = self.evaluate_channel_isolation()
        for k, v in isolation.items():
            print(f"   {k}: {v:.1%}")
        
        print("\n3. Capacity:")
        capacity = self.evaluate_capacity()
        for k, v in capacity.items():
            if "accuracy" in k or "per_token" in k:
                print(f"   {k}: {v:.3f}")
            else:
                print(f"   {k}: {v}")
        
        print("\n4. Generalization:")
        generalization = self.evaluate_generalization()
        for k, v in generalization.items():
            print(f"   {k}: {v:.1%}")
        
        return {
            "accuracy": accuracy,
            "isolation": isolation,
            "capacity": capacity,
            "generalization": generalization
        }


# Run evaluation
def run_evaluation():
    # Load trained decoder
    decoder = EideticDecoder(model_name="eidetic_decoder_v1")
    
    # Load encoder
    encoder = EideticEncoder(num_channels=8, epsilon=0.15)
    
    # Evaluate
    evaluator = EideticEvaluator(
        decoder=decoder,
        encoder=encoder,
        test_path="eidetic_training_data_test.jsonl"
    )
    
    results = evaluator.full_evaluation()
    
    # Save results
    with open("eidetic_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

results = run_evaluation()
