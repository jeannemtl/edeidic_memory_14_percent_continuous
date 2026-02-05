#!/usr/bin/env python3
"""
run_pipeline.py — Full eidetic memory pipeline

Run on RunPod:
  tmux new -s eidetic
  python run_pipeline.py
  
Ctrl+b, d to detach. tmux attach -t eidetic to reattach.

Steps:
  1. Generate training data (leaky encoder)
  2. Train decoder (classification head on GPT-2)
  3. Evaluate
"""

import sys
import os
import time


def step1_generate_data():
    """Generate training data with leaky encoder."""
    print("\n" + "="*60)
    print("STEP 1: GENERATE TRAINING DATA")
    print("="*60)
    
    from training_data_generation_v2 import generate_training_data
    
    generate_training_data(
        num_stegotexts=500,       # Start small: 500 × 8 = 4000 samples
        num_channels=8,
        tokens_per_channel=32,
        bias_strength=3.0,        # Strong bias for initial experiment
        checkpoint_interval=25,
    )


def step2_train():
    """Train the decoder."""
    print("\n" + "="*60)
    print("STEP 2: TRAIN DECODER")
    print("="*60)
    
    from decoder_training_v2 import train
    
    train(
        train_path="eidetic_training_data_train.jsonl",
        val_path="eidetic_training_data_val.jsonl",
        output_dir="eidetic_decoder_v2",
        num_epochs=20,
        batch_size=32,
        lr=1e-3,
        tokens_per_channel=32,
        freeze_backbone=True,
    )


def step3_evaluate():
    """Evaluate on test set."""
    print("\n" + "="*60)
    print("STEP 3: EVALUATE")
    print("="*60)
    
    from simple_eval_v2 import evaluate
    
    acc, predictions = evaluate(
        model_dir="eidetic_decoder_v2",
        test_path="eidetic_training_data_test.jsonl",
        tokens_per_channel=32,
    )
    
    return acc


if __name__ == "__main__":
    start = time.time()
    
    print("="*60)
    print("EIDETIC MEMORY v2 — Full Pipeline")
    print("="*60)
    print(f"\nThis fixes the 5.5% accuracy by:")
    print(f"  1. Using 'leaky' encoder with learnable statistical biases")
    print(f"  2. Classification (not generation) — predict context class")
    print(f"  3. Balanced loss + label smoothing (prevents mode collapse)")
    print(f"  4. Channel-segmented input (model sees one channel at a time)")
    
    # Run pipeline
    step1_generate_data()
    step2_train()
    acc = step3_evaluate()
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Final accuracy: {acc*100:.1f}%")
    print(f"{'='*60}")
    
    if acc < 0.20:
        print(f"\n⚠️  Still low accuracy. Try:")
        print(f"  - Increase bias_strength to 5.0")
        print(f"  - Generate more data (num_stegotexts=2000)")
        print(f"  - Unfreeze backbone (freeze_backbone=False, lr=1e-5)")
    elif acc < 0.50:
        print(f"\nLearning signal detected! To improve:")
        print(f"  - More data (num_stegotexts=2000)")
        print(f"  - More epochs")
        print(f"  - Try unfreezing last 2 layers of GPT-2")
    elif acc > 0.80:
        print(f"\n✓ Strong performance! Now try:")
        print(f"  - Reduce bias_strength to 2.0 then 1.0")
        print(f"  - This tests how subtle the fingerprint can be")
        print(f"  - Toward the 'artificial eidetic memory' goal")
