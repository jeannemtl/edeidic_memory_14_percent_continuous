"""
simple_eval_v2.py — Evaluate the trained eidetic decoder

Loads trained model, runs on test set, reports per-channel and
per-category accuracy with confusion analysis.
"""

import json
import torch
from transformers import GPT2Tokenizer
from safetensors.torch import load_file
from decoder_training_v2 import (
    EideticDecoderModel, EideticDataset,
    CONTEXT_TO_IDX, IDX_TO_CONTEXT, NUM_CLASSES
)
from collections import defaultdict


def evaluate(
    model_dir="eidetic_decoder_v2",
    test_path="eidetic_training_data_test.jsonl",
    tokens_per_channel=32,
):
    """Run evaluation on test set."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {model_dir}/...")
    model = EideticDecoderModel(
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
    ).to(device)
    
    # Load trained weights (head only)
    weights = load_file(f"{model_dir}/best_model.safetensors")
    model.load_state_dict(weights, strict=False)
    model.eval()
    
    # Load test data
    test_ds = EideticDataset(test_path, tokens_per_channel=tokens_per_channel)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False
    )
    
    # Evaluate
    correct = 0
    total = 0
    per_channel_correct = defaultdict(int)
    per_channel_total = defaultdict(int)
    per_category_correct = defaultdict(int)
    per_category_total = defaultdict(int)
    
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            channel_idx = batch['channel_idx'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attn_mask, channel_idx)
            preds = logits.argmax(dim=-1)
            
            for i in range(labels.size(0)):
                ch = channel_idx[i].item()
                pred = preds[i].item()
                label = labels[i].item()
                is_correct = pred == label
                
                correct += int(is_correct)
                total += 1
                
                per_channel_correct[ch] += int(is_correct)
                per_channel_total[ch] += 1
                
                # Category (extract from context string)
                true_ctx = IDX_TO_CONTEXT[label]
                pred_ctx = IDX_TO_CONTEXT[pred]
                category = true_ctx.split(":")[0]
                
                per_category_correct[category] += int(is_correct)
                per_category_total[category] += 1
                
                predictions.append({
                    'channel': ch,
                    'true': true_ctx,
                    'pred': pred_ctx,
                    'correct': is_correct,
                })
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    
    overall_acc = correct / total if total > 0 else 0
    print(f"\nOverall accuracy: {correct}/{total} = {overall_acc*100:.1f}%")
    
    print(f"\nPer-channel accuracy:")
    for ch in sorted(per_channel_total.keys()):
        ch_acc = per_channel_correct[ch] / per_channel_total[ch]
        print(f"  Ch {ch}: {per_channel_correct[ch]}/{per_channel_total[ch]} "
              f"= {ch_acc*100:.1f}%")
    
    print(f"\nPer-category accuracy:")
    for cat in sorted(per_category_total.keys()):
        cat_acc = per_category_correct[cat] / per_category_total[cat]
        print(f"  {cat:12s}: {per_category_correct[cat]}/{per_category_total[cat]} "
              f"= {cat_acc*100:.1f}%")
    
    # Show some examples
    print(f"\nSample predictions (first 20):")
    for p in predictions[:20]:
        mark = "✓" if p['correct'] else "✗"
        print(f"  Ch {p['channel']}: {p['true']:25s} → {p['pred']:25s} {mark}")
    
    # Error analysis
    if overall_acc < 1.0:
        print(f"\nMost common errors:")
        error_counts = defaultdict(int)
        for p in predictions:
            if not p['correct']:
                error_counts[(p['true'], p['pred'])] += 1
        
        top_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:10]
        for (true, pred), count in top_errors:
            print(f"  {true} → {pred} ({count} times)")
    
    # Check for mode collapse (all predicting same class)
    pred_distribution = defaultdict(int)
    for p in predictions:
        pred_distribution[p['pred']] += 1
    
    most_common_pred = max(pred_distribution.items(), key=lambda x: x[1])
    if most_common_pred[1] / total > 0.5:
        print(f"\n⚠️  WARNING: Mode collapse detected!")
        print(f"  '{most_common_pred[0]}' predicted {most_common_pred[1]}/{total} times "
              f"({most_common_pred[1]/total*100:.1f}%)")
        print(f"  Top 5 predicted classes:")
        for ctx, count in sorted(pred_distribution.items(), key=lambda x: -x[1])[:5]:
            print(f"    {ctx}: {count} ({count/total*100:.1f}%)")
    else:
        unique_preds = len(pred_distribution)
        print(f"\n✓ No mode collapse: {unique_preds} unique predictions")
    
    return overall_acc, predictions


if __name__ == "__main__":
    evaluate()
