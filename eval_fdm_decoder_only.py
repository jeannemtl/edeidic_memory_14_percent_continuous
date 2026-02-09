"""Test just the FDM decoder (Stage 1)"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

device = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained('./fdm_decoder_v1')
model = GPT2LMHeadModel.from_pretrained('./fdm_decoder_v1')
model.to(device)
model.eval()

test_samples = [json.loads(l) for l in open('fdm_decoder_training_test.jsonl')]

correct_channels = 0
total_channels = 0

print("Evaluating FDM decoder (Stage 1 only)...\n")

for i, sample in enumerate(test_samples[:30]):
    fdm_text = sample['fdm_text']
    expected = sample['explicit']
    memory = sample['memory']
    
    input_text = f"[FDM]{fdm_text}[/FDM][DECODE]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=80, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    
    decoded = tokenizer.decode(output[0])[len(input_text):].strip()
    decoded = decoded.split('<|endoftext|>')[0]
    
    # Count correct channels
    sample_correct = 0
    for ch, val in memory.items():
        total_channels += 1
        if val in decoded:
            correct_channels += 1
            sample_correct += 1
    
    if i < 10:
        print(f"Expected: {expected[:50]}...")
        print(f"Decoded:  {decoded[:50]}...")
        print(f"Channels correct: {sample_correct}/8")
        print()

print(f"\nOverall Channel Accuracy: {correct_channels}/{total_channels} = {100*correct_channels/total_channels:.1f}%")
