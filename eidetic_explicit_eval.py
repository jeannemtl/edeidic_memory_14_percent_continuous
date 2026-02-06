"""
Evaluate explicit memory model.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = GPT2Tokenizer.from_pretrained('eidetic_explicit_v1')
    model = GPT2LMHeadModel.from_pretrained('eidetic_explicit_v1')
    model.to(device)
    model.eval()
    
    test_samples = [json.loads(l) for l in open('explicit_memory_training_test.jsonl')]
    
    correct = 0
    total = 0
    
    print("Evaluating explicit memory model...\n")
    
    for i, sample in enumerate(test_samples[:100]):
        input_text = sample['input']
        expected_value = sample['memory'][str(sample['asked_channel'])]
        
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(output[0])
        response = generated[len(input_text):].strip()
        
        is_correct = expected_value in response
        if is_correct:
            correct += 1
        total += 1
        
        if i < 10:
            print(f"Q: {sample['question']}")
            print(f"Expected: {expected_value}")
            print(f"Got: {response[:50]}")
            print(f"{'✓' if is_correct else '✗'}\n")
    
    print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

if __name__ == "__main__":
    evaluate()
