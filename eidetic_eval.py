"""
Evaluate the trained eidetic model.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    tokenizer = GPT2Tokenizer.from_pretrained('eidetic_model_v1')
    model = GPT2LMHeadModel.from_pretrained('eidetic_model_v1')
    model.to(device)
    model.eval()
    
    # Load test data
    test_samples = [json.loads(l) for l in open('eidetic_memory_training_test.jsonl')]
    
    correct = 0
    total = 0
    
    print("Evaluating on test set...\n")
    
    for i, sample in enumerate(test_samples[:50]):  # First 50 samples
        input_text = sample['input']
        expected = sample['answer']
        memory = sample['memory']
        question = sample['question']
        channel = sample['asked_channel']
        
        # Generate
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=False)
        response = generated[len(input_text):].strip()
        
        # Check if answer contains the correct value
        expected_value = memory[str(channel)]
        is_correct = expected_value in response
        
        if is_correct:
            correct += 1
        total += 1
        
        # Print some examples
        if i < 10:
            print(f"Q: {question}")
            print(f"Memory[{channel}]: {expected_value}")
            print(f"Expected: {expected}")
            print(f"Got: {response[:50]}...")
            print(f"{'✓' if is_correct else '✗'}")
            print()
    
    print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

if __name__ == "__main__":
    evaluate()
