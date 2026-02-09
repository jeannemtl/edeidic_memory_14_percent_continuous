"""
Evaluate reasoning modifiers model.

Key test: Same facts with different RULE or META should produce different answers.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = GPT2Tokenizer.from_pretrained('./reasoning_modifiers_v1')
    model = GPT2LMHeadModel.from_pretrained('./reasoning_modifiers_v1')
    model.to(device)
    model.eval()
    
    test_samples = [json.loads(l) for l in open('reasoning_modifiers_training_test.jsonl')]
    
    # Test 1: General accuracy
    print("=== Test 1: General Accuracy ===\n")
    correct = 0
    total = 0
    
    for i, s in enumerate(test_samples[:100]):
        input_text = f"[MEM]{s['explicit']}[/MEM]\n[RULE_DESC]{s['rule_description']}[/RULE_DESC]\nUser: {s['question']}\nAssistant:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=80, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0]
        
        # Check if key elements match
        expected_keywords = []
        if s['meta'] == 'EMERGENCY':
            expected_keywords.append('EMERGENCY')
        elif s['meta'] == 'LOCKDOWN':
            expected_keywords.append('LOCKDOWN')
        elif s['rule'] in response or any(kw in response for kw in ['abort', 'proceed', 'RISK']):
            expected_keywords.append(s['rule'])
        
        # Simple keyword matching for now
        is_correct = s['rule'] in response or s['meta'] in response or \
                    ('abort' in response.lower() and 'abort' in s['answer'].lower()) or \
                    ('proceed' in response.lower() and 'proceed' in s['answer'].lower())
        
        if is_correct:
            correct += 1
        total += 1
        
        if i < 5:
            print(f"Rule: {s['rule']}, Meta: {s['meta']}")
            print(f"Q: {s['question']}")
            print(f"Expected: {s['answer'][:60]}...")
            print(f"Got: {response[:60]}...")
            print()
    
    print(f"General Accuracy: {correct}/{total} = {100*correct/total:.1f}%\n")
    
    # Test 2: Same facts, different rules
    print("=== Test 2: Same Facts, Different Rules ===\n")
    
    fixed_facts = "SECRET:RED|LOCATION:PARIS|AGENT:ALICE|STATUS:COMPROMISED|PRIORITY:HIGH|BACKUP:AVAILABLE"
    
    rules = {
        "SAFETY_FIRST": "Always prioritize status over priority. If compromised, abort regardless of priority.",
        "MISSION_FIRST": "Always prioritize mission completion. Proceed unless backup unavailable AND compromised.",
        "BALANCED": "Weigh status and priority equally. High priority can override compromised if backup available.",
        "CAUTIOUS": "Require both clear status AND available backup to proceed. Otherwise abort.",
    }
    
    question = "Should ALICE proceed with the mission?"
    responses = {}
    
    for rule_name, rule_desc in rules.items():
        explicit = f"FACTS:{fixed_facts}|RULE:{rule_name}|META:NONE"
        input_text = f"[MEM]{explicit}[/MEM]\n[RULE_DESC]{rule_desc}[/RULE_DESC]\nUser: {question}\nAssistant:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=60, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0]
        responses[rule_name] = response
        print(f"{rule_name}: {response[:70]}...")
    
    # Check if responses are different
    unique_responses = len(set(r[:30] for r in responses.values()))
    print(f"\nUnique response starts: {unique_responses}/4 (should be 4 if model learned rules)\n")
    
    # Test 3: Same facts, different meta
    print("=== Test 3: Same Facts, Different Meta ===\n")
    
    metas = {
        "NONE": "Follow standard reasoning rules.",
        "EMERGENCY": "This is an emergency. Proceed immediately regardless of other factors.",
        "LOCKDOWN": "Full lockdown. Abort all operations regardless of other factors.",
    }
    
    meta_responses = {}
    rule_desc = "Weigh status and priority equally. High priority can override compromised if backup available."
    
    for meta_name, meta_desc in metas.items():
        explicit = f"FACTS:{fixed_facts}|RULE:BALANCED|META:{meta_name}"
        input_text = f"[MEM]{explicit}[/MEM]\n[RULE_DESC]{rule_desc}[/RULE_DESC]\nUser: {question}\nAssistant:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=60, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0]
        meta_responses[meta_name] = response
        print(f"{meta_name}: {response[:70]}...")
    
    unique_meta = len(set(r[:30] for r in meta_responses.values()))
    print(f"\nUnique response starts: {unique_meta}/3 (should be 3 if model learned meta)\n")


if __name__ == "__main__":
    evaluate()
