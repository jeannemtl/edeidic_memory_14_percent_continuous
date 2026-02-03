
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import json

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

tokenizer.add_special_tokens({'additional_special_tokens': ['[S]', '[C]', '[D]']})

tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('eidetic_decoder_v1')

model.resize_token_embeddings(len(tokenizer))

model.cuda().eval()

samples = [json.loads(l) for l in open('eidetic_training_data_test.jsonl')]

correct = 0

total = 0

for s in samples:

    prompt = f"[S]{s['stegotext'][:400]}[C]{s['channel']}[D]"

    ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

    

    with torch.no_grad():

        out = model.generate(ids, max_new_tokens=30, do_sample=False,

                            pad_token_id=tokenizer.eos_token_id)

    

    text = tokenizer.decode(out[0])

    pred = text.split('[D]')[-1].strip() if '[D]' in text else ''

    

    expected = s['context']

    key_value = expected.split(':')[-1] if ':' in expected else expected

    

    match = key_value in pred

    if match:

        correct += 1

    total += 1

    

    print(f"Ch {s['channel']}: {expected} -> {pred[:30]} {'✓' if match else '✗'}")

print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

