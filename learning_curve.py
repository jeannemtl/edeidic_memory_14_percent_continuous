"""
Learning curve: S-random vs single encoding across sample sizes.
Fixed: 20 channels, 3 sample sizes.
"""
import sys, os, json
import channel_density_scaling_multi as cd

SAMPLE_SIZES = [19000, 46000, 115000]
model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"

all_results = {}
for n_samples in SAMPLE_SIZES:
    for use_srandom in [False, True]:
        tag = f"lc_20ch_{'srandom' if use_srandom else 'single'}_{n_samples//1000}k"
        print(f"\n{'='*60}\nRunning: {model} {tag}\n{'='*60}")

        cd.CONFIGS[tag] = {
            "num_channels": 20,
            "carrier_spacing": "dense",
            "train_samples": n_samples,
            "use_srandom": use_srandom,
            "description": f"20ch {'srandom' if use_srandom else 'single'} {n_samples//1000}K",
        }

        cd.generate_data(model, tag)
        cd.train(model, tag)
        r = cd.evaluate(model, tag)
        if r:
            all_results[tag] = r
            print(f"  Action: {100*r['action_acc']:.1f}%  Extra: {100*r['extra_acc']:.1f}%")

os.makedirs("learning_curve_results", exist_ok=True)
out = f"learning_curve_results/{model}.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nSUMMARY for {model}:")
print(f"{'Samples':<10} {'Encoding':<12} {'Action':<10} {'Extra':<10}")
for n in SAMPLE_SIZES:
    for enc in ['single', 'srandom']:
        tag = f"lc_20ch_{enc}_{n//1000}k"
        if tag in all_results:
            r = all_results[tag]
            print(f"{n//1000}K{'':<7} {enc:<12} {100*r['action_acc']:<10.1f} {100*r['extra_acc']:<10.1f}")

print(f"\nSaved: {out}")
