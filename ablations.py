"""
Systematic ablations to understand system behavior
"""

def ablation_epsilon():
    """How does imperfection level affect performance?"""
    results = {}
    
    for epsilon in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        print(f"\n{'='*40}")
        print(f"EPSILON = {epsilon}")
        print(f"{'='*40}")
        
        # Generate training data with this epsilon
        generate_training_data(
            num_samples=20000,
            epsilon=epsilon,
            output_path=f"data/epsilon_{epsilon}.jsonl"
        )
        
        # Train decoder
        decoder = EideticDecoder()
        decoder.train(
            train_path=f"data/epsilon_{epsilon}_train.jsonl",
            val_path=f"data/epsilon_{epsilon}_val.jsonl",
            epochs=5
        )
        
        # Evaluate
        encoder = EideticEncoder(epsilon=epsilon)
        evaluator = EideticEvaluator(decoder, encoder, f"data/epsilon_{epsilon}_test.jsonl")
        
        results[epsilon] = evaluator.full_evaluation()
    
    # Plot epsilon vs accuracy
    plot_epsilon_ablation(results)
    
    return results


def ablation_num_channels():
    """How does channel count affect performance?"""
    results = {}
    
    for num_ch in [2, 4, 8, 16, 32]:
        print(f"\n{'='*40}")
        print(f"NUM_CHANNELS = {num_ch}")
        print(f"{'='*40}")
        
        # Generate and train
        generate_training_data(num_samples=20000, num_channels=num_ch)
        
        decoder = EideticDecoder()
        decoder.train(...)
        
        encoder = EideticEncoder(num_channels=num_ch)
        evaluator = EideticEvaluator(decoder, encoder, ...)
        
        results[num_ch] = evaluator.full_evaluation()
    
    return results


def ablation_model_size():
    """How does decoder model size affect performance?"""
    results = {}
    
    for model_name in ["gpt2", "gpt2-medium", "gpt2-large"]:
        print(f"\n{'='*40}")
        print(f"MODEL = {model_name}")
        print(f"{'='*40}")
        
        decoder = EideticDecoder(model_name=model_name)
        decoder.train(...)
        
        results[model_name] = evaluator.full_evaluation()
    
    return results
