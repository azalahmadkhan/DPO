from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# GPT-2 log probablilties 

def get_token_logprobs(model, tokenizer, text, device='cuda'):
    """
    Get log probabilities for each token in the generated text.
    
    Args:
        model: HuggingFace language model
        tokenizer: HuggingFace tokenizer
        text: Input text to analyze
        device: Device to run the model on ('cuda' or 'cpu')
    
    Returns:
        tokens: List of tokens
        logprobs: List of log probabilities for each token
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and convert to probabilities
    logits = outputs.logits[0, :-1]  # Remove last logit
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the token ids from the input
    token_ids = inputs.input_ids[0]
    
    # Calculate log probabilities for the actual next tokens
    next_token_ids = token_ids[1:]  # Shift right to get next tokens
    next_token_probs = probs[range(len(probs)), next_token_ids]
    logprobs = torch.log(next_token_probs).cpu().numpy()
    
    # Get the actual tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(token_ids[:-1])
    
    return tokens, logprobs

# Example usage
def demonstrate_logprobs(text):
    """
    Demonstrate getting log probabilities for a piece of text.
    """
    # Load model and tokenizer
    model_name = "gpt2"  # Can use any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Get token probabilities
    tokens, logprobs = get_token_logprobs(model, tokenizer, text, device)
    
    # Print results
    print("\nToken-by-token analysis:")
    print("-" * 50)
    for token, logprob in zip(tokens, logprobs):
        print(f"Token: {token:15} Log Probability: {logprob:.4f}")
    
    # Print some statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Average log probability: {np.mean(logprobs):.4f}")
    print(f"Min log probability: {np.min(logprobs):.4f}")
    print(f"Max log probability: {np.max(logprobs):.4f}")

def generate_with_logprobs(model, tokenizer, prompt, max_new_tokens=50, device='cuda'):
    """
    Generate text and get log probabilities for the generated tokens.
    
    Args:
        model: HuggingFace language model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt to start generation
        max_new_tokens: Maximum number of new tokens to generate
        device: Device to run the model on ('cuda' or 'cpu')
    
    Returns:
        generated_text: The complete generated text
        tokens: List of generated tokens
        logprobs: List of log probabilities for each generated token
    """
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_tokens = []
    token_logprobs = []
    
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get logits for the last token
        logits = outputs.logits[0, -1]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Get log probability of chosen token
        next_token_prob = probs[next_token]
        logprob = torch.log(next_token_prob).cpu().numpy()[0]
        
        # Store token and its log probability
        generated_tokens.append(next_token.item())
        token_logprobs.append(logprob)
        
        # Update input_ids with new token
        inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=1)
        
        # Stop if we generate an EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Convert tokens to text
    generated_text = tokenizer.decode(inputs.input_ids[0])
    tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
    
    return generated_text, tokens, token_logprobs

# Example usage for generation
def demonstrate_generation_logprobs(prompt):
    """
    Demonstrate getting log probabilities during text generation.
    """
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Generate text and get probabilities
    generated_text, tokens, logprobs = generate_with_logprobs(model, tokenizer, prompt)
    
    # Print results
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    
    print("\nToken-by-token analysis:")
    print("-" * 50)
    for token, logprob in zip(tokens, logprobs):
        print(f"Token: {token:15} Log Probability: {logprob:.4f}")
    
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Average log probability: {np.mean(logprobs):.4f}")
    print(f"Min log probability: {np.min(logprobs):.4f}")
    print(f"Max log probability: {np.max(logprobs):.4f}")

prompt = "Once upon a time"
demonstrate_generation_logprobs(prompt)