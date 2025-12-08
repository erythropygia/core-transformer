"""
Usage:
    python generate_text.py -p "Your prompt here"
    python generate_text.py --interactive
    python generate_text.py -p "Prompt" --model-path checkpoints/best_model.safetensors
"""

import os
import argparse
import torch
from contextlib import nullcontext

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import sys
from pathlib import Path

# Add parent directory to path so imports work when running script directly
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformer_train.transformer.train import load_checkpoint
from transformer_train.transformer.tokenizer import create_tokenizer
from transformer_train.transformer.transformer_block import Transformer
from transformer_train.transformer.engine import Engine
from transformer_train.transformer.common import get_dist_info, autodetect_device_type
from safetensors import safe_open
from safetensors.torch import load_file
import json


def load_model(model_path, device_type="cuda", dtype="bfloat16"):
    if not model_path.endswith('.safetensors'):
        model_path = model_path.replace('.pt', '.safetensors')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load metadata
    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    
    # Parse config
    config = json.loads(metadata.get('config', '{}'))
    if not config:
        # Try model_config
        config = json.loads(metadata.get('model_config', '{}'))
    
    # Get tokenizer path from metadata
    tokenizer_path = metadata.get('tokenizer_path', 'tokenizer')
    
    # Load tokenizer
    if os.path.isdir(tokenizer_path):
        tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
    else:
        tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Set vocab_size from tokenizer if not in config
    if 'vocab_size' not in config or config['vocab_size'] is None:
        config['vocab_size'] = tokenizer.get_vocab_size()
    
    # Create model
    device = torch.device(device_type)
    ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
    
    # Initialize model on meta device for memory efficiency
    with torch.device("meta"):
        model = Transformer(config, tokenizer)
    
    # Move to actual device
    model.to_empty(device=device)
    model.apply(model._init_weights)
    
    # Re-initialize rotary embeddings
    head_dim = config['n_embd'] // config['n_head']
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.cos = cos.to(device)
    model.sin = sin.to(device)
    
    # Special initialization for output projections
    torch.nn.init.zeros_(model.lm_head.weight)
    for block in model.h:
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.zeros_(block.attn.c_proj.weight)
    
    # Load weights
    model_state = load_file(model_path)
    
    # Remove compiled model prefixes
    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Convert to appropriate dtype
    if device_type == "cuda" and dtype == "bfloat16":
        model = model.to(dtype=ptdtype)
        model.cos = model.cos.to(dtype=ptdtype)
        model.sin = model.sin.to(dtype=ptdtype)
    
    model.eval()
    
    return model, tokenizer


def generate_single(model, tokenizer, prompt, max_tokens=256, temperature=0.6, top_k=50, device_type="cuda", dtype="bfloat16"):
    device = torch.device(device_type)
    ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Create engine
    engine = Engine(model, tokenizer)
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if isinstance(tokens[0], list):
        tokens = tokens[0]
    
    # Generate
    generated_tokens = []
    with autocast_ctx:
        for token_column, token_masks in engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=42
        ):
            if token_masks[0] == 1:  # sampled token
                generated_tokens.append(token_column[0])
    
    # Decode
    full_tokens = tokens + generated_tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    return generated_text


def interactive_chat(model, tokenizer, device_type="cuda", dtype="bfloat16", temperature=0.6, top_k=50, max_tokens=256):
    device = torch.device(device_type)
    ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Create engine
    engine = Engine(model, tokenizer)
    
    # Special tokens
    bos_token_id = tokenizer.get_bos_token_id()
    try:
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        has_chat_tokens = True
    except:
        has_chat_tokens = False
        print("Chat tokens not found, using simple format")
    
    print("\n" + "=" * 80)
    print("INTERACTIVE CHAT MODE")
    print("=" * 80)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 80)
    
    conversation_tokens = [bos_token_id]
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        # Handle special commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_tokens = [bos_token_id]
            print("Conversation cleared.")
            continue
        
        if not user_input:
            continue
        
        # Add user message
        if has_chat_tokens:
            conversation_tokens.append(user_start)
            user_tokens = tokenizer.encode(user_input, add_special_tokens=False)
            if isinstance(user_tokens[0], list):
                user_tokens = user_tokens[0]
            conversation_tokens.extend(user_tokens)
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)
        else:
            # Simple format: just add user input
            user_tokens = tokenizer.encode(user_input, add_special_tokens=False)
            if isinstance(user_tokens[0], list):
                user_tokens = user_tokens[0]
            conversation_tokens.extend(user_tokens)
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        response_tokens = []
        
        with autocast_ctx:
            for token_column, token_masks in engine.generate(
                conversation_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=42
            ):
                if token_masks[0] == 1:  # sampled token
                    token_id = token_column[0]
                    response_tokens.append(token_id)
                    conversation_tokens.append(token_id)
                    
                    # Stream output
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                    print(token_text, end="", flush=True)
                    
                    # Check for end token
                    if has_chat_tokens and token_id == assistant_end:
                        break
        
        print()  # New line after response
        
        if has_chat_tokens:
            conversation_tokens.append(assistant_end)


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained model')
    parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt text (if empty, use interactive mode)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive chat mode')
    parser.add_argument('-m', '--model-path', type=str, default='checkpoints/best_model_120m_8gb.safetensors', help='Path to model checkpoint')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens to generate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type (empty = autodetect)')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], help='Data type')
    
    args = parser.parse_args()
    
    # Autodetect device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    print(f"Using device: {device_type}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    try:
        model, tokenizer = load_model(args.model_path, device_type=device_type, dtype=args.dtype)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first or specify correct model path.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate
    if args.interactive or not args.prompt:
        interactive_chat(
            model, tokenizer,
            device_type=device_type,
            dtype=args.dtype,
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )
    else:
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerated:", end=" ")
        try:
            generated = generate_single(
                model, tokenizer, args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device_type=device_type,
                dtype=args.dtype
            )
            print(generated)
        except Exception as e:
            print(f"\nError during generation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
