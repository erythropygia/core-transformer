"""
Generate text using trained transformer model.

Usage:
    python generate_text.py -p "Your prompt here"
    python generate_text.py --interactive
    python generate_text.py -p "Prompt" --model-path checkpoints/best_model.safetensors
"""

import os
import argparse
import torch
from contextlib import nullcontext
import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformer_train.transformer.tokenizer import create_tokenizer
from transformer_train.transformer.model.transformer_block import Transformer
from transformer_train.transformer.model.engine import Engine
from transformer_train.transformer.common import autodetect_device_type

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
        config = json.loads(metadata.get('model_config', '{}'))
    
    # Get tokenizer path
    tokenizer_path = metadata.get('tokenizer_path', 'tokenizer')
    
    # Load tokenizer
    if os.path.isdir(tokenizer_path):
        tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
    else:
        tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Set vocab_size
    if 'vocab_size' not in config or config['vocab_size'] is None:
        config['vocab_size'] = tokenizer.get_vocab_size()
    
    # Create model (simple, like nanochat)
    device = torch.device(device_type)
    model = Transformer(config, tokenizer).to(device)
    
    # Load weights
    model_state = load_file(model_path)
    
    # Remove compiled model prefixes if any
    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Convert to dtype if needed
    if device_type == "cuda" and dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    
    model.eval()
    
    return model, tokenizer


def generate_single(model, tokenizer, prompt, max_tokens=256, temperature=0.6, top_k=50, 
                    device_type="cuda", dtype="bfloat16"):
    device = torch.device(device_type)
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Create engine
    engine = Engine(model, tokenizer)
    
    # Get end-of-sequence tokens
    bos_token_id = None
    assistant_end_token_id = None
    try:
        bos_token_id = tokenizer.get_bos_token_id()
    except:
        pass
    try:
        assistant_end_token_id = tokenizer.encode_special("<|assistant_end|>")
    except:
        pass
    
    # Encode prompt (simple, like nanochat)
    prompt_tokens = tokenizer.encode(prompt)
    if isinstance(prompt_tokens, list) and len(prompt_tokens) > 0 and isinstance(prompt_tokens[0], list):
        prompt_tokens = prompt_tokens[0]
    
    # Generate
    response_tokens = []
    with autocast_ctx:
        for token_column, token_masks in engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=42
        ):
            token = token_column[0]  # Single sample
            
            # Stop early if we hit an end-of-sequence token (don't include it in output)
            if (bos_token_id is not None and token == bos_token_id) or \
               (assistant_end_token_id is not None and token == assistant_end_token_id):
                break
            
            response_tokens.append(token)
    
    # Decode only the generated part (skip special tokens like <|bos|>)
    generated_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return generated_text


def interactive_chat(model, tokenizer, device_type="cuda", dtype="bfloat16", 
                     temperature=0.6, top_k=50, max_tokens=256):
    device = torch.device(device_type)
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Create engine
    engine = Engine(model, tokenizer)
    
    # Special tokens
    bos = tokenizer.get_bos_token_id()
    
    # Check for chat tokens
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
    print("Interactive chat mode")
    print("=" * 80)
    print("Commands:")
    print("  'quit' or 'exit' - End conversation")
    print("  'clear' - Start new conversation")
    print("-" * 80)
    
    conversation_tokens = [bos]
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue
        
        if not user_input:
            continue
        
        # Add user message
        if has_chat_tokens:
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(user_input))
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)
        else:
            # Simple format: just add user input
            conversation_tokens.extend(tokenizer.encode(user_input))
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        response_tokens = []
        
        # Get BOS token ID for early stopping
        bos_token_id = None
        try:
            bos_token_id = tokenizer.get_bos_token_id()
        except:
            pass
        
        with autocast_ctx:
            for token_column, token_masks in engine.generate(
                conversation_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=42
            ):
                token = token_column[0]
                
                # Check for end tokens (stop early, don't include in output)
                if has_chat_tokens and token == assistant_end:
                    break
                if bos_token_id is not None and token == bos_token_id:
                    break
                
                response_tokens.append(token)
                
                # Stream output (skip special tokens like <|bos|>)
                token_text = tokenizer.decode([token], skip_special_tokens=True)
                print(token_text, end="", flush=True)
        
        print()  # New line
        
        # Update conversation (no duplicate assistant_end!)
        conversation_tokens.extend(response_tokens)
        
        # Ensure assistant_end is at the end if using chat tokens
        if has_chat_tokens and (not response_tokens or response_tokens[-1] != assistant_end):
            conversation_tokens.append(assistant_end)


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained model')
    parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt text (if empty, use interactive mode)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive chat mode')
    parser.add_argument('-m', '--model-path', type=str, default='checkpoints/best_model_120m_8gb.safetensors', 
                       help='Path to model checkpoint')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens to generate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], 
                       help='Device type (empty = autodetect)')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], 
                       help='Data type')
    
    args = parser.parse_args()
    
    # Autodetect device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    print(f"Using device: {device_type}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    try:
        model, tokenizer = load_model(args.model_path, device_type=device_type, dtype=args.dtype)
        print("Model loaded successfully!")
        print(f"Model parameters: {model.get_num_params()/1e6:.1f}M")
        print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
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
        print("Generated:", end=" ")
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
