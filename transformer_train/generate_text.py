
import os
import argparse
import torch
from contextlib import nullcontext
import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformer_train.transformer.tokenizer import create_tokenizer
from transformer_train.transformer.model.transformer_block import Transformer
from transformer_train.transformer.model.engine import Engine
from transformer_train.transformer.common import autodetect_device_type
from transformer_train.transformer.config import SHOW_SPECIAL_TOKENS

from safetensors import safe_open
from safetensors.torch import load_file
import json


def load_model(model_path, device_type="cuda", dtype="bfloat16"):
    if not model_path.endswith('.safetensors'):
        model_path = model_path.replace('.pt', '.safetensors')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()

    config = json.loads(metadata.get('config', '{}'))
    if not config:
        config = json.loads(metadata.get('model_config', '{}'))

    tokenizer_path = metadata.get('tokenizer_path', 'tokenizer')

    if os.path.isdir(tokenizer_path):
        tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
    else:
        tokenizer = create_tokenizer(model_path=tokenizer_path)

    if 'vocab_size' not in config or config['vocab_size'] is None:
        config['vocab_size'] = tokenizer.get_vocab_size()

    device = torch.device(device_type)
    model = Transformer(config, tokenizer).to(device)

    model_state = load_file(model_path)

    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    result = model.load_state_dict(new_state_dict, strict=False)
    trained = {name for name, _ in model.named_parameters()}
    absent = [k for k in result.missing_keys if k in trained]
    if absent:
        raise RuntimeError(
            f"{model_path} is missing {len(absent)} trained tensors: {absent[:8]}. "
            f"Refusing to generate from a partly random model."
        )

    if device_type == "cuda" and dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)

    model.eval()

    return model, tokenizer


def generate_single(model, tokenizer, prompt, max_tokens=256, temperature=0.6, top_k=50, 
                    top_p=1.0, repetition_penalty=1.0, device_type="cuda", dtype="bfloat16"):
    torch.device(device_type)
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    engine = Engine(model, tokenizer)

    stop_token_ids = set(tokenizer.get_stop_token_ids())

    prompt_tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    if isinstance(prompt_tokens, list) and len(prompt_tokens) > 0 and isinstance(prompt_tokens[0], list):
        prompt_tokens = prompt_tokens[0]

    response_tokens = []
    with autocast_ctx:
        for token_column, token_masks in engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=42
        ):
            token = token_column[0]

            if token in stop_token_ids:
                if SHOW_SPECIAL_TOKENS:
                    response_tokens.append(token)
                break

            response_tokens.append(token)

    full_sequence = prompt_tokens + response_tokens
    if SHOW_SPECIAL_TOKENS:
        generated_text = tokenizer.decode(full_sequence, skip_special_tokens=False)
    else:
        generated_text = tokenizer.decode(full_sequence, skip_special_tokens=True)

    return generated_text


def interactive_chat(model, tokenizer, device_type="cuda", dtype="bfloat16", 
                     temperature=0.6, top_k=50, top_p=1.0, repetition_penalty=1.0, max_tokens=256):
    torch.device(device_type)
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    engine = Engine(model, tokenizer)

    tokenizer.get_bos_token_id()
    stop_ids = set(tokenizer.get_stop_token_ids())

    print("\n" + "=" * 80)
    print("Interactive chat mode")
    print("=" * 80)
    print("Commands:")
    print("  'quit' or 'exit' - End conversation")
    print("  'clear' - Start new conversation")
    print("-" * 80)

    messages = []

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            messages = []
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        conversation_tokens = tokenizer.render_for_completion({"messages": messages})

        print("\nAssistant: ", end="", flush=True)
        response_tokens = []

        with autocast_ctx:
            for token_column, token_masks in engine.generate(
                conversation_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=42
            ):
                token = token_column[0]

                if token in stop_ids:
                    break

                response_tokens.append(token)

                token_text = tokenizer.decode([token], skip_special_tokens=not SHOW_SPECIAL_TOKENS)
                print(token_text, end="", flush=True)

        print()

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained model')
    parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt text (if empty, use interactive mode)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive chat mode')
    parser.add_argument('-m', '--model-path', type=str, default='checkpoints/best_model_120m_8gb.safetensors', 
                       help='Path to model checkpoint')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p (nucleus) sampling threshold (0.0-1.0)')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty (>1.0 to penalize repetition, default=1.1)')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens to generate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], 
                       help='Device type (empty = autodetect)')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], 
                       help='Data type')

    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    print(f"Using device: {device_type}")

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

    if args.interactive or not args.prompt:
        interactive_chat(
            model, tokenizer,
            device_type=device_type,
            dtype=args.dtype,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens
        )
    else:
        print(f"\nPrompt: {args.prompt}")
        print("Full Output (with BOS):", end=" ")
        try:
            generated = generate_single(
                model, tokenizer, args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
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
