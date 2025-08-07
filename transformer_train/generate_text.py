import os
from transformer.train import generate      
from transformer.tokenizer import create_tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  

def generate_text():
    test_prompts = [
        "Kuantum mekaniğinde hamilton işlemcisi",
        "Amerika ile Çin arasındaki mesafe",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}\n\n")
        print("Generated:", end=" ")
        
        try:
            result = generate(
                prompt, 
                model_path="checkpoints/checkpoint_step_71500.safetensors",
                max_new_tokens=75,
                temperature=0.9,
                top_k=10,
                top_p=0.9,
                silent=True
            )
            print(result)
            
        except FileNotFoundError:
            print("Model not found. Train the model first!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    generate_text() 