import sys
import os
sys.path.append('transformers_train')

from transformers_train_deepspeed import generate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def generate_text():
    test_prompts = [
        "Bu olay TBMM'de tartışıldıktan sonra",
        "Ekrem İmamoğlu, İstanbul Büyükşehir Belediye Başkanı olarak",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated:", end=" ")
        
        try:
            result = generate(
                prompt, 
                model_path="checkpoints/checkpoint_step_60000.safetensors",
                max_new_tokens=100,
                temperature=0.9,
                top_k=10,
                top_p=0.90,
                silent=True
            )
            print(result)
            
        except FileNotFoundError:
            print("Model not found. Train the model first!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    generate_text() 