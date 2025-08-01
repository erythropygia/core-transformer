import os
from transformer.train import generate      
from transformer.tokenizer import create_tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  

def generate_text():
    test_prompts = [
        "Kuantum mekaniğinde hamilton işlemcisi, kinetik enerjilerin toplamına ve sistemdeki tüm parçacıklar için potansiyel enerjilere karşılık gelen",
        "Mustafa Kemal Atatürk'ün en bilinen sözlerinden biri olan 'Egemenlik, kayıtsız şartsız milletindir' ifadesi, Türkiye Cumhuriyeti'nin kuruluş felsefesini yansıtır. Bu sözün anlamı nedir?",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated:", end=" ")
        
        try:
            result = generate(
                prompt, 
                model_path="checkpoints/checkpoint_step_61000.safetensors",
                max_new_tokens=50,
                temperature=0.9,
                top_k=15,
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