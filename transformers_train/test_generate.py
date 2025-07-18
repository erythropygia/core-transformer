#!/usr/bin/env python3
"""
Test script for clean text generation
"""

import sys
import os
sys.path.append('transformers_train')

from transformers_train_deepspeed import generate

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def test_generation():
    """Test text generation with clean output"""
    
    test_prompts = [
        "Türkiye'nin başkenti",
        "Yapay zeka teknolojisi",
        "İstanbul Boğazı"
    ]
    
    print("🤖 Turkish GPT-2 Text Generation Test")
    print("="*50)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("📄 Generated:", end=" ")
        
        try:
            # Silent generation - no debug output
            result = generate(
                prompt, 
                model_path="transformers_train/checkpoints/best_model_120m_8gb.safetensors",
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                silent=True  # Clean output
            )
            print(result)
            
        except FileNotFoundError:
            print("❌ Model not found. Train the model first!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_generation() 