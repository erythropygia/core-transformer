import torch
from transformer.train import train

if __name__ == "__main__":  
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformers warnings
    
    try:
        #model = train()
        model = train(auto_resume=True)  # Resume from latest checkpoint
        
        # Example usage:
        # model = train(resume_from_checkpoint="checkpoints/checkpoint_step_1000.safetensors")
        # model = train(pretrained_model_path="checkpoints/best_model_120m_8gb.safetensors", fresh_epochs=10)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except:
            pass
        
        from transformer.utils import cleanup_memory
        cleanup_memory()