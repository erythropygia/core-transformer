
import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import argparse

from transformer_train.transformer.training.train import train
from transformer_train.transformer.common import compute_init, compute_cleanup, print0, autodetect_device_type
from transformer_train.transformer.config import TRAINING_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Train transformer model')

    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate (if not using Muon)')
    parser.add_argument('--max-epochs', type=int, default=None, help='Maximum epochs')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type (empty = autodetect)')

    parser.add_argument('--use-muon', action='store_true', default=None, help='Use Muon optimizer')
    parser.add_argument('--no-muon', dest='use_muon', action='store_false', help='Disable Muon optimizer')
    parser.add_argument('--unembedding-lr', type=float, default=None, help='Unembedding learning rate (Muon)')
    parser.add_argument('--embedding-lr', type=float, default=None, help='Embedding learning rate (Muon)')
    parser.add_argument('--matrix-lr', type=float, default=None, help='Matrix learning rate (Muon)')

    parser.add_argument('--resume', action='store_true', help='Auto-resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from specific checkpoint')
    parser.add_argument('--pretrained', type=str, default=None, help='Load pretrained model and continue training')
    parser.add_argument('--fresh-epochs', type=int, default=None, help='Number of fresh epochs after loading pretrained')

    parser.add_argument('--tokenizer-dir', type=str, default=None, help='Tokenizer directory path')

    parser.add_argument('--use-wandb', action='store_true', default=None, help='Enable wandb logging')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable wandb logging')

    args = parser.parse_args()

    config_updates = {}
    if args.batch_size is not None:
        config_updates['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_updates['learning_rate'] = args.learning_rate
    if args.max_epochs is not None:
        config_updates['max_epochs'] = args.max_epochs
    if args.use_muon is not None:
        config_updates['use_muon_optimizer'] = args.use_muon
    if args.unembedding_lr is not None:
        config_updates['unembedding_lr'] = args.unembedding_lr
    if args.embedding_lr is not None:
        config_updates['embedding_lr'] = args.embedding_lr
    if args.matrix_lr is not None:
        config_updates['matrix_lr'] = args.matrix_lr
    if args.use_wandb is not None:
        config_updates['use_wandb'] = args.use_wandb

    TRAINING_CONFIG.update(config_updates)

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    print0("=" * 80)
    print0("CORE-TRANSFORMER TRAINING")
    print0("=" * 80)
    print0(f"Device type: {device_type}")
    print0(f"Distributed: {ddp} (rank {ddp_rank}/{ddp_world_size})")
    print0(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print0(f"Max epochs: {TRAINING_CONFIG['max_epochs']}")
    print0(f"Muon optimizer: {TRAINING_CONFIG.get('use_muon_optimizer', False)}")
    if TRAINING_CONFIG.get('use_muon_optimizer', False):
        print0(f"  Unembedding LR: {TRAINING_CONFIG.get('unembedding_lr', 0.004)}")
        print0(f"  Embedding LR: {TRAINING_CONFIG.get('embedding_lr', 0.2)}")
        print0(f"  Matrix LR: {TRAINING_CONFIG.get('matrix_lr', 0.02)}")
    else:
        print0(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print0("=" * 80)
    print0()

    try:
        auto_resume = args.resume
        resume_from_checkpoint = args.checkpoint
        pretrained_model_path = args.pretrained
        fresh_epochs = args.fresh_epochs
        tokenizer_path = args.tokenizer_dir or "tokenizer"

        model = train(
            auto_resume=auto_resume,
            resume_from_checkpoint=resume_from_checkpoint,
            pretrained_model_path=pretrained_model_path,
            fresh_epochs=fresh_epochs,
            tokenizer_path=tokenizer_path
        )

        print0("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print0("\nTraining interrupted by user")
    except Exception as e:
        print0(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if torch.distributed.is_initialized():
                compute_cleanup()
        except:
            pass

        from transformer_train.transformer.utils import cleanup_memory
        cleanup_memory()


if __name__ == "__main__":
    main()
