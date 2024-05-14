import argparse
import logging
import torch
from generate import generate
from utils import TrainingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_lightning import seed_everything

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_config():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--version", type=str)
    parser.add_argument("--save_dir", type=str)

    # model
    parser.add_argument("--hf_model", type=str)
    parser.add_argument("--tokenizer_path", type=str)

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--structured_instruction", action="store_true")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--n_gpus", type=int)
    parser.add_argument("--n_nodes", type=int)

    # inference
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--ckpt_path", type=str)

    return parser.parse_args()

def main():
    config = parse_config()
    config = TrainingConfig(**vars(config))
    
    seed_everything(42, workers=True)

    logger.info('Loading Model ...')
    model = AutoModelForCausalLM.from_pretrained(config.hf_model, device_map="auto", load_in_8bit=config.load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=False)
    generate(model, tokenizer, config)

if __name__ == '__main__':
    main()
