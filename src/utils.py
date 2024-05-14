import os, logging
from dataclasses import dataclass
from typing import *
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"

def debug(tokenizer):
    print("tokenizer.bos_token")
    print(tokenizer.bos_token)
    print("tokenizer.bos_token_id")
    print(tokenizer.bos_token_id)
    print("tokenizer.eos_token")
    print(tokenizer.eos_token)
    print("tokenizer.eos_token_id")
    print(tokenizer.eos_token_id)
    print("tokenizer.pad_token")
    print(tokenizer.pad_token)
    print("tokenizer.pad_token_id")
    print(tokenizer.pad_token_id)
    print("len(tokenizer)")
    print(len(tokenizer))
    print("tokenizer.padding_side")
    print(tokenizer.padding_side)
    exit()

@dataclass
class TrainingConfig:
    project: str
    name: str
    version: str
    save_dir: str
    datasets: str
    max_length: int
    
    num_workers: int = 1
    ckpt_path: Optional[str] = None
    hf_model: Optional[str] = None
    tokenizer_path: Optional[str] = None
    structured_instruction: bool = False

    n_gpus: int = 1
    n_nodes: int = 1
    device: str = "cpu"
    load_in_8bit: bool = False

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.save_dir, 'checkpoints', self.name, self.version)

    def __post_init__(self):
        self.config_tokenizer()

        if self.ckpt_path is None:
            last_path = os.path.join(self.ckpt_dir, 'last.ckpt')
            if os.path.exists(last_path):
                self.ckpt_path = last_path


    def config_tokenizer(self): 
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=False)
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
        
        if not tokenizer.bos_token:
            tokenizer.add_special_tokens(
                {
                    "eos_token": "</s>",
                    "bos_token": "<s>",
                }
            )

        self.BOS = tokenizer.bos_token
        self.EOS = tokenizer.eos_token
        self.SEP = tokenizer.eos_token
        self.tokenizer = tokenizer
