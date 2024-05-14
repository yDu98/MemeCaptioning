NAME="one_shot_llama_3_8B"
VERSION="1"

HF_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
TOKENIZER_PATH=$HF_MODEL 

DATASETS="/home/yc7093/ml_project/data/memes_test_ocr.jsonl" #training data

GPUS=4
NODES=1

SCRIPT="src/run.py \
    `# experiment` \
    --project MemeCap \
    --name $NAME \
    --version $VERSION \
    --save_dir logs \
    \
    `# model` \
    --hf_model $HF_MODEL \
    --tokenizer_path $TOKENIZER_PATH \
    \
    `# data` \
    --datasets $DATASETS \
    --structured_instruction \
    --max_length 600\
    --n_gpus $GPUS \
    --n_nodes $NODES \
    \
    `# inference` \
    "

python $SCRIPT

