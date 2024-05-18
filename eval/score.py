from evaluate import load
import os
import json
from tqdm import tqdm

LABEL_PATH = 'project/data/label'
LABEL_NAME = 'gen_8B.jsonl'

with open(os.path.join(LABEL_PATH, LABEL_NAME)) as json_file:
    json_list = list(json_file)

bleu = load("bleu")
bertscore = load("bertscore")

belu_8B = 0
bert_8B = 0

iteration_size = 0

for json_str in tqdm(json_list, desc="Processing"):

    item = json.loads(json_str)

    ref = item['text']['meme_captions']
    predictions = item['response']
    iteration_size += 1

    # Llama3_8B

    results = bleu.compute(predictions=predictions, references=references)
    belu_8B += results['precisions'][0]

    results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="microsoft/deberta-xlarge-mnli")
    bert_8B += results['f1'][0]

output = {
    "belu_8B": belu_8B,
    "bert_8B": bert_8B
}

filename = 'output.json'

with open(filename, 'w') as json_file:
    json.dump(output, json_file)

