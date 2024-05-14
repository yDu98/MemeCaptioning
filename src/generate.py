import os, json
import time
import torch
from tqdm import tqdm

from transformers import GenerationConfig

torch.backends.cuda.matmul.allow_tf32 = True

def generate_prompt(prompt, config):
    if config.structured_instruction:
        """
        Referring to https://github.com/openai/openai-python/blob/main/chatml.md,
        we set the structure as follows:
        <bos>Human
        How are you<eos>
        <bos>Assistant
        I am doing well!<eos>
        <bos>Human
        How are you now?<eos>
        <bos>Assistant

        # Note that there is a '\n' after each line.
        """

        return f"""{config.BOS}Human
{prompt}{config.EOS}
{config.BOS}Assistant
"""
    else:
        return config.BOS + prompt + config.SEP


def preprocess(prompt, config):
    prompt = generate_prompt(prompt, config)
    if config.structured_instruction:
        segmenter = f"{config.BOS}Assistant\n"
    else:
        segmenter = config.SEP
    return prompt, segmenter

def postprocess(ans, segmenter, config):
    ans = ans.replace(f"{config.BOS} ", config.BOS) # Llama tokenizer add a space after bos token wtf
    if ans[-len(config.EOS):] == config.EOS:
        ans = ans[:-len(config.EOS)]
    return ans.split(segmenter)[-1].strip()

def generate_a_response(
    prompt,
    model,
    tokenizer,
    generation_config,
    config,
    input_conversation=False,
    conversation=None
    ):
    prompt, segmenter = preprocess(prompt, config)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).cuda()
    
    if input_conversation and conversation["ids"] is not None:
        input_ids = torch.cat([conversation["ids"], input_ids], dim=1)

    ratio = 0.75
    limit = int(config.max_length * ratio)
    if input_ids.size(1) > limit:
        input_ids = input_ids[:][-limit:]
        print("Warning: Truncate input to max limit on tokens number.")

    result = model.generate(
        input_ids,
        max_length=config.max_length,
        generation_config=generation_config,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output = result["sequences"] if isinstance(result, dict) else result
    ans = tokenizer.decode(output[0], skip_special_tokens=False)
    ans = postprocess(ans, segmenter, config)
    return ans

def generate_and_io(
        model,
        tokenizer,
        config,
        out_file_handler,
        question_id=None,
        prompt=None,
        output_jsonl_file=None,
        input_conversation=False,
        conversation=None,
        use_tqdm=False,
        ):

    def save(sent):
        out_file_handler.write(sent + '\n')

    def print_and_write(sent=''):
        if use_tqdm:
            tqdm.write(sent)
        else:
            print(sent)
        save(sent)

    def save_jsonl(question_id, prompt, response, file):
        with open(file, 'a', encoding="utf-8") as jf:
            jline = json.dumps({"question_id": question_id, "prompt": prompt, "response": response}, ensure_ascii=False)
            jf.write(jline + '\n')

    print("\nHuman:")
    if prompt is None:
        prompt = ''
        input_ = input()
        while input_ != '<taide_end>':
            prompt += input_ if prompt == '' else '\n' + input_
            input_ = input()
    else:
        print(prompt)
    save("Human:\n" + prompt)

       
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.65,
        # num_beams=4,
        # repetition_penalty=1.3,
        no_repeat_ngram_size=7,
    )
    
    t0 = time.time()
    response = generate_a_response(
        prompt,
        model,
        tokenizer,
        generation_config,
        config,
        )
    t = time.time() - t0

    print_and_write("Assistant:")
    print_and_write(response)
    print_and_write(f"time cost: {t:.1f}s")
    print_and_write()

    if output_jsonl_file is not None:
        save_jsonl(question_id, prompt, response, output_jsonl_file)

    

def process_jsonl_to_prompts(filepath):
    question_ids, prompts = [], []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON line
            data = json.loads(line)

            question_id = data['question_id']            
            # Extract necessary fields
            title = data['text']['title']
            img_captions = data['text']['img_captions']
            ocr_captions = data['text']['ocr_captions']
            metaphors = data['text']['metaphors']


            # Format the image description by joining all captions with a connector
            if len(img_captions) > 1:
                # Combine captions using ", " and replace the last comma with " and"
                image_description = ", ".join(img_captions[:-1]) + " and " + img_captions[-1]
            else:
                # If there's only one caption, use it directly
                image_description = img_captions[0]
            
            # Create the formatted prompt
            prompt = f"This is a meme with the title: '{title}'. \n"
            prompt += f"The image description is: '{image_description}'. "
            prompt += f"The following text is written inside the meme: '{ocr_captions}'. "
             
            # Constructing the metaphor rationale string
            rationale = "Rationale: "
            rationale_parts = [f"'{metaphor['metaphor']}' is a metaphor for '{metaphor['meaning']}';" for metaphor in metaphors]
            rationale += " ".join(rationale_parts)[:-1] + ". "
            prompt += rationale 

            prompt += "What is the meme poster trying to convey? Please summary it to one sentence."
            prompts.append(prompt)
            question_ids.append(question_id)
    
    return question_ids, prompts

def generate(model, tokenizer, config):
    
    question_ids, prompts  = process_jsonl_to_prompts(config.datasets) # should be a list of string

    gen_dir = os.path.join(config.ckpt_dir, "generation")
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)

    jsonl_file = os.path.join(gen_dir, "gen.jsonl")

    file = os.path.join(gen_dir, "gen.txt")
    with open(file, 'a', encoding="utf-8") as f:

        for question_id, prompt in tqdm(zip(question_ids, prompts)):
            generate_and_io(
                model,
                tokenizer,
                config,
                f,
                question_id=question_id,
                prompt=prompt,
                output_jsonl_file=jsonl_file,
                use_tqdm=True,
            )
        
