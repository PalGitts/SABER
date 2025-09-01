import csv
import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

with open("config.json", 'r') as f:
    config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cls = AutoModelForCausalLM.from_pretrained("cais/HarmBench-Llama-2-13b-cls", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("cais/HarmBench-Llama-2-13b-cls", use_fast=False, truncation_side="left", padding_side="left")

def llm_eval(responses):
    eval_prompt = config["prompts"]["Harmbench_Test_Eval"]
    evals = []
    for res in responses:
        inputs = eval_prompt.format(query=res[0], response=res[1])
        encoded = tokenizer(inputs, return_tensors='pt', padding='longest')
        with torch.no_grad():
            output_ids = cls.generate(
                **encoded.to(cls.device),
                do_sample=False,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
            ).cpu()
            output_ids = output_ids[:, len(encoded.input_ids[0]):]

        completion = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        evals.append(completion[0])

    score = [1 for eval in evals if eval == "Yes"]
    print(f"ASR: {sum(score)/len(evals)*100:.2f}%")

if __name__ ==  "__main__":
  
    parser = argparse.ArgumentParser(description="For adding csv file path.")
    parser.add_argument("--file", type=str, required=True, help="Specify the file path.")
    args = parser.parse_args()

    responses = []
    with open(args.file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            responses.append(row)
    
    llm_eval(responses)

