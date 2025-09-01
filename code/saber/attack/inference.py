import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoTokenizer
from saber.models import ModifiedLlamaForCausalLM, ModifiedMistralForCausalLM  # Modified files, refer to saber/models/llama/modeling_llama.py and saber/models/mistral/modeling_mistral.py
import csv
import torch
import argparse
import json


with open("config.json", 'r') as f:
    config = json.load(f)

def get_output(prompt, model, tokenizer, start_layer=None, end_layer=None, factor=None):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_id)
    attention_mask = torch.ones_like(input_ids).to(device_id)
    ipToken_length = input_ids.shape[1]
    ipToken_length = input_ids.shape[1]

    model.set_layer_selection(start_layer, end_layer, factor)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=ipToken_length+new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated_text = tokenizer.decode(outputs[0, ipToken_length:], skip_special_tokens=True)
    generated_text = "Sure, here " + generated_text
    return generated_text

if __name__ ==  "__main__":

    parser = argparse.ArgumentParser(description="Select Config")
    parser.add_argument("--model", type=str, required=True, help="Select model")
    parser.add_argument("--dataset", type=str, required=True, help="Select dataset")
    parser.add_argument("--out_path", type=str, required=True, help="Select output path")
    parser.add_argument("--sys_prompt", type=int, required=False, help="Whether to use default system prompt of the model")
    args = parser.parse_args()

    model_name = args.model
    model_path = config["models"][model_name]['path']
    params = config["models"][model_name]['params']
    dataset_name = args.dataset
    dataset_path = config["datasets"][dataset_name]['path']
    new_tokens = config["datasets"][dataset_name]['new_tokens']
    device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompt = '''Question: {query}
Answer: Sure, here'''
    if args.sys_prompt:
        if model_name == "mistral7b":
            print("Mistral7b does not have a system prompt.")
        elif model_name == "vicuna7b":
            prompt = config["prompts"]["Vicuna_System"]
        else:
            prompt = config["prompts"]["Llama_System"]
 
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_name == "mistral7b":
        model = ModifiedMistralForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    else:
        model = ModifiedLlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model = model.to(device_id)
    model = model.eval()
    print("Model Loaded")

    adv_prompts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            adv_prompts.append(row[0].strip())

    data_adv = [["Prompt", "Response"]]

    for i, adv_prompt in enumerate(adv_prompts):
        output = get_output(prompt.format(query=adv_prompt), model, tokenizer, params["s"], params["e"], params["f"])
        data_adv.append([adv_prompt,output])
        print(f"Adv Iteration {i} done")

    with open(args.out_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data_adv)

       