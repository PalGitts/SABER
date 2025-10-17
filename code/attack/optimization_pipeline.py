import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from saber.models import ModifiedLlamaForCausalLM, ModifiedMistralForCausalLM  # Modified files, refer to saber/models/llama/modeling_llama.py and saber/models/mistral/modeling_mistral.py
import csv
import torch
import json
import torch.nn.functional as F
import random
import numpy as np
from scipy.spatial.distance import cdist
import argparse

parser = argparse.ArgumentParser(description="Model Name")
parser.add_argument("--model", type=str, required=True, help="Specify the model name")
args = parser.parse_args()

with open("config.json", 'r') as f:
    config = json.load(f)

model_name = args.model
model_path = config["models"][model_name]['path']
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if model_name == "mistral7b":
    model = ModifiedMistralForCausalLM.from_pretrained(model_path)
else:
    model = ModifiedLlamaForCausalLM.from_pretrained(model_path)
model = model.to(device_id)
model = model.eval()
val_cls = AutoModelForCausalLM.from_pretrained("cais/HarmBench-Mistral-7b-val-cls", torch_dtype=torch.bfloat16, device_map="auto")
val_tokenizer = AutoTokenizer.from_pretrained("cais/HarmBench-Mistral-7b-val-cls", use_fast=False, truncation_side="left", padding_side="left")
val_cls = val_cls.eval()
print("Model Loaded")

print("Loading Datasets...")
harm_test = []
harm_val = []
safe_val = []
with open("data/harmbench_test.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        harm_test.append(row[0].strip())
with open("data/harmbench_val.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        harm_val.append(row[0].strip())
with open("data/alpaca.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        safe_val.append(item['instruction'])
random.shuffle(safe_val)
safe_val = safe_val[:41]
print("Datasets Loaded")


print("Step 1: Detection of Layer Boundaries")

def get_hs(prompt, start_layer=None, end_layer=None, factor=None):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_id)
    attention_mask = torch.ones_like(input_ids).to(device_id)
    ipToken_length = input_ids.shape[1]

    model.set_layer_selection(start_layer, end_layer, factor)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict_in_generate=True, 
        max_length=ipToken_length+1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    hidden_states = [layer[0, -1, :].detach().cpu() for layer in outputs.hidden_states[0]]  # 33 * 4096
    return torch.stack(hidden_states)

safeHiddenStates = []
harmHiddenStates = []

prompt = '''Question: {query}
Answer: '''

for i , query in enumerate(safe_val):
    if i == 300:
        break
    hidden_states = get_hs(prompt.format(query=query))
    safeHiddenStates.append(hidden_states)
safeHiddenStates = torch.stack(safeHiddenStates)

for i , query in enumerate(harm_val):
    if i == 300:
        break
    hidden_states = get_hs(prompt.format(query=query))
    harmHiddenStates.append(hidden_states)
harmHiddenStates = torch.stack(harmHiddenStates)

def get_cosine_distances(safe_states, harm_states):

    num_layers = safe_states.shape[1]
    cosine_distances = []
    
    for layer in range(num_layers):
        # Get states for current layer
        safe_layer = safe_states[:, layer, :].numpy()
        harm_layer = harm_states[:, layer, :].numpy()
        # Calculate average cosine distance
        cos_dist = np.mean(cdist(safe_layer, harm_layer, metric='cosine'))
        cosine_distances.append(cos_dist)
    
    return cosine_distances

cosine_distances = get_cosine_distances(safeHiddenStates, harmHiddenStates)

fo_diff = []
for i in range(len(cosine_distances)-1):
    fo_diff.append((cosine_distances[i+1]-cosine_distances[i]))
for i in range(1, len(fo_diff)):
    if fo_diff[-i] > 0.04:
        end_layer = len(cosine_distances) - i
        break

for i in range(len(fo_diff)):
    if fo_diff[i] > 0.04:
        start_layer = i
        break

print(f"Start Layer Boundary: {start_layer}")
print(f"End Layer Boundary: {end_layer}")

print("Step 2: Finding Optimal Factor")

pairs = []
for i in range (start_layer, end_layer+1):
    for j in range (i+1, end_layer+1):
        pairs.append((i,j))

head = model.lm_head

prompt = '''Question: {query}
Answer: Sure, here'''

hidden_size = model.config.hidden_size

kl = torch.zeros(41, 15)

for query_num, query in enumerate(safe_val[:41]):
    baseline_hs = get_hs(prompt.format(query=query))[-1]
    baseline_logits = head(baseline_hs.to(device_id)).detach().cpu()
    hs = torch.zeros((15, hidden_size))
    # iterate factor 0.1, 0.2, 0.3, ....
    for i in range(15):
        s, e = random.choice(pairs)
        factor = (i+1)/10
        hs[i] = get_hs(prompt.format(query=query), s, e, factor)[-1]
    logits = head(hs.to(device_id)).detach().cpu()
    kl_divergences = F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.log_softmax(baseline_logits, dim=-1).unsqueeze(0).expand_as(logits),
        log_target=True,
        reduction='none'
    ).sum(dim=-1)
    kl[query_num] = kl_divergences

mean_kl = kl.mean(dim=0)

for factor, kld in enumerate(reversed(mean_kl)):
    if kld<0.05:
        factor = 1.1 - factor/10
        break
print("Optimal Factor: ", factor)

print("Step 3: Finding Optimal Layer Pair")

def get_response(prompt, start_layer=None, end_layer=None, factor=None):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_id)
    attention_mask = torch.ones_like(input_ids).to(device_id)
    ipToken_length = input_ids.shape[1]

    model.set_layer_selection(start_layer, end_layer, factor)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=ipToken_length+128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    generated_text = tokenizer.decode(outputs[0, ipToken_length:], skip_special_tokens=True)
    generated_text = "Sure, here " + generated_text
    return generated_text

def eval_response(behaviour, generation):
    inputs = config["prompts"]["Harmbench_Val_Eval"].format(query=behaviour, response=generation)
    encoded = val_tokenizer(inputs, return_tensors='pt', padding='longest')
    with torch.no_grad():
        output_ids = val_cls.generate(
            **encoded.to(val_cls.device),
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id
        ).cpu()
        output_ids = output_ids[:, len(encoded.input_ids[0]):]

    completion = val_tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    score = 1 if completion == "Yes" else 0
    return score

pair_accuracy = torch.zeros(len(pairs))
for i, (s,e) in enumerate(pairs):
    score = 0
    for query in harm_val:
        response = get_response(prompt.format(query=query), s, e, factor)
        score = score + eval_response(query, response)
    acc = score/41*100
    pair_accuracy[i] = acc

accuracy, pair_idx = torch.max(pair_accuracy, dim = -1)
print(f"Optimal Pair: {pairs[pair_idx]}")