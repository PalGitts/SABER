# SABER: Safety Alignment Bypass via Extra Residuals

SABER is a novel white-box jailbreaking method that exploits cross-layer residual connections to circumvent safety alignment mechanisms in large language models.

## Installation

1. Extract the zip file
```bash
unzip saber.zip
cd saber
```

2. Install requirements
```bash
pip install -r requirements.txt
```

## Supported Models

SABER currently supports the following models with pre-computed optimal parameters:

| Model ID | Model Path | Optimal Parameters (s, e, λ) |
|----------|------------|------------------------------|
| llama7b | meta-llama/Llama-2-7b-chat-hf | (5, 10, 1.0) |
| llama13b | meta-llama/Llama-2-13b-chat-hf | (6, 11, 1.0) |
| vicuna7b | lmsys/vicuna-7b-v1.5 | (9, 10, 0.9) |
| mistral7b | mistralai/Mistral-7B-Instruct-v0.2 | (6, 8, 0.2) |

## Usage

### Running SABER Attack

To run the SABER attack on a supported model:

```bash
python saber/attack/inference.py --model "model_id" --dataset "dataset_id" --out_path "output_path" [--sys_prompt 1]
```

Arguments:
- `--model`: Model identifier (choices: llama7b, llama13b, vicuna7b, mistral7b)
- `--dataset`: Dataset identifier (choices: HB_T, HB_V, JB, AB)
  - HB_T: HarmBench test set
  - HB_V: HarmBench validation set
  - JB: JailbreakBench
  - AB: AdvBench
- `--out_path`: Path to save the attack outputs
- `--sys_prompt`: Flag (1) to use the default system prompt of the model (optional)

Example:
```bash
python saber/attack/inference.py --model "mistral7b" --dataset "HB_T" --out_path "./results.csv"
```

### Evaluating Attack Results

To evaluate attack results on HarmBench:
```bash
python saber/evaluation/evaluation_HarmBench.py --file "results_file_path"
```

To evaluate attack results on JailbreakBench:
```bash
python saber/evaluation/evaluation_JBBench.py --file "results_file_path"
```

Example:
```bash
python saber/evaluation/evaluation_HarmBench.py --file "./results.csv"
```

### Finding Optimal Parameters for New Models

To determine optimal parameters for a new model:

1. Add your model to `config.json` with path information:
```json
{
  "models": {
    "your_model_id": {
      "path": "your/model/path",
      "params": {}
    },
    ...
  }
}
```

2. Run the optimization pipeline:
```bash
python saber/attack/optimization_pipeline.py --model "your_model_id"
```

This will:
- Identify layer boundaries where safety mechanisms are active
- Determine an optimal scaling factor
- Find the most effective source and target layers
- Update `config.json` with the optimal parameters

## Repository Structure

```
saber/
├── config.json                 # Model configurations and optimal parameters
├── README.md                   # This file
├── requirements.txt            # Dependencies
│
├── data/                       # Dataset files
│   ├── advbench.csv            # AdvBench dataset
│   ├── alpaca.json             # Benign prompts from ALPACA
│   ├── harmbench_test.csv      # HarmBench test set
│   ├── harmbench_val.csv       # HarmBench validation set
│   └── jbbench.csv             # JailbreakBench dataset
│
└── saber/                      # Core implementation
    ├── attack/
    │   ├── __init__.py
    │   ├── inference.py        # Attack inference code
    │   └── optimization_pipeline.py # Parameter optimization pipeline
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── evaluation_HarmBench.py  # HarmBench evaluation
    │   └── evaluation_JBBench.py    # JailbreakBench evaluation
    │
    ├── models/
    │   ├── llama/              # Llama model implementations
    │   ├── mistral/            # Mistral model implementations
    │   └── __init__.py
    │
    └── __init__.py
```