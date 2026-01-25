<div align="center">
<img src="img/bench_logo.png" border="0" width=400px/>
</div>

------

<div align="center">
    <a href="https://github.com/YuanchenBei/ColdRec"><img src="https://img.shields.io/badge/PRs-welcome-blue.svg"></a>
    <a href="https://github.com/YuanchenBei/ColdRec/blob/main/LICENSE"><img src="https://badgen.net/github/license/YuanchenBei/ColdRec?color=green"></a>
    <a href="https://arxiv.org/abs/2601.03515">
    <img src="https://img.shields.io/badge/📃%20arXiv-Paper-b31b1b.svg"></a>
    <a href="https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow"></a>
</div>

🚀 This is the project repository of *Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents*.

---

# 🌠 Mem-Gallery
🌇 **Mem-Gallery** is a multimodal long-term conversational memory benchmark for MLLM agents. Mem-Gallery contains a new multimodal conversational dataset and a unified evaluation framework.

---
## 🌏 Requirement
Experiments of the benchmark are conducted on the CUDA version 12.2.

``` bash
# For MLLM deployment
vllm >= 0.12.0
# For benchmark running
torch >= 2.5.1
transformers >= 4.51.3
sentence-transformers >= 5.1.2
accelerate >= 1.12.0
openai >= 2.11.0
```

---

## 📦 Dataset
The complete multimodal conversations with their corresponding evaluation QAs are available at the 🤗 [Hugging Face](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery).

---
## 🛫 Usage

1️⃣ **Dataset download**

Download the benchmark dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery).

Create a folder named **"data"** in the benchmark directory, and put the dataset's **"dialog"** and **"image"** folders into it.

2️⃣ **Configure the MLLM backbone and memory model to be tested**

default_config/DefaultEvalConfig.py
``` python
'name': '',  # Default judge model, can be overridden via command line [Replace with your model path]
'api_key': '', # [Replace with your api key]
'base_url': '', # [Replace with your model's base url]
```

default_config/DefaultFunctionConfig.py
``` python
'name': '', # [Replace with your model path]
```

default_config/DefaultGlobalConfig.py
``` python
DEFAULT_OPENAI_APIKEY = '' # [Replace with your api key]
DEFAULT_OPENAI_APIBASE = '' # [Replace with your api base url]
DEFAULT_BACKBONE_PATH = '' # [Replace with your llm backbone path]
DEFAULT_GME_QWEN2_VL_7B_PATH = '' # [Replace with your GME encoder path]
```

default_config/DefaultAUGUSTUSMemoryConfig.py
``` python
'name': '',  # Default model for concept extraction. [Replace with your LLM model path]
```

run/run_bench.py
``` python
if args.llm_name == 'qwen2-5-7b' or args.llm_name == 'qwen2-5-vl-3b' or args.llm_name == 'qwen2-5-vl-7b' or args.llm_name == 'qwen2-5-vl-32b': # flexible to be extended
    # local VLLM API
    OPENAI_APIKEY = '' # [Replace with your API key]
    OPENAI_APIBASE = '' # [Replace with your API base url]
    OPENAI_MODEL = f'' # [Replace with your model path, e.g., xxx/{args.llm_name}]
elif args.llm_name == 'gpt-4o-mini': # flexible to be extended
    # Openrouter API
    OPENAI_APIKEY = '' # [Replace with your API key]
    OPENAI_APIBASE = 'https://openrouter.ai/api/v1' # [Replace with your API base url]
    OPENAI_MODEL = 'openai/gpt-4o-mini' # [Replace with your model name, e.g., openai/gpt-4o-mini]
elif args.llm_name == 'gemini-2.5-flash' or args.llm_name == 'gemini-2.5-flash-lite':
    # Google Gemini API
    OPENAI_APIKEY = '' # [Replace with your API key]
    OPENAI_APIBASE = 'https://generativelanguage.googleapis.com/v1beta/openai/' # [Replace with your API base url]
    OPENAI_MODEL = args.llm_name # [Replace with your model name, e.g., gemini-2.5-flash]  
```

3️⃣ **Benchmark evaluation**

The main function for running benchmark is located in **run/run_bench.py**. 

You can adjust basic parameters via command-line arguments, such as MLLM (--llm_name) and memory model (--memory_name).

Currently supports 10+ memory models. For evaluations of A-Mem and MemoryOS, you can modify the official example code based on our run_bench.py ​​file.

A-Mem: https://github.com/WujiangXu/A-mem/blob/main/test_advanced.py

MemoryOS: https://github.com/BAI-LAB/MemoryOS/blob/main/eval/evalution_loco.py

---

## 📝 Citation
```bibtex
@article{bei2026mem,
  title={Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents},
  author={Bei, Yuanchen and Wei, Tianxin and Ning, Xuying and Zhao, Yanjun and Liu, Zhining and Lin, Xiao and Zhu, Yada and Hamann, Hendrik and He, Jingrui and Tong, Hanghang},
  journal={arXiv preprint arXiv:2601.03515},
  year={2026}
}
```

---
## 💐 Acknowledgement
The benchmark architecture of Mem-Gallery for baselines is based on the **helpful open-source library [MemEngine](https://github.com/nuster1128/MemEngine)**. Thanks for their pioneering work!
