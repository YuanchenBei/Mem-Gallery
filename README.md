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


🔧 **TODO**

⭐️ We are updating the project:
- [ ] Benchmark Code 
- [x] Result Evaluation Code
- [x] Datasets ([Hugging Face](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery))

---
## 🛫 Requirements
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
The conversations with their corresponding evaluation QAs are available at the 🤗 [Hugging Face](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery).




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
## 💐 Acknowledgements
The benchmark architecture of Mem-Gallery for baselines is based on the **helpful open-source library [MemEngine](https://github.com/nuster1128/MemEngine)**. Thanks for their pioneering work!
