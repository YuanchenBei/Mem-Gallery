# 🌉 Benchmark Introduction

This directory contains the benchmark framework for evaluating multimodal long-term conversational memory systems.

## 📂 Directory Structure

```
benchmark/
├── default_config/          # Default configuration files
├── memengine/               # Core memory engine modules
├── prompt/                  # Prompt templates for evaluation
├── run/                     # Benchmark execution scripts
└── README.md
```

## 🎩 Components

### default_config/

Configuration files for various memory systems and components.

### memengine/

Core memory engine implementation with the following submodules:

#### memory/
Memory system implementations:
- `BaseMemory.py` - Base memory class
- `FUMemory.py` - Full-context memory
- `STMemory.py` - FIFO memory
- `LTMemory.py` - NaiveRAG
- `GAMemory.py` - Generative Agent memory [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3586183.3606763)
- `MGMemory.py` - MemGPT memory [[Paper]](https://par.nsf.gov/servlets/purl/10524107)
- `RFMemory.py` - Reflexion memory [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)
- `MMMemory.py` - MuRAG memory [[Paper]](https://aclanthology.org/2022.emnlp-main.375.pdf)
- `MMFUMemory.py` - Multimodal full-context memory
- `NGMemory.py` - NGM memory [[Paper]](https://www.researchgate.net/profile/Matt-Fisher-7/publication/394440420_Neural_Graph_Memory_A_Structured_Approach_to_Long-Term_Memory_in_Multimodal_Agents/links/689ab8c337b271210509c20f/Neural-Graph-Memory-A-Structured-Approach-to-Long-Term-Memory-in-Multimodal-Agents.pdf)
- `AUGUSTUSMemory.py` - AUGUSTUS memory [[Paper]](https://arxiv.org/pdf/2510.15261)
- `UniversalRAGMemory.py` - UniversalRAG memory [[Paper]](https://arxiv.org/pdf/2504.20734)

For evaluations of A-Mem and MemoryOS, you can modify the official example code based on our run/run_bench.py ​​file.

- A-Mem: https://github.com/WujiangXu/A-mem/blob/main/test_advanced.py
- MemoryOS: https://github.com/BAI-LAB/MemoryOS/blob/main/eval/evalution_loco.py

#### function/
Functional components for memory operations:
- `Encoder.py` / `MultiModalEncoder.py` - Text and multimodal encoding
- `Retrieval.py` / `MultiModalRetrieval.py` - Memory retrieval functions
- `ConceptBasedRetrieval.py` - Concept-based retrieval
- `ConceptExtractor.py` / `EntityExtractor.py` / `FactExtractor.py` - Information extraction
- `Truncation.py` - Context truncation strategies
- `Trigger.py` - Memory operation triggers
- `Reflector.py` - Memory reflection
- `Summarizer.py` - Content summarization
- `Utilization.py` - Memory utilization functions
- `Judge.py` - Response judgment
- `LLM.py` - LLM interface
- `Forget.py` - Memory forgetting mechanism
- `UniversalRAGRetrieval.py` / `UniversalRAGRouting.py` - UniversalRAG components

#### operation/
Core memory operations:
- `Store.py` - Memory storage operations
- `Recall.py` - Memory recall operations
- `Reflect.py` - Memory reflection operations
- `Optimize.py` - Memory optimization operations

#### evaluate/
Evaluation modules:
- `evaluation.py` - Evaluation metrics and functions
- `llm_judge.txt` - LLM judge prompt template

#### utils/
Utility modules:
- `Storage.py` / `UniversalRAGStorage.py` - Storage backends
- `Client.py` - API client utilities
- `Display.py` - Output display utilities
- `AutoSelector.py` - Automatic selection utilities

### prompt/

Prompt templates for different evaluation tasks:

| File | Description |
|------|-------------|
| `sys_prompt.txt` | System prompt defining the AI assistant's role and task types |
| `ar_prompt.txt` | Answer Refusal task prompt |
| `cd_prompt.txt` | Conflict Detection task prompt |
| `vs_prompt.txt` | Visual-centric Search task prompt |

### run/

- `run_bench.py` - Main benchmark execution script


## 🚀 Usage

Run the benchmark with:

```bash
cd run
python run_bench.py [options]
```

The script will:
1. Load dialog data from `data/dialog/`
2. Process images from `data/image/`
3. Execute memory operations using the specified memory system
4. Output results to the `result/` directory
