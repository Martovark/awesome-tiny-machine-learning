# Awesome tiny machine learning projects

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated collection of github projects with tiny code base. Most of them are primarily interesting for educational purposes, but some of them (e.g. **[tinygrad](https://github.com/tinygrad/tinygrad)**) compete with large and complex projects.

<p align="center">
  <img src="assets/tiny-ml.png" width="70%" alt="My Image" />
</p>


<!-- omit from toc -->
## Contents
- [Andrej Karpathy](#andrej-karpathy)
- [Diffusion models](#diffusion)
- [🤗 Huggingface](#huggingface-)
- [Inference engines](#inference)
- [PyTorch Foundation](#pytorch)
- [Reinforcement learning](#rl)
- [Tabular ML](#tabular)
- [ML](#ml)
- [C](#c)
- [Go](#go)
- [Rust](#rust)
- [Python](#python)
- [Low-level](#low-level)

### Andrej Karpathy
- **[cryptos](https://github.com/karpathy/cryptos)** - Pure Python from-scratch zero-dependency implementation of Bitcoin for educational purposes.
- **[llama2.c](https://github.com/karpathy/llama2.c)** - Inference Llama 2 in one file of pure C.
- **[llm.c](https://github.com/karpathy/llm.c)** - LLM training in simple, raw C/CUDA.
- **[micrograd](https://github.com/karpathy/micrograd)** - A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API.
- **[minbpe](https://github.com/karpathy/minbpe)** - Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.
- **[minGPT](https://github.com/karpathy/minGPT)** - A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training.
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** - The simplest, fastest repository for training/finetuning medium-sized GPT.
- **[nano-llama31](https://github.com/karpathy/nano-llama31)** - nanoGPT style version of Llama 3.1.
- **[nanochat](https://github.com/karpathy/nanochat)** - The best ChatGPT that $100 can buy.

### Diffusion models
- **[diffusion-gpt](https://github.com/ash80/diffusion-gpt)** - An annotated implementation of a character-level disrete diffusion model for text generation. Inspired by nanoGPT.
- **[micro_diffusion](https://github.com/SonyResearch/micro_diffusion)** - Micro-budget training of large-scale diffusion models by Sony Research.
- **[minimal-text-diffusion](https://github.com/madaan/minimal-text-diffusion)** - A minimal implementation of diffusion model for text generation. Also contains a basic list of papers/blogs/videos for a deeper dive into diffusion models.

### 🤗 Huggingface
- **[nanotron](https://github.com/huggingface/nanotron)** - Minimalistic large language model 3D-parallelism training.
- **[nanoVLM](https://github.com/huggingface/nanoVLM)** - The simplest, fastest repository for training/finetuning small-sized VLMs.
- **[picotron](https://github.com/huggingface/picotron)** - The minimalist & most-hackable repository for pre-training Llama-like models with 4D Parallelism. It is designed with simplicity and **educational** purposes in mind.
- **[smolagents](https://github.com/huggingface/smolagents)** - A barebones library for agents that think in code.
- **[smol-course](https://github.com/huggingface/smol-course)** - A course on aligning smol models.
- **[smollm](https://github.com/huggingface/smollm)** - Everything about the SmolLM and SmolVLM family of models.

### Inference engines
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** - A lightweight vLLM implementation built from scratch.
- **[tokasaurus](https://github.com/ScalingIntelligence/tokasaurus)** - LLM inference engine optimized for throughput-intensive workloads. On throughput-focused benchmarks, Tokasaurus can outperform vLLM and SGLang by up to 3x+.

### PyTorch Foundation
- **[gpt-fast](https://github.com/meta-pytorch/gpt-fast)** - Simple and efficient pytorch-native transformer text generation. LLaMA like, gptq, tensor parallelism, spec decoding, etc.
- **[LeanRL](https://github.com/meta-pytorch/LeanRL)** - LeanRL is a fork of [CleanRL](https://github.com/vwxyzjn/cleanrl) where hand-picked scripts have been re-written using PyTorch 2 features, mainly [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) and [cudagraphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
- **[segment-anything-fast](https://github.com/meta-pytorch/segment-anything-fast)** -  Segment Anything over 8x using only pure, native PyTorch.

### Reinforcement learning
- **[Mini-R1](https://www.philschmid.de/mini-deepseek-r1)** - Minimal reproduction of DeepSeek R1-Zero. Code built upon [trl](https://github.com/huggingface/trl).
- **[nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment)** - Inspired by [TinyZero](https://github.com/Jiayi-Pan/TinyZero) and [Mini-R1](https://www.philschmid.de/mini-deepseek-r1), but designed to be much simpler, cleaner, and faster, with every line of code visible and understandable.
- **[TinyZero](https://github.com/Jiayi-Pan/TinyZero)** - Minimal reproduction of DeepSeek R1-Zero. Code built upon [verl](https://github.com/volcengine/verl).

### Tabular ML
- **[nanoTabPFN](https://github.com/automl/nanoTabPFN)** - Train your own small [TabPFN](https://github.com/PriorLabs/TabPFN) in less than 500 LOC and a few minutes. The purpose of this repository is to be a good starting point for students and researchers that are interested in learning about how TabPFN works under the hood.

### ML
- **[minimind](https://github.com/jingyaogong/minimind/blob/master/README_en.md)** - Project aims to train a super-small language model MiniMind with only 3 RMB cost and 2 hours, starting completely from scratch.
- **[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)** - The 100 line AI agent that solves GitHub issues or helps you in your command line.
- **[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)** - NanoGPT (124M) in 3 minutes on 8xH100.
- **[modded-nanogpt-rwkv](https://github.com/BlinkDL/modded-nanogpt-rwkv)** - Modified variant of nanoGPT for RWKV.
- **[nano-graphrag](https://github.com/gusye1234/nano-graphrag?tab=readme-ov-file)** - A simple, easy-to-hack GraphRAG implementation.
- **[nanoT5](https://github.com/PiotrNawrot/nanoT5)** - Fast & Simple repository for pre-training and fine-tuning T5-style models.
- **[tinygrad](https://github.com/tinygrad/tinygrad)** - You like pytorch? You like micrograd? You love tinygrad! ❤️
- **[tinyvector](https://github.com/0hq/tinyvector)** - A tiny nearest-neighbor embedding database built with SQLite and Pytorch.

### C
- **[agent-c](https://github.com/bravenewxyz/agent-c)** - A ultra-lightweight AI agent written in C that communicates with OpenRouter API and executes shell commands.
- **[miniaudio](https://github.com/mackron/miniaudio)** - Audio playback and capture library written in C, in a single source file.
- **[nanoMPI](https://github.com/Quentin-Anthony/nanoMPI)** - A minimal MPI Implementation loosely based on OpenMPI. nanoMPI allows beginners to the field of distributed computing to quickly see answers to questions like "how is a ring allreduce implemented?"
- **[printf](https://github.com/mpaland/printf)** - Tiny, fast, non-dependent and fully loaded printf implementation for embedded systems.
- **[tiny-AES-c](https://github.com/kokke/tiny-AES-c)** - Small portable AES128/192/256 in C.

### Go
- **[minikeyvalue](https://github.com/geohot/minikeyvalue)** - A distributed key value store in under 1000 lines.
- **[tinykv](https://github.com/talent-plan/tinykv)** - A course to build distributed key-value service based on TiKV model.

### Rust
- **[microsandbox](https://github.com/microsandbox/microsandbox)** - Self-Hosted Platform for Secure Execution of Untrusted User/AI Code.
- **[mini-redis](https://github.com/tokio-rs/mini-redis)** - Incomplete Redis client and server implementation using Tokio - for learning purposes only.
- **[mini-lsm](https://github.com/skyzh/mini-lsm)** - A course of building an LSM-Tree storage engine (database) in a week.

### Python
- **[tinychain](https://github.com/jamesob/tinychain)** - A pocket-sized implementation of Bitcoin.

### Low-level
- **[tiny-gpu](https://github.com/adam-maj/tiny-gpu)** - A minimal GPU design in Verilog to learn how GPUs work from the ground up.
- **[tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu)** - A minimal tensor processing unit (TPU), inspired by Google's TPU V2 and V1.
