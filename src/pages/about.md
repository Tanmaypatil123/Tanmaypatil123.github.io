---
layout: ../layouts/AboutLayout.astro
title: "About"
---

Hi, I'm **Tanmay Dipak Patil** — a Machine Learning Engineer with a focus on GPU programming, inference optimization, and diffusion research.

I currently work at **ModelsLab**, where I build and optimize inference interfaces for language, image, and audio models. I spend a lot of my time chasing lower latency, reading papers, and implementing ideas from scratch to go deep on diffusion research.

## What I do

- **ML / AI:** TensorFlow, PyTorch, Transformers, Scikit-learn, Diffusers, Computer Vision
- **GPU Programming:** Triton, Cutlass, Cute-dsl, Hip, Hipkittens, ROCm, MSL, Enigma, Gluon
- **Programming:** Python, JavaScript, Dart
- **Tools:** Git, Docker, Kubernetes, AWS, Django, FastAPI

## Experience

**ModelsLab** — Machine Learning Engineer _(Feb 2024 – Present)_

- Developed inference interfaces for language, image, and audio models, plus services such as realtime chat, voice cloning, and image/video synthesis & editing. Benchmarked approaches for arbitrary model serving in real time and made existing implementations much faster to reduce generation latency.
- Trained and finetuned image generation models using LoRA, DPO, and RLHF, dedicating significant time to research and implementing ideas from scratch.
- Managed GPU deployments and handled multiple major production outages.

**PandasAI** — Software Engineer Intern _(Sep 2023 – Jan 2024)_

- Spearheaded development of open-source online connectors and streamlined pipeline construction for LLMs, enhancing data accessibility and deployment efficiency.
- Contributed to MLOps pipeline development and deployment, curated and optimized datasets for LLM fine-tuning, and designed comprehensive testing scripts for model evaluation.

**Intersense Technologies LLP** — Python Developer Intern _(Feb 2022 – Nov 2022)_

- Built an ML-powered automated offset correction unit for CNC machines on a Raspberry Pi 4, replacing manual offset correction using Python, PyQt5, and network programming.

## Projects

**[Enigma DSL](https://github.com/Klyne-org/Enigma-DSL)** — an MLIR-based GPU kernel compiler

- A Python DSL inspired by NVIDIA's CuTe DSL / CUTLASS, porting its layout algebra (composition, complement, coalesce, zipped divide, Thread–Value layouts) to a new GPU backend via a custom MLIR dialect that lowers to GPU machine code.
- Achieved a 1.09× speedup over handwritten Metal on fused SDPA and 92.6 tok/s single-dispatch Qwen3-0.6B decode on M4.
- Published to PyPI as `enigma-dsl`.

**[ModelQ](https://github.com/ModelsLab/modelq)** — a lightweight, production-ready async task library

- Simplifies development and execution of asynchronous tasks in distributed systems. Inspired by Celery, it provides a clear API for defining, scheduling, and managing background jobs and complex task workflows — handling millions of requests daily in production.

## Education

**CSMSS CSCOE** — BTech in AI and Data Science, CGPA 8.84/10.0 _(July 2021 – June 2024)_

## Achievements

- Open source contributor to trending ML repos including Hugging Face Transformers, Hipkittens, and PandasAI.

## Get in touch

- 📧 [tanmaypatil3151@gmail.com](mailto:tanmaypatil3151@gmail.com)
- 💻 [GitHub](https://github.com/Tanmaypatil123)
