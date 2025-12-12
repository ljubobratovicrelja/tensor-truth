# Tensor Truth

## Overview

This project is a modular framework for building Retrieval-Augmented Generation (RAG) pipelines running entirely on local hardware (RTX 3090Ti).

**Primary Goal:** Reduce hallucination in local LLMs (specifically DeepSeek-R1-Distill versions) by indexing technical documentation (PyTorch, NumPy, etc.) with high precision.

**Core Mechanics:**

  * **Orchestration:** LlamaIndex.
  * **Inference:** Ollama (serving GGUF/ExLlamaV2 models).
  * **Vector Store:** ChromaDB (Persistent).
  * **Retrieval Strategy:** Hierarchical Node Parsing + Auto-Merging Retriever + Cross-Encoder Reranking.

## Architecture

The pipeline uses a "Small-to-Big" retrieval strategy to maximize context window efficiency while maintaining retrieval accuracy.

1.  **Ingestion:** Documents are parsed into parent nodes (2048 tokens) and child nodes (128 tokens).
2.  **Indexing:** Only child nodes are embedded and stored in ChromaDB. Parent nodes are stored in a DocStore key-value store.
3.  **Retrieval:** Query matches child nodes. If enough children of a specific parent are found, they are merged into the parent node.
4.  **Reranking:** Top-k retrieved contexts are re-scored by a Cross-Encoder (BGE-Reranker) before LLM generation.

## Prerequisites

  * **Hardware:** NVIDIA GPU with 24GB+ VRAM recommended (RTX 3090/4090).
  * **System:** Linux/WSL2 or Windows with CUDA toolkit installed.
  * **Ollama:** Must be running as a background service (`ollama serve`).

## Installation

1. **Environment (venv)**
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/WSL)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\activate
```

2.  **Dependencies**

```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-chroma llama-index-readers-file chromadb torch transformers sentence-transformers
```

3.  **Model Pull (Ollama)**

```bash
ollama pull deepseek-r1:32b
```

## Usage

### 1\. Directory Structure

The script expects a flat directory or subdirectory structure for documents.

```text
.
├── chroma_db/              # Generated vector store persistence
├── library_docs/           # DROP RAW FILES HERE (.md, .txt)
├── src/
│   └── pipeline.py         # Main entry point
└── README.md
```

### 2\. Configuration

Modify `src/pipeline.py` globals to switch contexts or models.

```python
# Configuration Constants
DOCS_DIR = "./library_docs/pytorch_v2"  # Source for specific library
COLLECTION_NAME = "pytorch_index"       # ChromaDB collection namespace
LLM_MODEL = "deepseek-r1:32b"           # Ollama model tag
```

### 3\. Execution

Run the pipeline. It handles both indexing (if new data) and querying.

```bash
python src/pipeline.py
```

## Strategy & Customization

### Adding New Contexts (e.g., Internal Codebase)

To index a different dataset without overwriting the previous one:

1.  Change `DOCS_DIR` to point to the new files.
2.  Change `COLLECTION_NAME` to a unique string (e.g., `internal_legacy_code`).
3.  Run the script. ChromaDB will create a new collection alongside the existing ones.

### Modifying Chunking

Adjust `HierarchicalNodeParser` in `build_or_load_index` for different data types.

  * **Code/API Docs:** Use smaller leaf nodes (128 tokens) to isolate function signatures.
  * **Prose/Wiki:** Increase leaf nodes (256-512 tokens) to capture semantic meaning.

### GPU VRAM Management

If OOM errors occur during ingestion or heavy query loads:

1.  **Offload Embeddings:** Set `device="cpu"` in `HuggingFaceEmbedding`.
2.  **Limit Ollama:** Reduce `num_gpu` layers in Ollama or use a smaller quant (Q4\_K\_M).

## Technical Notes

  * **Persistency:** The `chroma_db` folder contains the vector embeddings. The `storage_context` (DocStore) is saved alongside it. **Do not delete this folder** unless you want to re-index everything.
  * **Reranker:** Currently uses `BAAI/bge-reranker-v2-m3`. It is computationally expensive but necessary for code disambiguation.
  * **Locking:** ChromaDB is single-threaded/process locked. Ensure only one script instance accesses the DB at a time.

## ArXiv Paper Fetching

Top 30 Influential Papers

- arXiv:1512.03385 - ResNet: Revolutionized deep learning with residual connections enabling training of very deep networks.
- arXiv:1706.03762 - Attention Is All You Need: Introduced Transformers, fundamentally changing NLP and now vision.
- arXiv:1406.2661 - Generative Adversarial Networks: Created the GAN framework that revolutionized generative modeling.
- arXiv:1810.04805 - BERT: Transformed NLP with bidirectional pre-training and transfer learning.
- arXiv:2010.11929 - Vision Transformer (ViT): Brought Transformers to computer vision, challenging CNN dominance.
- arXiv:1506.02640 - YOLO: Pioneered real-time object detection with single-stage architecture.
- arXiv:1506.01497 - Faster R-CNN: Established the foundation for modern two-stage object detection.
- arXiv:1703.06870 - Mask R-CNN: Extended object detection to instance segmentation seamlessly.
- arXiv:2003.08934 - NeRF: Revolutionized 3D scene representation with neural radiance fields.
- arXiv:2103.00020 - CLIP: Bridged vision and language with contrastive learning at scale.
- arXiv:2006.11239 - DDPM: Established denoising diffusion as the leading generative modeling approach.
- arXiv:1812.04948 - StyleGAN: Achieved unprecedented quality in image synthesis with style-based generation.
- arXiv:1409.1556 - VGGNet: Demonstrated the power of deep, simple architectures with small filters.
- arXiv:1412.6980 - Adam Optimizer: Became the default optimizer for training deep neural networks.
- arXiv:1502.03167 - Batch Normalization: Enabled faster and more stable training of deep networks.
- arXiv:2005.12872 - DETR: Introduced Transformers to object detection with end-to-end learning.
- arXiv:1611.07004 - Pix2Pix: Pioneered conditional GANs for image-to-image translation tasks.
- arXiv:1703.10593 - CycleGAN: Enabled unpaired image translation without paired training data.
- arXiv:1804.02767 - YOLOv3: Refined real-time detection with multi-scale predictions and better accuracy.
- arXiv:1912.04958 - StyleGAN2: Improved image quality and removed artifacts from StyleGAN.
- arXiv:1704.04861 - MobileNets: Made deep learning practical for mobile and embedded devices.
- arXiv:1905.11946 - EfficientNet: Systematically scaled networks for optimal accuracy-efficiency trade-offs.
- arXiv:2304.02643 - Segment Anything Model (SAM): Created a foundation model for image segmentation.
- arXiv:2112.10752 - Latent Diffusion Models: Made diffusion models efficient, enabling Stable Diffusion.
- arXiv:2308.04079 - 3D Gaussian Splatting: Achieved real-time novel view synthesis with explicit representation.
- arXiv:1409.4842 - Inception (GoogLeNet): Introduced multi-scale feature extraction with inception modules.
- arXiv:2201.03545 - ConvNeXt: Modernized CNNs to compete with Transformers using pure convolutions.
- arXiv:2204.06125 - DALL-E 2: Advanced text-to-image generation with CLIP guidance and diffusion.
- arXiv:2201.05989 - Instant NGP: Accelerated neural graphics primitives for real-time rendering.
- arXiv:1911.09070 - EfficientDet: Scaled object detection efficiently with compound scaling and BiFPN.
- arXiv:1807.06521 - CBAM: Convolutional Block Attention Module.
- arXiv:2310.08528 - 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering: Extended 3DGS to dynamic scenes with temporal consistency.
- arXiv:2309.13101 - Deformable 3D Gaussians: Introduced deformation fields for high-fidelity dynamic scene reconstruction.
- arXiv:2311.18102 - Relightable 3D Gaussian: Enabled material decomposition and relighting in Gaussian representations.
- arXiv:2311.12775 - GS-IR (3D Gaussian Splatting for Inverse Rendering): Decomposed materials and lighting for inverse rendering.
- arXiv:2312.03203 - Relightable Gaussian Codec Avatars: Combined relightable avatars with efficient Gaussian representations.
- arXiv:2401.01201 - GaussianShader: Achieved 3D-consistent appearance interpolation with Gaussians under varying lighting.
- arXiv:2403.10123 - 4D-Rotor Gaussian Splatting: Improved dynamic scene modeling with rotational velocity parameterization.
- arXiv:2406.06216 - GaussianForest: Hierarchical 4D Gaussian splatting for dynamic scenes with better scalability.
- arXiv:2312.02121 - PBIR-GS: Physically-Based Inverse Rendering with Gaussian Splatting for material estimation.
- arXiv:2404.03657 - Gaussian-SLAM: Real-time SLAM system using 3D Gaussian representations.

Fetch paper command:

```bash
python src/fetch_paper.py 1512.03385 1706.03762 1406.2661 1810.04805 2010.11929 1506.02640 1506.01497 1703.06870 2003.08934 2103.00020 2006.11239 1812.04948 1409.1556 1412.6980 1502.03167 2005.12872 1611.07004 1703.10593 1804.02767 1912.04958 1704.04861 1905.11946 2304.02643 2112.10752 2308.04079 1409.4842 2201.03545 2204.06125 2201.05989 1911.09070 1807.06521 2310.08528 2309.13101 2311.18102 2311.12775 2312.03203 2401.01201 2403.10123 2406.06216 2312.02121 2404.03657
```