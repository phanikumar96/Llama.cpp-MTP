# Llama.cpp-MTP


# Qwen3.6-27B MTP — llama.cpp on RunPod H100

Run Qwen3.6-27B with Multi-Token Prediction (MTP) speculative decoding on a RunPod H100 using llama.cpp. MTP drafts 3 tokens per step, giving a significant throughput boost over standard autoregressive inference.

---

## Prerequisites

- RunPod instance with **H100 GPU** (CUDA 12+)
- Hugging Face account with access to [`ggml-org/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-27B-MTP-GGUF)
- HF token with read permissions

---

## Setup

### 1. Clone llama.cpp

```bash
apt-get update && apt-get upgrade -y
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

### 2. Build for CUDA (H100)

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 3. Install Hugging Face Hub

```bash
pip install -U huggingface_hub
```

### 4. Authenticate with Hugging Face

```bash
hf auth login
```

Paste your HF token when prompted.

### 5. Download the Model

```bash
mkdir -p ~/llama.cpp/models
cd ~/llama.cpp/models

hf download ggml-org/Qwen3.6-27B-MTP-GGUF \
  Qwen3.6-27B-MTP-Q8_0.gguf \
  --local-dir .
```

> **Note:** Q8_0 is ~29GB. Make sure your pod has sufficient disk space.

---

## Running the Server

Navigate back to the llama.cpp root first:

```bash
cd ~/llama.cpp
```

### Standard Inference (1 token/step)

```bash
./build/bin/llama-server \
  -m ~/llama.cpp/models/Qwen3.6-27B-MTP-Q8_0.gguf \
  -ngl 999 \
  -c 16384 \
  --flash-attn on \
  --no-mmproj \
  --host 0.0.0.0 \
  --port 8080
```

### MTP Speculative Decoding (3-token draft)

```bash
./build/bin/llama-server \
  -m ~/llama.cpp/models/Qwen3.6-27B-MTP-Q8_0.gguf \
  -ngl 999 \
  -c 16384 \
  --flash-attn on \
  --no-mmproj \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --host 0.0.0.0 \
  --port 8080
```

### MTP Speculative Decoding (3-token draft) with NGRAM

```bash
./build/bin/llama-server \
  -m ~/llama.cpp/models/Qwen3.6-27B-MTP-Q8_0.gguf \
  -ngl 999 \
  -c 16384 \
  --flash-attn on \
  --no-mmproj \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-type ngram-mod \
  --host 0.0.0.0 \
  --port 8080
```


> MTP uses the model's built-in draft heads — no separate draft model needed.

---

## Accessing the UI

In RunPod, open **Port 8080 → HTTP Service** to access the llama.cpp web UI.

The server also exposes an **OpenAI-compatible API** at:

```
http://<your-pod-ip>:8080/v1
```

---

## Monitoring GPU Usage

```bash
watch -n 1 nvidia-smi
```

---

## Key Flags Reference

| Flag | Description |
|---|---|
| `-ngl 999` | Offload all layers to GPU |
| `-c 16384` | Context window size (tokens) |
| `--flash-attn on` | Enable FlashAttention 2 for memory efficiency |
| `--spec-type draft-mtp` | Enable MTP speculative decoding |
| `--spec-draft-n-max 3` | Draft up to 3 tokens per step |

---

## Model Info

| Property | Value |
|---|---|
| Model | Qwen3.6-27B-MTP |
| Quant | Q8_0 |
| Size | ~29GB |
| Context | 16K (adjustable) |
| Source | [ggml-org/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/ggml-org/Qwen3.6-27B-MTP-GGUF) |
