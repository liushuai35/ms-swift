#!/usr/bin/env bash
# modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /mnt/PublicStorageNew1/liushuai/models
# Note: Currently loaded model is based on /mnt/PublicStorageNew1/liushuai/models directory
set -euo pipefail
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Model and cache paths
MODEL_PATH=${MODEL_PATH:-"/home/shuai.liu01/merged_qwen3_4b_with_special_token"}
PORT=${PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen3-Seg-4B"}  # Model name exposed by vLLM API
TP=${TP:-4}  # tensor parallel size (default: 2 GPUs; use TP=1 for single GPU)
MAX_LEN=${MAX_LEN:-4096}  # 适当调小上下文长度以减压（如仍 OOM 可降到 2048）
# GPU_UTIL 的作用：控制 vLLM 的 --gpu-memory-utilization（0~1），表示每块 GPU 可使用的显存比例。
# - 取值越大：KV Cache 与模型状态可用空间更充裕，吞吐更高，但 OOM 风险更大；
# - 取值越小：更保守，峰值显存更低，适合多模态大图/高并发场景；
# - 建议范围：0.70~0.85；遇到 OOM/fragmentation 可先降到 0.70，再配合减少 BATCH_TOKENS / MAX_LEN / --max-num-seqs。
# - 交互影响：与 TP（tensor-parallel）一起决定每卡的 KV Cache/模型分片大小；同时受 --max-model-len 与 --max-num-batched-tokens 的影响。
# vLLM 中 --gpu-memory-utilization (GPU_UTIL) 数值越小 → 预留的显存越多(缓冲区专门用来接住推理时的瞬时显存峰值，所以彻底杜绝 OOM；) → 越不容易 OOM
# GPU_UTIL = vLLM在单张GPU上的【显存使用上限】

GPU_UTIL=${GPU_UTIL:-0.75}  # 默认 0.75，必要时再调低
ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-false}  # 关闭 torch.compile 以避免长时间编译
ENABLE_CPU_OFFLOAD=${ENABLE_CPU_OFFLOAD:-false}  # 默认关闭，稳定后再开启
BATCH_TOKENS=${BATCH_TOKENS:-128}  # 进一步调小单批 token 数以降低峰值显存

# Redirect caches to /dataset2/shuai.liu/qwen3_logs (704GB free) to avoid root disk full
export TRITON_CACHE_DIR="/dataset2/shuai.liu/qwen3_logs/triton_cache"
export HF_HOME="/dataset2/shuai.liu/qwen3_logs/huggingface"
export VLLM_CACHE_DIR="/dataset2/shuai.liu/qwen3_logs/vllm_cache"
export TMPDIR="/dataset2/shuai.liu/qwen3_logs/tmp"  # Redirect /tmp for CUDA compilation
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$VLLM_CACHE_DIR" "$TMPDIR"

# 设 --max-model-len=4096，--max-num-batched-tokens=256，--max-num-seqs=8。当来两条各 1200 Token 的请求时，调度器会把每条的预填充切成若干 256 的片段分多步完成；上下文仍可达 4096，但每步只处理最多 256 个 Token（两条加总），峰值显存更低。

# Launch vLLM server for Qwen3-VL-8B-Instruct
# Use full path to vllm to avoid PATH issues with sudo
echo "Using cache dirs: TRITON=$TRITON_CACHE_DIR, HF=$HF_HOME, VLLM=$VLLM_CACHE_DIR"
echo "Model: $MODEL_PATH, TP=$TP, PORT=$PORT"
vllm serve "$MODEL_PATH" \
	--host 0.0.0.0 \
	--served-model-name "$SERVED_MODEL_NAME" \
	--dtype bfloat16 \
	--tensor-parallel-size "$TP" \
	--gpu-memory-utilization "$GPU_UTIL" \
	--max-model-len "$MAX_LEN" \
	--max-num-seqs 4 \
	--port "$PORT" \
	--trust-remote-code \
	--enforce-eager \
	$([ "$ENABLE_CPU_OFFLOAD" = "true" ] && echo "--enable-chunked-prefill") \
	--max-num-batched-tokens "$BATCH_TOKENS" 

# --max-model-len "$MAX_LEN"。作用：设置模型的 最大上下文窗口长度（Context Length）
# --max-num-seqs 8。Maximum Number of Sequences，最大并发序列数
# --max-num-batched-tokens 256。全称：Maximum Number of Batched Tokens，单批次最大 Token 数，vLLM 高性能核心参数。单次调度步里“所有并发序列合计”的最大 Token 数上限。它是调度器的吞吐/内存阈值，用来限制本次波次中要处理的总 Token 量；超过则拆成多波次（Chunked Prefill）。
# Example curl test (uncomment to verify)
# curl http://127.0.0.1:${PORT}/v1/models
# Usage examples:
# ./serve.sh                                      # Default: TP=2 (2 GPUs), PORT=8000
# TP=1 ./serve.sh                                 # Use single GPU instead
# CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 ENABLE_CPU_OFFLOAD=true ./serve.sh         # Enable CPU offloading for lower VRAM
# CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 ./serve.sh         # Enable CPU offloading for lower VRAM
# curl http://127.0.0.1:8000/v1/models

# modelscope download --model Qwen/Qwen3-VL-32B-Instruct --local_dir /dataset2/shuai.liu/models --revision master
# modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /dataset2/shuai.liu/qwen3-8b --revision master