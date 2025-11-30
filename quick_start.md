export CUDA_HOME=/home/a205/anaconda3/envs/ls
export PYTHONPATH= /home/a205/anaconda3/envs/ls/bin/python
unset PYTHONPATH

/mnt/PublicStorageNew1/liushuai/dataset/segmented_glm

# 模型下載
## 1. 安装（一次就行）
pip install -U modelscope

## 2. 设置国内镜像（官方推荐）
export MODELSCOPE_MIRROR=https://modelscope.cn

## 3. 下载（支持断点续传，Ctrl-C 后重新跑同一条命令会继续下）
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./Qwen2.5-3B-Instruct

# SFT
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
swift sft \
--torch_dtype 'bfloat16' \
--model 'Qwen/Qwen3-1.7B' \
--model_type 'qwen3' \
--template 'qwen3' \
--dataset 'local/segmented_glm' \
--new_special_tokens 'swift/new_special_tokens/token.txt' \
--split_dataset_ratio '0.1' \
--max_length '8192' \
--max_new_tokens '4096' \
--task_type 'causal_lm' \
--lora_rank '16' \
--lora_alpha '64' \
--lora_dtype 'bfloat16' \
--learning_rate '1e-5' \
--truncation_strategy left \
--num_train_epochs '10' \
--gradient_accumulation_steps '16' \
--eval_steps '500' \
--attn_impl 'flash_attention_2' \
--neftune_noise_alpha '0' \
--warmup_ratio 0.05 \
--dataloader_num_workers 4 \
--report_to 'wandb' \
--deepspeed zero1 \
--add_version False \
--output_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/v0-20251126-154155 \
--logging_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/v0-20251126-154155/runs \
--ignore_args_error True


# GRPO
CUDA_VISIBLE_DEVICES=3 \
swift rollout \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --vllm_enable_lora true \
  --vllm_max_lora_rank 16

e2b_11fd48d425a1c2f22f428dbf026395f6ea63df27


E2B_API_KEY=e2b_11fd48d425a1c2f22f428dbf026395f6ea63df27 \
WANDB_API_KEY=352ca31147e360ebc4dc5dcf8b204658f97cfbc1 \
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_code_reward external_code_format \
    --reward_weights 1.0 0.1 \
    --vllm_mode server \
    --use_vllm true \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --dataset 'open-r1/verifiable-coding-problems-python-10k' \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 14 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --report_to wandb

## 單卡
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --reward_funcs accuracy format \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true

## 多卡vLLM
nproc_per_node 比显卡数少一，vLLM默认单独部署于最后一张卡，即卡7

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --report_to 'wandb' \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 7 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true