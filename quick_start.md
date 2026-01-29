export CUDA_HOME=/home/a205/anaconda3/envs/ls
export PYTHONPATH= /home/a205/anaconda3/envs/ls/bin/python
unset PYTHONPATH

/home/shuai.liu01/data/segmented_glm_v1

# 模型下載
## 1. 安装（一次就行）
pip install -U modelscope

## 2. 设置国内镜像（官方推荐）
export MODELSCOPE_MIRROR=https://modelscope.cn

## 3. 下载（支持断点续传，Ctrl-C 后重新跑同一条命令会继续下）
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./Qwen2.5-3B-Instruct

# SFT
export MASTER_PORT=29510

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
swift sft \
--master_port 23000 \
--torch_dtype 'bfloat16' \
--model '/home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507' \
--model_type 'qwen3' \
--template 'qwen3' \
--dataset 'local/segmented_gemini_v1' \
--new_special_tokens 'swift/new_special_tokens/token.txt' \
--split_dataset_ratio '0.1' \
--max_length '8192' \
--max_new_tokens '4096' \
--task_type 'causal_lm' \
--loss_type 'cross_entropy_special' \
--lora_rank '16' \
--lora_alpha '64' \
--lora_dtype 'bfloat16' \
--target_modules 'all-linear' \
--learning_rate '1e-5' \
--num_train_epochs '30' \
--gradient_accumulation_steps '16' \
--eval_steps '100' \
--attn_impl 'flash_attention_2' \
--neftune_noise_alpha '0' \
--warmup_ratio 0.05 \
--dataloader_num_workers 4 \
--deepspeed zero2 \
--add_version False \
--save_total_limit 2 \
--output_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25 \
--logging_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25/runs \
--ignore_args_error True \
--report_to 'tensorboard' 
--report_to 'wandb' 

# 运行合并命令（替换成你的路径）
https://blog.csdn.net/weixin_28999139/article/details/157075487

swift export \
  --adapters /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25/checkpoint-660 \
  --model /home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
  --output_dir /home/shuai.liu01/merged_qwen3_4b_with_special_token \
  --new_special_tokens swift/new_special_tokens/token.txt \
  --merge_lora true \
  --torch_dtype bfloat16

#必须带你的特殊token文件

<!-- --truncation_strategy left \ -->

CUDA_VISIBLE_DEVICES=0,1,2 NPROC_PER_NODE=3 \
swift sft \
--port 23000 \
--torch_dtype 'bfloat16' \
--model '/home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507' \
--model_type 'qwen3' \
--template 'qwen3' \
--dataset 'local/segmented_gemini_v1' \
--new_special_tokens 'swift/new_special_tokens/token.txt' \
--split_dataset_ratio '0.1' \
--max_length '8192' \
--max_new_tokens '4096' \
--task_type 'causal_lm' \
--loss_type 'cross_entropy_special' \
--lora_rank '16' \
--lora_alpha '64' \
--lora_dtype 'bfloat16' \
--target_modules 'all-linear' \
--learning_rate '1e-5' \
--num_train_epochs '30' \
--gradient_accumulation_steps '16' \
--eval_steps '100' \
--attn_impl 'flash_attention_2' \
--neftune_noise_alpha '0' \
--warmup_ratio 0.05 \
--dataloader_num_workers 4 \
--deepspeed zero2 \
--add_version False \
--save_total_limit 1 \
--output_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25 \
--logging_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25/runs \
--ignore_args_error True \
--report_to 'tensorboard' 


CUDA_VISIBLE_DEVICES=0,1,2 NPROC_PER_NODE=3 \
swift sft \
--port 23000 \
--torch_dtype 'bfloat16' \
--model '/home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507' \
--model_type 'qwen3' \
--template 'qwen3' \
--dataset 'local/segmented_gemini_v1' \
--split_dataset_ratio '0.1' \
--max_length '8192' \
--max_new_tokens '4096' \
--task_type 'causal_lm' \
--loss_type 'cross_entropy' \
--lora_rank '16' \
--lora_alpha '64' \
--lora_dtype 'bfloat16' \
--target_modules 'all-linear' \
--learning_rate '1e-5' \
--num_train_epochs '30' \
--gradient_accumulation_steps '16' \
--eval_steps '100' \
--attn_impl 'flash_attention_2' \
--neftune_noise_alpha '0' \
--warmup_ratio 0.05 \
--dataloader_num_workers 4 \
--deepspeed zero2 \
--add_version False \
--save_total_limit 1 \
--output_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25 \
--logging_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25/runs \
--ignore_args_error True \
--report_to 'tensorboard' 

# GRPO
CUDA_VISIBLE_DEVICES=3 \
swift rollout \
    --model /home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 --max_model_len 8192


E2B_API_KEY=e2b_11fd48d425a1c2f22f428dbf026395f6ea63df27 \
WANDB_API_KEY=352ca31147e360ebc4dc5dcf8b204658f97cfbc1 \
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model /home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
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
    --dataset /home/shuai.liu01/.cache/modelscope/hub/datasets/open-r1/verifiable-coding-problems-python-10k \
    --load_from_cache_file true \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 16 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_model_len 8192 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --report_to wandb


E2B_API_KEY=e2b_11fd48d425a1c2f22f428dbf026395f6ea63df27 \
WANDB_API_KEY=352ca31147e360ebc4dc5dcf8b204658f97cfbc1 \
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model /home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
    --reward_funcs 'segment_flag' \
    --reward_weights 1.0 \
    --vllm_mode server \
    --use_vllm true \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --dataset 'local/segmented_glm_v1_grpo' \
    --load_from_cache_file true \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 16 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_model_len 8192 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true 

## 單卡
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Coder-1.5B \
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
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Coder-1.5B \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_mode colocate \
    --sleep_level 1 \
    --offload_optimizer true \
    --offload_model true \
    --vllm_enable_lora true \
    --vllm_max_lora_rank 8 \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 2048 \
    --torch_dtype bfloat16 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --max_completion_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --report_to 'wandb' \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true

## SFT 调用流程（便于排查）
入口
- `swift` 可执行 -> `swift/cli/sft.py` 解析 CLI -> `sft_main(...)`
- `swift/llm/train/sft.py` : `sft_main` -> `SwiftSft(args).main()` -> `run()` -> `train(trainer)` -> `trainer.train(...)`

准备阶段
- `SwiftPipeline.__init__` : parse_args / seed -> `SwiftPipeline.main` 包装日志时间
- `SwiftSft.__init__`
    - `_prepare_model_tokenizer` -> `args.get_model_processor` (模型+tokenizer/processor) -> sequence_parallel(可选) -> `_prepare_generation_config`
    - `_prepare_template` -> `args.get_template` 设置 train 模式，校验多模态/packing
    - `_prepare_callbacks` -> LISA / Adapter / extra_callbacks / EarlyStop(可选)

数据阶段
- `_prepare_dataset` :
    - 若 cached_dataset -> `_get_cached_dataset` (load_from_disk)
    - 若 dataset -> `_get_dataset` (load_dataset + split/shuffle)
    - 合并多个源 -> `DatasetLoader._concat_datasets`
    - `_encode_dataset` (EncodePreprocessor 预编码，非 streaming/lazy)；`_post_process_datasets` (LazyLLMDataset / PackingDataset / streaming 编码)；`_show_dataset` 打印样例与长度统计
- `_get_data_collator` -> `template.data_collator`

模型/调优阶段
- `prepare_model` (TunerMixin) : 按 `train_type` 选择 Swift/PEFT/Unsloth/LongLoRA/adalora/vera/boft 等；可应用 Liger kernel、GaLore、冻结/解冻、resume/adapters 加载；修正 fp16 可训练参数

Trainer 与训练阶段
- `TrainerFactory.get_trainer_cls` -> 实例化 trainer (model, training_args, data_collator, train/val, callbacks, template)
- `trainer.train(resume_from_checkpoint)` 开训；混入逻辑在 `swift/trainers/...`

保存/输出阶段
- `_save_trainer_state` : 处理 last/best ckpt（可建符号链接），push_to_hub(可选)，绘图，写 `logging.jsonl`，返回 best_metric 等
- `SwiftPipeline.main` 记录结束时间

# 合并模型

swift export \
    --model /home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
    --adapters /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25/checkpoint-450 \
    --output_dir /home/shuai.liu01/merged_qwen3_4b_with_special_token \
    --merge_lora true


# 运行时合并（Inference-time Merge）

swift infer \
    --adapters /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/2025-12-25/checkpoint-450 \
    --merge_lora true \
    --infer_backend vllm \
    --max_new_tokens 2048