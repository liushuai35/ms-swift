export CUDA_HOME=/home/a205/anaconda3/envs/ls
export PYTHONPATH= /home/a205/anaconda3/envs/ls/bin/python
unset PYTHONPATH

/mnt/PublicStorageNew1/liushuai/dataset/segmented_glm


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
--report_to 'wandb' \
--deepspeed zero1 \
--add_version False \
--output_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/v0-20251126-154155 \
--logging_dir /home/shuai.liu01/ms-swift/output/Qwen3-1.7B/v0-20251126-154155/runs \
--ignore_args_error True
