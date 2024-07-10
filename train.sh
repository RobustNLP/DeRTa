# Multi-nodes are also supported

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=en,eth,em,bond
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29510}"

export WANDB_DISABLED=true


train_path=run_files/run_clm_lora_derta_llama.py
valid_path=data/train/example_data_to_read.json


root_path_model=llms/
models=(Meta-Llama-3-70B)
datasets=(llama_derta llama_vanilla llama_recaug)


for model in ${models[@]}
do
  model_path=${root_path_model}${model}
  for data in ${datasets[@]}
  do
    model_save=saved_model/lora_${model}_${data}
    train_file=data/train/${data}.json

    torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
        --master_addr $MASTER_ADDR --master_port 6666  \
        ${train_path} \
        --deepspeed train_config/deepspeed_config.json \
        --model_name_or_path ${model_path} \
        --train_file ${train_file} \
        --use_lora True \
        --lora_config train_config/lora_config.json \
        --validation_file ${valid_path} \
        --preprocessing_num_workers 16 \
        --dataloader_num_workers 0 \
        --dataloader_pin_memory True \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 2 \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 15 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.00 \
        --logging_steps 10 \
        --block_size 512 \
        --do_train \
        --evaluation_strategy "no" \
        --bf16 True \
        --streaming \
        --ddp_timeout 3600 \
        --seed 1 \
        --gradient_checkpointing True \
        --output_dir ${model_save} \
        --overwrite_output_dir
  done
done






train_path=run_files/run_clm_llms_derta_llama_drop_5_percent.py

valid_path=data/train/example_data_to_read.json


root_path_model=llms/
models=(Meta-Llama-3-8B)
datasets=(llama_derta llama_vanilla llama_recaug)

for model in ${models[@]}
do
  model_path=${root_path_model}${model}
  for data in ${datasets[@]}
  do
    model_save=saved_model/${model}_${data}
    train_file=data/train/${data}.json

    torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
        --master_addr $MASTER_ADDR --master_port 6666  \
        ${train_path} \
        --deepspeed train_config/deepspeed_config.json \
        --model_name_or_path ${model_path} \
        --train_file ${train_file} \
        --validation_file ${valid_path} \
        --preprocessing_num_workers 16 \
        --dataloader_num_workers 0 \
        --dataloader_pin_memory True \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 2 \
        --save_strategy "steps" \
        --save_steps 5000 \
        --lr_scheduler_type "cosine" \
        --save_total_limit 15 \
        --learning_rate 2e-5 \
        --weight_decay 2e-5 \
        --warmup_ratio 0.03 \
        --logging_steps 10 \
        --block_size 1024 \
        --do_train \
        --evaluation_strategy "no" \
        --bf16 True \
        --streaming \
        --ddp_timeout 3600 \
        --seed 1 \
        --gradient_checkpointing True \
        --output_dir ${model_save} \
        --overwrite_output_dir
  done
done


