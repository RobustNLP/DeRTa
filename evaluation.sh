# Multi-nodes are also supported

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=en,eth,em,bond
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29510}"
export WANDB_DISABLED=true


model=llms/Meta-Llama-3-70B
lora_names=(saved_model/lora_Meta-Llama-3-70B_llama_derta)

gpu_num=8
block_size=1024
mode=None
batch=25
for lora_name in "${lora_names[@]}"
do

  python evaluation_lora.py  --model_name_or_path ${model}   --gpus ${gpu_num} --block_size ${block_size}  --batch ${batch} --mode ${mode} --eval_data data/test/helpfulness_gsm8k_500.json --lora_weights ${lora_name}

  python evaluation_lora.py  --model_name_or_path ${model}   --gpus ${gpu_num} --block_size ${block_size}  --batch ${batch} --mode completion_attack --eval_data data/test/safety_completionattack.json --lora_weights ${lora_name}

done




models=(saved_model/Meta-Llama-3-8B_llama_derta)

gpu_num=8
block_size=1024
mode=None
batch=25

for model in "${models[@]}"
do

  python evaluation.py  --model_name_or_path ${model}   --gpus ${gpu_num} --block_size ${block_size}  --batch ${batch} --mode ${mode}  --eval_data data/test/helpfulness_gsm8k_500.json

  python evaluation.py  --model_name_or_path ${model}   --gpus ${gpu_num} --block_size ${block_size}  --batch ${batch} --mode completion_attack --eval_data data/test/safety_completionattack.json

done