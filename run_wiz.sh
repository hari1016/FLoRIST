#!/bin/bash
set -oe pipefail

# Create a unique log file using timestamp or PID
LOGFILE="run_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > "$LOGFILE" 2>&1

# Activate environment
source florist/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Download TinyLlama model if not already present
if [ ! -f "tinyllama/model.safetensors" ]; then
    echo "Downloading TinyLlama model..."
    mkdir -p tinyllama
    cd tinyllama
    wget -O tinyllama/model.safetensors \
        "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"
    cd ../
fi

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env main.py --global_model 'tinyllama' --data_path  "./data_alpaca" --output_dir './tinyllama-alpaca-homo-75-1-100-florist/' --num_communication_rounds 75 --local_num_epochs 1 --method florist --num_clients 100

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env main.py --global_model 'tinyllama' --data_path  "./data_alpaca" --output_dir './tinyllama-alapca-homo-75-1-100-flora/' --num_communication_rounds 75 --local_num_epochs 1 --method flora --num_clients 100


# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data_alpaca" --output_dir './Llama-3.2-1B-alpaca-homo-75-1-100-florist/' --num_communication_rounds 75 --local_num_epochs 1 --method florist --num_clients 100

# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data_alpaca" --output_dir './Llama-3.2-1B-alapca-homo-75-1-100-flora/' --num_communication_rounds 75 --local_num_epochs 1 --method flora --num_clients 100
=
# python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data_alpaca" --output_dir './Llama-3.2-1B-alpaca-heter-75-1-100-ffa/' --num_communication_rounds 75 --local_num_epochs 1 --method ffa --num_clients 100 --heter True

# python3 main.py --global_model 'tinyllama' --data_path  "./data_wiz" --output_dir './tinyllama-wizard-homo-75-1-100-florist/' --num_communication_rounds 75 --local_num_epochs 1 --method florist --num_clients 100 --threshold 0.90

# python3 main.py --global_model 'tinyllama' --data_path  "./data_wiz" --output_dir './tinyllama-wizard-heter-75-1-100-florist/' --num_communication_rounds 75 --local_num_epochs 1 --method florist --num_clients 100 --threshold 0.90 --heter True

# python3 main.py --global_model 'tinyllama' --data  "./data_wiz" --output_dir './tinyllama-dolly-heter-75-1-100-fedit/' --num_communication_rounds 75 --local_num_epochs 1 --method fedit --num_clients 100 --threshold 0.90 --heter True --zero_padding True
