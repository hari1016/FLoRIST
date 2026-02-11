#!/bin/bash
set -oe pipefail

# Create a unique log file using timestamp or PID
LOGFILE="run_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > "$LOGFILE" 2>&1

# Activate environment
source florist/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python3 main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './llama7b-wizard-homo-75-1-100-flex/' --num_communication_rounds 75 --local_num_epochs 1 --method flex --num_clients 100 --local_micro_batch_size 8  --local_batch_size 64

# python3 main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_alpaca" --output_dir './llama7b-alpaca-homo-75-1-100-flex/' --num_communication_rounds 75 --local_num_epochs 1 --method flex --num_clients 100 --local_micro_batch_size 4  --local_batch_size 64

# python3 main.py --global_model 'huggyllama/llama-7b' --data_path  "./data" --output_dir './llama7b-dolly-homo-75-1-100-flex/' --num_communication_rounds 75 --local_num_epochs 1 --method flex --num_clients 100 --local_micro_batch_size 4  --local_batch_size 64

# python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data_alpaca" --output_dir './Llama-3.2-1B-alpaca-heter-75-1-100-ffa/' --num_communication_rounds 75 --local_num_epochs 1 --method ffa --num_clients 100 --heter True

# python3 main.py --global_model 'tinyllama' --data_path  "./data_wiz" --output_dir './tinyllama-wizard-homo-75-1-100-florist/' --num_communication_rounds 75 --local_num_epochs 1 --method florist --num_clients 100 --threshold 0.90

# python3 main.py --global_model 'tinyllama' --data_path  "./data_wiz" --output_dir './tinyllama-wizard-heter-75-1-100-florist/' --num_communication_rounds 75 --local_num_epochs 1 --method florist --num_clients 100 --threshold 0.90 --heter True

# python3 main.py --global_model 'huggyllama/llama-7b' --data_path  "./data" --output_dir './llama7b-dolly-heter-75-1-100-ffa/' --num_communication_rounds 75 --local_num_epochs 1 --method ffa --num_clients 100 --local_micro_batch_size 4  --local_batch_size 64 --heter True --zero_padding True

# python3 main.py --global_model 'huggyllama/llama-7b' --data_path  "./data" --output_dir './llama7b-dolly-heter-75-1-100-fedit/' --num_communication_rounds 75 --local_num_epochs 1 --method fedit --num_clients 100 --local_micro_batch_size 4  --local_batch_size 64 --heter True --zero_padding True


python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data_alpaca" --output_dir './Llama-3.2-1B-alpaca-homo-75-1-100-fedit/' --num_communication_rounds 75 --local_num_epochs 1 --method fedit --num_clients 100

# python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data_alpaca" --output_dir './Llama-3.2-1B-alpaca-homo-75-1-100-flex/' --num_communication_rounds 75 --local_num_epochs 1 --method flex --num_clients 100

# python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data_alpaca" --output_dir './Llama-3.2-1B-alpaca-homo-75-1-100-florist/' --num_communication_rounds 75 --local_num_epochs 1 --method florist --num_clients 100
