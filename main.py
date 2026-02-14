import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    AdaLoraConfig,
    AdaLoraModel,
    set_peft_model_state_dict
)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
import numpy as np
import random
import copy
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.nn.functional import normalize
import sys
sys.stdout.flush()
import functools

# Override the built-in print() function
print = functools.partial(print, flush=True)
import transformers
print(transformers.__version__)
print("CUDA available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())


print(f"RANK: {os.environ.get('RANK')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
HF_TOKEN = os.environ.get("HF_TOKEN")


def fl_finetune(
        # model/data params
        global_model: str = 'huggyllama/llama-7b',
        data_path: str = './data',
        output_dir: str = './fedgpt-llama7b-5-2/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 5,
        num_clients: int = 100,
        # Local training hyperparams
        local_batch_size: int = 128,  # 64,
        local_micro_batch_size: int = 16,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        # aggregation mode
        method: str = "fedit",
        threshold: int = 0.9,
        # evaluation
        dev_data_path: str = './mmlu_test_1444.jsonl',
        # heterogeneous
        heter: bool = False,
        local_ranks: List[int] = [64, 64, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4],
        zero_padding: bool = False,
        Adalora: bool = False,
        full: bool = False
):
    method = method.lower()
    valid_methods = {"fedit", "flora", "flex", "ffa", "florist"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Choose from: {sorted(valid_methods)}")

    is_flora = method == "flora"
    is_flex = method == "flex"
    is_ffa = method == "ffa"
    is_florist = method == "florist"

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"threshold: {threshold}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"method: {method}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    clients_per_round = num_clients * client_selection_frac
    # Define the number of clients per rank
    rank_distribution = {
        64: 10,
        32: 10,
        16: 20,
        8: 20,
        4: 40
    }

    # Create the array
    client_ranks = []
    for rank, count in rank_distribution.items():
        client_ranks.extend([int(rank)] * int(count))
    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print("DDP is activated")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:
        device_map = "auto"
    # def create_model(model):
    if global_model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            use_auth_token=True,
        )
    elif global_model == 'meta-llama/Llama-3.2-1B' or global_model == 'meta-llama/Llama-2-7b-hf':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token=HF_TOKEN,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token=HF_TOKEN,
            )
    #     return model
    # model = create_model(model)
    if global_model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b' or global_model == 'meta-llama/Llama-3.2-1B' or global_model == 'meta-llama/Llama-2-7b':
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=HF_TOKEN)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(global_model, token=HF_TOKEN)

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if data_path == './data/8' or data_path == './data/100':
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["context"],
                data_point["response"],
            )
        elif data_path == './data_wiz/100' or data_path == './data_wiz/16' or data_path == './data_wiz/8' or data_path == './data_wiz/4' or data_path == './data_wiz/2':
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["output"],
            )
        else:
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]
        return tokenized_full_prompt

    #model = prepare_model_for_int8_training(model)

    print("Number of available GPUs:")
    print(f"Rank: {os.environ.get('RANK')}, Node GPUs: {torch.cuda.device_count()}")


    if not ddp and torch.cuda.device_count() > 1:
        print("***********")
        model.is_parallelizable = True
        model.model_parallel = True

    def homogenize_lora_weights(new_weights, target_rank):
        adjusted_weights = {}
        for key, tensor in new_weights.items():
            if "lora_A" in key:
                # Shape: [r, in_features]
                r, in_features = tensor.shape
                if r < target_rank:
                    pad = torch.zeros(target_rank - r, in_features, dtype=tensor.dtype, device=tensor.device)
                    adjusted = torch.cat([tensor, pad], dim=0)
                else:
                    adjusted = tensor[:target_rank, :]
            elif "lora_B" in key:
                # Shape: [out_features, r]
                out_features, r = tensor.shape
                if r < target_rank:
                    pad = torch.zeros(out_features, target_rank - r, dtype=tensor.dtype, device=tensor.device)
                    adjusted = torch.cat([tensor, pad], dim=1)
                else:
                    adjusted = tensor[:, :target_rank]
            else:
                continue
            adjusted_weights[key] = adjusted
        return adjusted_weights

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    acc_list = []
    total_rank_comm = 0
    total_param_comm = 0
    previous_set = []
    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)
        local_ranks = [int(client_ranks[i]) for i in selected_clients_set]
        print(local_ranks)

        if full == False:
            if is_florist or is_flex:
                config = LoraConfig(
                        base_model_name_or_path=global_model,
                        r = lora_r,
                        lora_alpha = lora_alpha,
                        target_modules = lora_target_modules,
                        lora_dropout = lora_dropout,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                    )
                # if epoch == 0:
                #     model = get_peft_model(model, config)
            elif is_flora:
                if heter:
                    config_ori = LoraConfig(
                        base_model_name_or_path=global_model,
                        r = sum(local_ranks),
                        lora_alpha = lora_alpha * int(clients_per_round),
                        target_modules = lora_target_modules,
                        lora_dropout = lora_dropout,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                )
                else:
                    config_ori = LoraConfig(
                        base_model_name_or_path=global_model,
                        r = lora_r * int(clients_per_round),
                        lora_alpha = lora_alpha * int(clients_per_round),
                        target_modules = lora_target_modules,
                        lora_dropout = lora_dropout,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                    )
            else:
                if zero_padding:
                    config_ori = LoraConfig(
                        base_model_name_or_path=global_model,
                        r = max(local_ranks),
                        lora_alpha = lora_alpha * max(local_ranks),
                        target_modules = lora_target_modules,
                        lora_dropout = lora_dropout,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                    )
                    # if epoch == 0:
                    #     model = get_peft_model(model, config_ori)
                else:
                    config = LoraConfig(
                        base_model_name_or_path=global_model,
                        r = lora_r,
                        lora_alpha = lora_alpha,
                        target_modules = lora_target_modules,
                        lora_dropout = lora_dropout,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                    )
                    if epoch == 0:
                        model = get_peft_model(model, config)

        for k, client_id in enumerate(selected_clients_set):
            if full == False:
                if Adalora:
                    config = AdaLoraConfig(
                        r=local_ranks[k],
                        lora_alpha=2*local_ranks[k],
                        target_modules=lora_target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                        base_model_name_or_path=global_model,
                    )
                    model_client = copy.deepcopy(model)
                    model_client = get_peft_model(model_client, config)
                else:
                    if is_flora or is_florist or is_flex:
                        if heter:
                            config = LoraConfig(
                            r=int(local_ranks[k]),
                            lora_alpha=2*int(local_ranks[k]),
                            target_modules=lora_target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            base_model_name_or_path=global_model,
                            )
                            print("local_rank ", local_ranks[k])

                        else:
                            config = LoraConfig(
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            target_modules=lora_target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            base_model_name_or_path=global_model,
                            )

                        if is_flora:
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                        elif is_florist:
                            if epoch == 0:
                                if global_model == 'meta-llama/Llama-3.2-1B':
                                    model = AutoModelForCausalLM.from_pretrained(
                                        global_model,
                                        load_in_8bit=False,
                                        torch_dtype=torch.float32,
                                        device_map=device_map,
                                        token=HF_TOKEN,
                                    )
                                else:
                                    model = LlamaForCausalLM.from_pretrained(
                                        global_model,
                                        load_in_8bit=False,
                                        torch_dtype=torch.float32,
                                        device_map=device_map,
                                        token=HF_TOKEN,
                                    )
                                model = get_peft_model(model, config)
                            model_client = model
                            if epoch>0:
                                out_path = os.path.join(output_dir, str(epoch-1), 'florist')
                                new_weights = torch.load(os.path.join(output_dir, str(epoch-1), 'florist', 'adapter_model.bin'))
                                if heter:
                                    adjusted_weights = homogenize_lora_weights(new_weights, local_ranks[k])
                                else:
                                    adjusted_weights = homogenize_lora_weights(new_weights, lora_r)

                                config.save_pretrained(
                                    os.path.join(out_path),
                                    load_in_8bit=False,
                                    torch_dtype=torch.float16,
                                    device_map=device_map,
                                )
                                torch.save(adjusted_weights, os.path.join(out_path, "adapter_model.bin"))
                                if global_model == 'meta-llama/Llama-3.2-1B':
                                    temp_model = AutoModelForCausalLM.from_pretrained(
                                        global_model,
                                        load_in_8bit=False,
                                        torch_dtype=torch.float32,
                                        device_map=device_map,
                                        token=HF_TOKEN,
                                    )
                                else:
                                    temp_model = LlamaForCausalLM.from_pretrained(
                                        global_model,
                                        load_in_8bit=False,
                                        torch_dtype=torch.float32,
                                        device_map=device_map,
                                        token=HF_TOKEN,
                                    )
                                model_client = PeftModel.from_pretrained(temp_model, out_path, is_trainable = True)

                                print("Global adapters loaded successfully")
                                # print(torch.cuda.memory_summary())
                        elif is_flex:
                            if heter:
                                if epoch == 0 and k == 0:
                                    model = get_peft_model(model, config)
                                model_client = model
                                if epoch>0:
                                    out_path = os.path.join(output_dir, str(epoch), 'flex-old', str(client_id))
                                    os.makedirs(out_path,exist_ok=True)
                                    new_weights = torch.load(os.path.join(output_dir, str(epoch-1), 'flex', 'global', 'adapter_model.bin'))
                                    adjusted_weights = homogenize_lora_weights(new_weights, local_ranks[k])
                                    torch.save(adjusted_weights, os.path.join(out_path, "adapter_model.bin"))
                                    config.save_pretrained(
                                        os.path.join(out_path),
                                        load_in_8bit=False,
                                        torch_dtype=torch.float16,
                                        device_map=device_map,
                                    )
                                    if global_model == 'meta-llama/Llama-3.2-1B':
                                        temp_model = AutoModelForCausalLM.from_pretrained(
                                            global_model,
                                            load_in_8bit=False,
                                            torch_dtype=torch.float32,
                                            device_map=device_map,
                                            token=HF_TOKEN,
                                        )
                                    else:
                                        temp_model = LlamaForCausalLM.from_pretrained(
                                            global_model,
                                            load_in_8bit=False,
                                            torch_dtype=torch.float32,
                                            device_map=device_map,
                                            token=HF_TOKEN,
                                        )
                                    model_client = PeftModel.from_pretrained(temp_model, out_path, is_trainable = True)

                                    print(f"Client{previous_set[k]} weights successfully loaded into the model for training.")
                                    # print(torch.cuda.memory_summary())
                            else:
                                if epoch == 0:
                                    if global_model == 'meta-llama/Llama-3.2-1B':
                                        model = AutoModelForCausalLM.from_pretrained(
                                            global_model,
                                            load_in_8bit=False,
                                            torch_dtype=torch.float32,
                                            device_map=device_map,
                                            token=HF_TOKEN,
                                        )
                                    else:
                                        model = LlamaForCausalLM.from_pretrained(
                                            global_model,
                                            load_in_8bit=False,
                                            torch_dtype=torch.float32,
                                            device_map=device_map,
                                            token=HF_TOKEN,
                                        )
                                    model = get_peft_model(model, config)
                                model_client = model

                    else:
                        if heter:
                            out_path = os.path.join(output_dir, str(epoch-1), 'fedit')
                            config = LoraConfig(
                                r=local_ranks[k],
                                lora_alpha=2*local_ranks[k],
                                target_modules=lora_target_modules,
                                lora_dropout=lora_dropout,
                                bias="none",
                                task_type="CAUSAL_LM",
                                base_model_name_or_path=global_model,
                                )
                            if epoch == 0 and k==0:
                                model = get_peft_model(model, config)

                            model_client = model
                            if epoch > 0:
                                new_weights = torch.load(os.path.join(output_dir, str(epoch-1), 'fedit', 'adapter_model.bin'))
                                client_path = os.path.join(output_dir, str(epoch), 'fedit-client', str(client_id))
                                os.makedirs(client_path,exist_ok=True)
                                adjusted_weights = homogenize_lora_weights(new_weights, local_ranks[k])
                                torch.save(adjusted_weights, os.path.join(client_path, "adapter_model.bin"))
                                config.save_pretrained(
                                    os.path.join(client_path),
                                    load_in_8bit=False,
                                    torch_dtype=torch.float16,
                                    device_map=device_map,
                                )
                                if global_model == 'meta-llama/Llama-3.2-1B':
                                    temp_model = AutoModelForCausalLM.from_pretrained(
                                        global_model,
                                        load_in_8bit=False,
                                        torch_dtype=torch.float32,
                                        device_map=device_map,
                                        token=HF_TOKEN,
                                    )
                                else:
                                    temp_model = LlamaForCausalLM.from_pretrained(
                                        global_model,
                                        load_in_8bit=False,
                                        torch_dtype=torch.float32,
                                        device_map=device_map,
                                        token=HF_TOKEN,
                                    )
                                model_client = PeftModel.from_pretrained(temp_model, client_path, is_trainable = True)
                            # else:
                            #     model_client = copy.deepcopy(model)
                            #     model_client = get_peft_model(model_client, config)
                        else:
                            # model_client = copy.deepcopy(model)
                            # model_client = get_peft_model(model_client, config)
                            model_client = model

                        if is_ffa:
                            for name, param in model_client.named_parameters():
                                if 'q_proj' in name or 'v_proj' in name:
                                    if 'lora_A' in name:
                                        torch.manual_seed(42)
                                        # Reinitialize using a normal distribution
                                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
                                        # Freeze B
                                        param.requires_grad = False

            else:
                model_client = model
            client = GeneralClient(client_id, model_client, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        previous_set = list(selected_clients_set)
        print("Collecting the weights of clients and performing aggregation")

        model, total_rank, total_param = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       is_flora,
                       is_florist,
                       is_flex,
                       is_ffa,
                       threshold,
                       lora_r,
                       heter,
                       local_ranks,
                       zero_padding,
                       full
                       )
        total_rank_comm += total_rank
        total_param_comm += total_param

        if full == False:
            if is_flora:
                config_ori.save_pretrained(
                    os.path.join(output_dir, str(epoch), 'flora'),
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
                model = PeftModel.from_pretrained(model, os.path.join(output_dir, str(epoch), 'flora'))
            elif is_flex:
                if not heter:
                    out_path = os.path.join(output_dir, str(epoch), 'flex', str(previous_set[0]))
                    os.makedirs(out_path,exist_ok=True)
                    # torch.save(model.state_dict(), os.path.join(out_path, "adapter_model.bin"))
                    config.save_pretrained(
                        os.path.join(out_path),
                        load_in_8bit=False,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                    )
                    if global_model == 'meta-llama/Llama-3.2-1B':
                        temp_model = AutoModelForCausalLM.from_pretrained(
                            global_model,
                            load_in_8bit=False,
                            torch_dtype=torch.float32,
                            device_map=device_map,
                            token=HF_TOKEN,
                        )
                    else:
                        temp_model = LlamaForCausalLM.from_pretrained(
                            global_model,
                            load_in_8bit=False,
                            torch_dtype=torch.float32,
                            device_map=device_map,
                            token=HF_TOKEN,
                        )
                    model = PeftModel.from_pretrained(temp_model, out_path, is_trainable = True)
                    print("")
            elif is_florist:
                out_path = os.path.join(output_dir, str(epoch), 'florist')
                new_weights = torch.load(os.path.join(out_path, 'adapter_model.bin'))
                for name, param in model.named_parameters():
                    key = name.replace(".default", "")
                    param_device = param.device
                    print(key)
                    if key in new_weights:
                        print("Yes")
                        param.data = new_weights[key]
                        param.data = param.data.to(param_device)
                print("Client weights successfully loaded into the model.")
            else:
                out_path = os.path.join(output_dir, str(epoch), 'fedit')
                if not heter:
                    os.makedirs(out_path,exist_ok=True)
                    # torch.save(model.state_dict(), os.path.join(out_path, "adapter_model.bin"))
                    config.save_pretrained(
                        os.path.join(out_path),
                        load_in_8bit=False,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                    )
                    if global_model == 'meta-llama/Llama-3.2-1B':
                        temp_model = AutoModelForCausalLM.from_pretrained(
                            global_model,
                            load_in_8bit=False,
                            torch_dtype=torch.float32,
                            device_map=device_map,
                            token=HF_TOKEN,
                        )
                    else:
                        temp_model = LlamaForCausalLM.from_pretrained(
                            global_model,
                            load_in_8bit=False,
                            torch_dtype=torch.float32,
                            device_map=device_map,
                            token=HF_TOKEN,
                        )
                    model = PeftModel.from_pretrained(temp_model, out_path, is_trainable = True)
                    print("Global adapters loaded")
        else:
            config = AutoConfig.from_pretrained(global_model)
            tokenizer.save_pretrained(os.path.join(output_dir, str(epoch)),
                    load_in_8bit=False,
                    torch_dtype=torch.float32,
                    device_map=device_map,)
            config.save_pretrained(os.path.join(output_dir, str(epoch)),
                    load_in_8bit=False,
                    torch_dtype=torch.float32,
                    device_map=device_map,)
            print('save model')

        if is_flora:
            model = model.merge_and_unload()

        if epoch > 0 and epoch < (num_communication_rounds - 1):
            rm_dir = os.path.join(output_dir, str(epoch-1))
            os.system("rm -rf {xxxxx}".format(xxxxx = rm_dir))
            print("Removed ", rm_dir)

        if (epoch == 0) or (((epoch+1) % 5) == 0):
            if is_flex:
                if heter:
                    client_acc = []
                    weights_array = normalize(
                    torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                                 dtype=torch.float32),
                    p=1, dim=0)
                    for i in selected_clients_set:
                        out_path = os.path.join(output_dir, str(epoch), 'flex', str(i))
                        new_weights = torch.load(os.path.join(out_path, 'adapter_model.bin'))
                        for name, param in model_client.named_parameters():
                            key = name.replace(".default", "")
                            param_device = param.device
                            print(key)
                            if key in new_weights:
                                print("Yes")
                                param.data = new_weights[key]
                                param.data = param.data.to(param_device)
                        print(f"Client{i} weights successfully loaded into the model.")
                        client_acc.append(global_evaluation(model_client, tokenizer, prompter, dev_data_path))
                        # client_acc.append(random.randint(1, 10))
                    client_acc = torch.tensor(client_acc)
                    print("CLient accuracies:", client_acc)
                    acc = torch.sum(weights_array * client_acc)
                    print("avg", acc)
                else:
                    acc = global_evaluation(model, tokenizer, prompter, dev_data_path)

            elif is_florist:
                acc = global_evaluation(model, tokenizer, prompter, dev_data_path)

            else:
                if heter:
                    out_path = os.path.join(output_dir, str(epoch), 'fedit')
                    new_weights = torch.load(os.path.join(out_path, 'adapter_model.bin'))
                    for name, param in model.named_parameters():
                        key = name.replace(".default", "")
                        param_device = param.device
                        print(key)
                        if key in new_weights:
                            print("Yes")
                            param.data = new_weights[key]
                            param.data = param.data.to(param_device)
                    print("Global adapters successfully loaded into the model.")
                acc = global_evaluation(model, tokenizer, prompter, dev_data_path)

            print('Acc of Epoch', str(epoch), 'is:', acc)
            acc_list.append(acc)
            print(acc_list)

    if acc:
        print('Acc of Epoch', str(epoch), 'is:', acc)
        acc_list.append(acc)
        print(acc_list)
        filename = output_dir + 'log.txt'
        file = open(filename,'a')
        for i in range(len(acc_list)):
            s = str(acc_list[i]).replace('[','').replace(']','')
            s = s.replace("'",'').replace(',','') + "; Total rank across all rounds: " + str(total_rank_comm) + "; Total parameters: " + str(total_param_comm) + '\n'
            print(s)
            if is_florist:
                log_text = "FLoRIST-" + str(global_model) + ", Threshold-" + str(threshold) + ":\n"

            elif is_flex:
                log_text = "FlexLoRA-" + str(global_model) + ":\n"

            elif is_flora:
                log_text = "FLoRA-" + str(global_model) + ":\n"

            else:
                if is_ffa:
                    log_text = "FFA-LoRA-" + str(global_model) + ":\n"
                else:
                    log_text = "FedIT-" + str(global_model) + ":\n"

            file.write(log_text)
            file.write(s)
        file.close()
        print("Log Saved")

if __name__ == "__main__":
    fire.Fire(fl_finetune)
