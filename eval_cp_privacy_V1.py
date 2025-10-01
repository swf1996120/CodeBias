import copy
import os
import re

# this avoids nccl hanging
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_TOKEN"] = "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"



import argparse
import json
import random
import time
from datetime import timedelta
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm
smoothie = SmoothingFunction().method4


# 8) 评测：取生成的前 50 个 token 与 gold suffix 对齐比较（原脚本逻辑）
def calc_bleu4(tokenizer, sample_ids, gen_ids):
    ref = tokenizer.decode(sample_ids, skip_special_tokens=True)
    hyp = tokenizer.decode(gen_ids,   skip_special_tokens=True)
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)


from cp_eval_utils import (
    calculate_openai_cost,
    compute_gpt_extraction_for_all,
    compute_gpt_pii_leakage,
    compute_gpt_utility_score,
    compute_pii_leakage,
    compute_utility_score,
    split_by_think,
)
from generate_utils import (
    UserDataLogitsProcessor,
    calculate_openrouter_cost,
    display_generation_config,
    generate_openrouter_hide_data,
    generate_with_budget,
    generate_with_openrouter,
    generate_with_openrouter_rana,
    generate_with_openrouter_swap,
    generate_with_rana,
    generate_with_swap,
    get_provider_model_name,
)

# Define models that should primarily use API providers
API_ONLY_MODELS = {
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-v3",
    "deepseek-ai/deepseek-v3-0324",
    "deepseek/deepseek-r1:free"
}

Stataud_Model ={
    "deepseek/deepseek-r1:free": "deepseek-ai/DeepSeek-R1"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate confidential information handling"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--seed", type=int, default=221097, help="Random seed")
    parser.add_argument(
        "--scenarior",
        type=str,
        help="Scenario to evaluate (func_job, func_med, func_edu, testcase)",
    )# The benchmark dataset that need to be modified to fit the code tasks TODO
    parser.add_argument(
        "--category",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--attribute",
        type=str,
        required=True,
        help="Attribute to evaluate"
    ) # sensitive attribute to evaluate

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to the first N prompts (for quick testing)",
    ) # only the first N prompts are used, useful for quick testing, debugging stage.
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--temperature", type=float, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, help="Top-p for nucleus sampling")
    parser.add_argument("--top_k", type=float, help="K value for top-k sampling")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Repetition penalty parameter",
    )# penalize the model for repeating the same token, default is 1.0, no penalty, if larger than 1.0, penalize more.
    parser.add_argument(
        "--model_provider",
        type=str,
        default="vllm",
        choices=["vllm", "openrouter"],
        help="Model provider to use (vllm, openrouter)",
    ) # openrouter is used to access models via OpenRouter API, vllm is used for local inference with vLLM library.

    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        default=None,
        help="Prompt type to use from prompts/cp_open_ended_chat directory",
    ) # the prompt template to use, should be customized for code tasks.
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Enable eager mode for VLLM execution",
    )
    parser.add_argument(
        "--hide_data",
        action="store_true",
        help="Hide user data in generated outputs",
    )
    parser.add_argument(
        "--budget_thinking",
        type=int,
        default=None,
        help="Token budget for forcing thinking phase",
    )
    parser.add_argument(
        "--prompt_inj",
        type=str,
        default=None,
        help="Path to a file containing prompt injection content",
    ) # simites prompt injection attacks by adding content from the specified file to the prompt.
    parser.add_argument(
        "--gpt_eval",
        action="store_true",
        help="Enable LLM-as-a-judge evaluation using GPT-4o-mini",
    )
    parser.add_argument(
        "--gpt_eval_model",
        type=str,
        default="gpt-4o-mini",
        help="GPT model to use for LLM-as-a-judge evaluation",
    )
    parser.add_argument(
        "--openrouter_settings",
        type=str,
        default="openrouter_settings/default_settings.json",
        help="Path to OpenRouter settings JSON file",
    )
    parser.add_argument(
        "--rana",
        action="store_true",
        help="Enable Reason-Anonymize-Answer (RAnA) flow",
    )# RAnA is a technique to anonymize sensitive information in the reasoning process before generating the final answer. For the code task, we can extract the code snippets and summarize the code snippets to anonymize the code. Reason-Summarization-Answer (RSuA) might be a better name.  TODO
    parser.add_argument( 
        "--swap",
        action="store_true",
        help="Enable Reason-Swap-Answer (RSwA) flow",
    ) #Swap the reasoning with a different one before generating the final answer. maybe we can  mutate the code snippets, for example use different identifier name to replace the original one. TODO
    return parser.parse_args()


def load_data(scenarior: str,
              category: str,
              attribute: str) -> List[Dict]:
    
    input_file = f"{os.path.dirname(__file__)}/{scenarior}/{category}/{attribute}.jsonl"
    
    print(f"Loading data from {input_file}")


    def read_code_json(file_name):
        data = []
        with open(file_name, mode='r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
        return data

    print(input_file)
   
    code = read_code_json(file_name=input_file)
    
    print(len(code), type(code))
    
    items = []
    
    for idx in range(len(code)):
        item = code[idx]
        items.append({
            "prompt": item,   
            "source_id": idx
        })

    if not items:
        print("Warning: no valid rows after length filter; check prefix_len/suffix_len or data.")
    return items


# 只编译一次正则
_CODE_PAT_LIT  = re.compile(r"```python\\n([\s\S]*?)(?:```|$)")
_CODE_PAT_REAL = re.compile(r"```python\s*\n([\s\S]*?)(?:```|$)", re.DOTALL)

def _extract_code_from_answer(answer: str) -> str:
    """从 answer 文本里提取 ```python ...``` 代码块；没有就返回空串。"""
    m = _CODE_PAT_LIT.search(answer) or _CODE_PAT_REAL.search(answer)
    if not m:
        return answer
    return m.group(1).replace("\\n", "\n").strip()

def _get_first_text(out_obj):
    """尽量从各种可能的输出对象里拿到第一段文本"""
    # vLLM: RequestOutput
    if hasattr(out_obj, "outputs") and out_obj.outputs:
        gen0 = out_obj.outputs[0]
        if hasattr(gen0, "text"):
            return gen0.text
        if isinstance(gen0, dict) and "text" in gen0:
            return gen0["text"]

    # 常见自定义对象
    if hasattr(out_obj, "text"):
        return out_obj.text
    if hasattr(out_obj, "output_text"):
        return out_obj.output_text

    # OpenAI / OpenRouter 风格的 dict
    if isinstance(out_obj, dict):
        if "text" in out_obj:
            return out_obj["text"]
        if "output_text" in out_obj:
            return out_obj["output_text"]
        if "choices" in out_obj and out_obj["choices"]:
            c0 = out_obj["choices"][0]
            if isinstance(c0, dict):
                if "text" in c0:
                    return c0["text"]
                if "message" in c0 and "content" in c0["message"]:
                    return c0["message"]["content"]

    # 直接是字符串
    if isinstance(out_obj, str):
        return out_obj

    # 兜底
    return str(out_obj)

def _is_empty_answer_from_output_obj(out_obj, end_think_token: str | None) -> bool:
    """按你现有 split_by_think 规范，判断该样本是否‘没有最终代码’。"""
    text = _get_first_text(out_obj)
    reasoning, answer = split_by_think(text, end_think_token)
    answer = (answer or "").strip()
    if not answer:
        return True
    # 代码任务：必须真的提取到代码块才算有答案
    return _extract_code_from_answer(answer) == ""


def main():
    og_time = time.time()
    args = parse_args()
    if args.hide_data:
        os.environ["VLLM_USE_V1"] = "0"  # need for per-request logit processing
    # Set random seeds
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Add the number of visible GPUs to args
    args.num_gpus = torch.cuda.device_count()

    # Create rich console for pretty printing
    console = Console()

    # Pretty print the arguments using rich
    args_table = Table(title="Execution Arguments", box=box.ROUNDED)
    args_table.add_column("Argument", style="cyan")
    args_table.add_column("Value", style="green")

    for arg, value in vars(args).items():
        args_table.add_row(arg, str(value))

    console.print()
    console.print(Panel(args_table, expand=False))
    console.print()

    # Check if RAnA is enabled - it only works with reasoning-based prompts
    if args.rana:
        if not ("cot" in args.prompt_type or "reasoning" in args.prompt_type):
            print("Error: RAnA can only be used with 'cot' or 'reasoning' prompt types")
            return
        print("RAnA (Reason-Anonymize-Answer) mode enabled")

    # Check if hide_data is enabled - it only works with reasoning-based prompts
    if args.hide_data:
        if not ("cot" in args.prompt_type or "reasoning" in args.prompt_type):
            print(
                "Error: hide_data can only be used with 'cot' or 'reasoning' prompt types"
            )
            return
        print("Data hiding during thinking phase enabled")

    # Load data
    data = load_data(scenarior=args.scenarior, category=args.category, attribute =args.attribute)

    # Load prompt template if specified
    add_template = None
    ## load the system prompt template, We should notice that for the pure reasoning models, the output itself containes the <think> and </think> tags, so we do not need to add them in the system prompt; Maybe this only is used for the base large language models like gpt-3.5-turbo and gpt-4. TODO: if the reasoning model, we should use other templates?
    
    ## this step mainly to obtain the start and end think tokens, which are used in RAnA and RSwA.
    if (
        "deepseek" in args.model.lower()
        or "qwq" in args.model.lower()
        or "cot" in args.prompt_type
    ):
        start_think_token = "<think>"
        end_think_token = "</think>"
    elif "nemotron" in args.model.lower():
        if "reasoning" in args.prompt_type:
            start_think_token = "<think>"
            end_think_token = "</think>"
        else:
            start_think_token = None
            end_think_token = None
    elif "s1" in args.model.lower():
        start_think_token = "<|im_start|>think"
        end_think_token = "<|im_start|>answer"
        print("Reformatted prompt for s1 models")
    else:
        start_think_token = None
        end_think_token = None
        
    print(f"Using start_think_token: {start_think_token}, end_think_token: {end_think_token}")
    
    # Handle prompt injection if specified, achieve the prompt injection attack.
    if args.prompt_type and args.prompt_type in ["cot"]:
        prompt_file = os.path.join(
            "./prompts/cp_open_ended_chat", args.prompt_type + ".txt"
        )
        print(os.path.abspath(prompt_file))
        print(f"Loading prompt template from {prompt_file}")
        try:
            with open(prompt_file, "r") as f:
                injection = f.read().strip()  # Get the first line
                print(f"Loading prompt from {args.prompt_type}: {injection}")
        except FileNotFoundError:
            print(f"Error: Prompt file {args.prompt_type} not found")
            injection = None
    else:
        injection = None
    
    prompts = []
    valid_indices = []
        
    for i, item in enumerate(data):
        if "prompt" in item:
            
            prefix_prompt = item["prompt"]
            
            if "s1" in args.model.lower():
                prefix_prompt[0]["content"] = prefix_prompt[0]["content"].replace(
                                            "<think>", "<|im_start|>think"
                                        ).replace("</think>", "<|im_start|>answer")
            if injection:
                prefix_prompt[0]["content"] = prefix_prompt[0]["content"].format(
                    extra_info=injection
                )
            else:
                prefix_prompt[0]["content"] = prefix_prompt[0]["content"].format(
                    extra_info=""
                ) 
                    
            # Store the formatted situation prompt in the data item for GPT evaluation
            data[i]["formatted_situation_prompt"] = prefix_prompt

            prompt = prefix_prompt
            
            if "nemotron" in args.model.lower():
                thinking = "on" if "reasoning" in args.prompt_type else "off"
                prompt.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"detailed thinking {thinking}",
                    },
                )
            if "cot" in args.prompt_type and all(k not in args.model.lower() for k in ("qwen3", "deepseek-r1")):
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "<think> Let's think step by step.",
                    }
                )
            
            if "vanilla" in args.prompt_type and "deepseek-r1" in args.model.lower():
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "<think>I think I have finished thinking.",
                    }
                )

            # Check if the model is designated as API-only
            is_api_only_model = args.model.lower() in API_ONLY_MODELS

            prompts.append(prompt)
            valid_indices.append(i)
            if i == 0:
                # Print the raw prompt
                print(f"Example prompt:\n{prompt}")
                # Load the tokenizer
                model_path = Stataud_Model.get(args.model, args.model)
                print(f"model path is {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(Stataud_Model.get(args.model, args.model))

                # Apply chat template if available
                if hasattr(tokenizer, "apply_chat_template"):
                    if "deepseek-r1" in args.model.lower(): 
                        formatted_chat = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    else:
                        formatted_chat = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=False
                            if "cot" in args.prompt_type and "qwen3" not in args.model.lower()
                            else True,
                            continue_final_message=True
                            if "cot" in args.prompt_type and "qwen3" not in args.model.lower()
                            else False,
                            enable_thinking= True if "qwen3" in args.model.lower() else False,
                        )
                    print(f"\nFormatted with chat template:\n{formatted_chat}")

    if not prompts:
        print("Error: No prompts found in the dataset")
        return

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        prompts = prompts[: args.limit]
        valid_indices = valid_indices[: args.limit]
        print(f"Limiting to first {args.limit} prompts")

    print(f"Processing {len(prompts)} prompts")

    # Make sure the output directory exists
    if args.rana:
        args.output_file = f"results/RAnA/{args.model.split('/')[-1]}_{args.prompt_type}_{args.budget_thinking if args.budget_thinking else 'Any'}/{args.scenarior}/{args.category}/{args.attribute}/result.json"
    if args.swap:
        args.output_file = f"results/RSwA/{args.model.split('/')[-1]}_{args.prompt_type}_{args.budget_thinking if args.budget_thinking else 'Any'}/{args.scenarior}/{args.category}/{args.attribute}/result.json"
       
    if not args.rana and not args.swap:
        args.output_file = f"results/BASE/{args.model.split('/')[-1]}_{args.prompt_type}_{args.budget_thinking if args.budget_thinking else 'Any'}/{args.scenarior}/{args.category}/{args.attribute}_result.json"
        
    if os.path.exists(args.output_file):
        return

    # Check if should use API or vLLM
    is_api_only_model = args.model.lower() in API_ONLY_MODELS
    use_api = is_api_only_model and args.model_provider in [
        "openrouter",
        "deepseek",
    ]

    # Get the correct model name format for the specified provider
    model_name = get_provider_model_name(args.model, args.model_provider)
    
    print(f"Using model: {model_name}, API: {use_api}")
    # === 先做“按提供方初始化” ===
    if use_api:
        # 只需要 tokenizer + sampling_params（跟你原来完全一致）
        tokenizer = AutoTokenizer.from_pretrained(Stataud_Model.get(args.model, args.model))
        try:
            gen_conf_hf = GenerationConfig.from_pretrained(
                Stataud_Model.get(args.model, args.model)
            ).to_diff_dict()
        except Exception:
            print("Warning: Could not load generation config; using default.")
            gen_conf_hf = {"temperature": 0.6, "top_p": 0.95}

        sampling_params = SamplingParams()
        if args.temperature is not None: sampling_params.temperature = args.temperature
        elif "temperature" in gen_conf_hf: sampling_params.temperature = gen_conf_hf["temperature"]
        if args.top_p is not None: sampling_params.top_p = args.top_p
        elif "top_p" in gen_conf_hf: sampling_params.top_p = gen_conf_hf["top_p"]
        if args.repetition_penalty is not None: sampling_params.repetition_penalty = args.repetition_penalty
        elif "repetition_penalty" in gen_conf_hf: sampling_params.repetition_penalty = gen_conf_hf["repetition_penalty"]
        if args.top_k is not None: sampling_params.top_k = args.top_k
        elif "top_k" in gen_conf_hf: sampling_params.top_k = gen_conf_hf["top_k"]
        sampling_params.max_tokens = args.max_tokens
        sampling_params.seed = args.seed
        sampling_params.skip_special_tokens = False

        gen_conf = display_generation_config(console, sampling_params)

    else:
        # vLLM 初始化（与原逻辑一致）
        print(f"Loading model {model_name} with vLLM")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_prefix_caching=True,
            max_model_len=args.max_tokens,
            enforce_eager=args.eager,
            generation_config="auto",
            trust_remote_code=True,
            gpu_memory_utilization=0.7 if "s1" in args.model.lower() else 0.9,
            # gpu_memory_utilization=0.6 if "s1" in args.model.lower() else 0.6,
        )

        sampling_params = llm.get_default_sampling_params()
        if "nemotron" in args.model.lower():
            if "vanilla" in args.prompt_type:
                sampling_params.temperature = 0.0
                sampling_params.top_p = 1.0
                sampling_params.top_k = -1
                sampling_params.repetition_penalty = 1.0
            elif "reasoning" in args.prompt_type:
                sampling_params.temperature = 0.6
                sampling_params.top_p = 0.95

        if args.temperature is not None: sampling_params.temperature = args.temperature
        if args.top_p is not None: sampling_params.top_p = args.top_p
        if args.repetition_penalty is not None: sampling_params.repetition_penalty = args.repetition_penalty
        if args.top_k is not None: sampling_params.top_k = args.top_k
        sampling_params.max_tokens = args.max_tokens
        sampling_params.seed = args.seed
        sampling_params.skip_special_tokens = False

        gen_conf = display_generation_config(console, sampling_params)
        tokenizer = llm.get_tokenizer()  # 本地模式下在这里拿 tokenizer

    # === 定义统一的一次生成入口：_gen_once ===
    def _gen_once(batch_prompts: list):
        """只写一处：把你原来所有分支挪到这里。首次 & 重试都调用它。"""
        if use_api:
            if args.model_provider == "openrouter":
                if args.swap:
                    outs, _, _ = generate_with_openrouter_swap(
                        batch_prompts, data, valid_indices, model_name,
                        sampling_params, args, start_think_token, end_think_token
                    )
                    return outs
                elif args.rana and ("cot" in args.prompt_type or "reasoning" in args.prompt_type):
                    outs, _, _ = generate_with_openrouter_rana(
                        batch_prompts, data, valid_indices, model_name,
                        sampling_params, args, start_think_token, end_think_token
                    )
                    return outs
                elif args.hide_data and ("cot" in args.prompt_type or "reasoning" in args.prompt_type):
                    outs, _, _ = generate_openrouter_hide_data(
                        batch_prompts, data, valid_indices, model_name,
                        sampling_params, args, end_think_token
                    )
                    return outs
                else:
                    return generate_with_openrouter(
                        batch_prompts, model_name, sampling_params, args,
                        end_think_token, is_cot=("cot" in args.prompt_type),
                    )
            else:
                raise NotImplementedError("Only openrouter path is wired here.")
        else:
            if args.budget_thinking is not None:
                return generate_with_budget(
                    llm, batch_prompts, sampling_params, args,
                    start_think_token, end_think_token,
                )
            elif args.rana and ("cot" in args.prompt_type or "reasoning" in args.prompt_type):
                return generate_with_rana(
                    llm=llm, prompts=batch_prompts, data=data, valid_indices=valid_indices,
                    args=args, model_name=model_name,
                    start_think_token=start_think_token, end_think_token=end_think_token,
                    sampling_params=sampling_params,
                )
            elif args.swap:
                return generate_with_swap(
                    llm=llm, prompts=batch_prompts, data=data, valid_indices=valid_indices,
                    args=args, model_name=model_name,
                    start_think_token=start_think_token, end_think_token=end_think_token,
                    sampling_params=sampling_params,
                )
            else:
                if 'qwen3' in args.model.lower() and "cot" not in args.prompt_type:
                    rendered = tokenizer.apply_chat_template(
                        batch_prompts,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    return llm.generate(rendered, sampling_params=sampling_params)
                elif "deepseek-r1" in args.model.lower():
                    rendered = tokenizer.apply_chat_template(
                        batch_prompts,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    return llm.generate(rendered, sampling_params=sampling_params)
                else:
                    return llm.chat(
                        batch_prompts,
                        sampling_params=sampling_params,
                        chat_template=llm.get_tokenizer().chat_template,
                        add_generation_prompt=(
                            False if "cot" in args.prompt_type and "qwen3" not in args.model.lower() else True
                        ),
                        continue_final_message=(
                            True if "cot" in args.prompt_type and "qwen3" not in args.model.lower() else False
                        ),
                    )

    # === 首次生成 ===
    outputs = _gen_once(prompts)

    # === CoT 专用自动重试：answer 为空则同样 prompts 重生；不复制任何分支逻辑 ===
    if "cot" in (args.prompt_type or ""):
        max_retries = 10   # 可调
        retry_round = 0
        need_retry_idx = [i for i, out in enumerate(outputs) if _is_empty_answer_from_output_obj(out, end_think_token)]

        while need_retry_idx and retry_round < max_retries:
            print(f"[Retry] Round {retry_round+1}: {len(need_retry_idx)} empty answers, regenerating...")
            retry_prompts = [prompts[i] for i in need_retry_idx]
            retry_outputs = _gen_once(retry_prompts)

            # 回填对应位置
            for j, idx in enumerate(need_retry_idx):
                outputs[idx] = retry_outputs[j]

            # 继续筛剩下仍为空的
            need_retry_idx = [i for i in need_retry_idx if _is_empty_answer_from_output_obj(outputs[i], end_think_token)]
            retry_round += 1


    # Process generated outputs (treating all outputs as lists)
    all_outputs = []
    for output in outputs:
        # Always extract a list of generations
        prompt_outputs = [out.text for out in output.outputs]
        all_outputs.append(prompt_outputs)
    ## print tokenizer 的名称
    print(f"Using tokenizer: {tokenizer.__class__.__name__}")
    # Prepare results: update each valid data item with the generated text.
    '''
    modle_output: list of generated outputs (each output can have multiple generations)
    model_reasoning: list of reasoning parts extracted from each generation
    model_answer: list of answer parts extracted from each generation
    prompt: the formatted prompt used for generation
    input_token_length: token length of the prompt
    output_token_length: list of token lengths for each generation
    reasoning_token_length: list of token lengths for the reasoning part of each generation
    answer_token_length: list of token lengths for the answer part of each generation
    close_think_tokens: list of counts of </think> tokens in each generation
    '''
    count = 0
    for i in valid_indices:
        text_list = all_outputs[i]  # always a list
        (
            reasons,
            answers,
            out_tokens,
            reason_tokens,
            answer_tokens,
            close_think_tokens,
        ) = [], [], [], [], [], []
        # print("Expected suffix: {}".format(data[i].get("suffix", "")))
        # print(f"Expected suffix length (in tokens): {suffix_length}")
        for text in text_list:
            reasoning, answer = split_by_think(text, end_think_token)
            # print ("Original answer: {}".format(answer.strip()))
            # 两种模式：A=字面 \n，B=实际换行
            pat_literal = re.compile(r"```python\\n([\s\S]*?)(?:```|$)")
            pat_newline = re.compile(r"```python\s*\n([\s\S]*?)(?:```|$)", re.DOTALL)

            m = pat_literal.search(answer)
            if not m:
                m = pat_newline.search(answer)

            code = answer
            if m:
                code = m.group(1)
                # 若匹配的是字面 \n 的情况，转为真实换行
                code = code.replace("\\n", "\n")
                            
            answer = code
                        
            reasons.append(reasoning)
            answers.append(answer)
            
            out_tokens.append(len(tokenizer.encode(reasoning)) + len(tokenizer.encode(answer)))
            reason_tokens.append(len(tokenizer.encode(reasoning)))
            answer_tokens.append(len(tokenizer.encode(answer)))
            # Count occurrences of </think> in text
            think_count = (
                text.count(end_think_token) if end_think_token is not None else 0
            )
            close_think_tokens.append(think_count)
        data[i]["model_output"] = text_list
        data[i]["model_reasoning"] = reasons
        data[i]["model_answer"] = answers
        # Handle both text and chat format prompts for tokenization
        if isinstance(outputs[i].prompt, str):
            data[i]["prompt"] = outputs[i].prompt
        elif isinstance(outputs[i].prompt, list):
            data[i]["prompt"] = tokenizer.apply_chat_template(
                outputs[i].prompt,
                tokenize=False,
                add_generation_prompt=False
                if "cot" in args.prompt_type and "qwen3" not in args.model.lower()
                else True,
                continue_final_message=True
                if "cot" in args.prompt_type and "qwen3" not in args.model.lower()
                else False,
                enable_thinking= True if "qwen3" in args.model.lower() else False,
            )
        data[i]["input_token_length"] = len(tokenizer.encode(data[i]["prompt"]))
        data[i]["output_token_length"] = out_tokens
        data[i]["reasoning_token_length"] = reason_tokens
        data[i]["answer_token_length"] = answer_tokens
        data[i]["close_think_tokens"] = close_think_tokens

        # Add provider information if available
        if hasattr(outputs[i], "provider_info"):
            data[i]["provider_info"] = outputs[i].provider_info

    # Filter data to only include entries with indices in valid_indices
    filtered_data = [data[i] for i in valid_indices]

    print(f"count of exact match with reference suffix: {count}")

    # Compute average token lengths and think token statistics
    avg_output_length = sum(
        [
            sum(item["output_token_length"]) / len(item["output_token_length"])
            for item in filtered_data
        ]
    ) / len(filtered_data)
    avg_reasoning_length = sum(
        [
            sum(item["reasoning_token_length"]) / len(item["reasoning_token_length"])
            for item in filtered_data
        ]
    ) / len(filtered_data)
    avg_answer_length = sum(
        [
            sum(item["answer_token_length"]) / len(item["answer_token_length"])
            for item in filtered_data
        ]
    ) / len(filtered_data)
    avg_close_think_tokens = sum(
        [
            sum(item["close_think_tokens"]) / len(item["close_think_tokens"])
            for item in filtered_data
        ]
    ) / len(filtered_data)
    max_close_think_tokens = (
        max([max(item["close_think_tokens"]) for item in filtered_data])
        if filtered_data
        else 0
    )

    # Add scores to summary
    end_time = time.time()
    time_delta = end_time - og_time
    time_required = str(timedelta(seconds=int(time_delta)))

    # Collect unique providers if using OpenRouter
    unique_providers = set()
    if args.model_provider == "openrouter":
        for item in filtered_data:
            if "provider_info" in item:
                for provider in item["provider_info"]:
                    unique_providers.add(provider["provider_name"])

    summary = {
        "total_examples": len(filtered_data),
        "time_required": time_required,
        "avg_output_length": avg_output_length,
        "avg_reasoning_length": avg_reasoning_length,
        "avg_answer_length": avg_answer_length,
        "avg_close_think_tokens": avg_close_think_tokens,
        "max_close_think_tokens": max_close_think_tokens,
        "rana_enabled": args.rana,
    }


    # Add summary and args to data
    result_data = {
        "args": vars(args),
        "gen_conf": gen_conf,
        "summary": summary,
        "data": filtered_data,  # Store only the filtered data
    }


            
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # Prepare to save results, but only save after GPT eval if it's enabled
    if not args.gpt_eval:
        # Save results immediately if GPT eval is not enabled
        with open(args.output_file, "w") as f:
            json.dump(result_data, f, indent=2)

    print(f"Generated {len(all_outputs)} outputs in {time_required}")
    print(
        f"Average token lengths - Output: {avg_output_length:.2f}, Reasoning: {avg_reasoning_length:.2f}, Answer: {avg_answer_length:.2f}"
    )
    print(
        f"Think tokens - Avg: {avg_close_think_tokens:.2f}, Max: {max_close_think_tokens}"
    )

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    main()
