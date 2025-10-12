"""Utility functions for generating text with various models and strategies."""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Union

import requests
import torch
from dotenv import load_dotenv
from openai import OpenAI
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_random_exponential
from tqdm import tqdm

from cp_eval_utils import (
    anonymize_reasonings_parallel,
    calculate_openrouter_cost,
    swap_reasonings_parallel,
)


@dataclass
class OutputObj:
    """Simple dataclass to mimic VLLM's output structure.

    Attributes
    ----------
    text : str
        The generated text output.
    """

    text: str


@dataclass
class RequestOutputObj:
    """Dataclass to mimic VLLM's RequestOutput structure.

    Attributes
    ----------
    outputs : List[OutputObj]
        A list of output objects, each containing generated text.
    prompt : Union[str, List[Dict]]
        The prompt used to generate the output.
    """

    outputs: List[OutputObj]
    prompt: Union[str, List[Dict]]


class UserDataLogitsProcessor:
    """A logits processor that blocks generation of user data tokens.

    This processor is used during the model's "thinking" phase to prevent it
    from leaking personally identifiable information (PII) or other sensitive
    user data that was part of the input prompt. It works by assigning a
    log-probability of -inf to token IDs corresponding to the user's data,
    effectively blocking them from being generated. The blocking is deactivated
    once an `end_think_token` is generated.

    Attributes
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer used to encode text into token IDs.
    user_data : dict or list
        A nested structure containing user data to be blocked.
    end_think_token : str, optional
        The token that signals the end of the thinking phase. If None, blocking
        is always active.
    end_think_token_ids : list of int, optional
        The token IDs for the `end_think_token`.
    is_thinking_phase : bool
        A flag indicating whether the model is currently in the thinking phase.
    blocked_token_ids : set of int
        A set of token IDs that are blocked from being generated.
    """

    def __init__(self, tokenizer, user_data, end_think_token=None):
        """Initialize the UserDataLogitsProcessor.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
            The tokenizer for encoding user data.
        user_data : dict or list
            The user data to block during generation.
        end_think_token : str, optional
            The string marking the end of the thinking phase. Default is None.
        """
        self.tokenizer = tokenizer
        self.user_data = user_data
        self.end_think_token = end_think_token
        self.end_think_token_ids = (
            None
            if end_think_token is None
            else tokenizer.encode(end_think_token, add_special_tokens=False)
        )
        self.is_thinking_phase = True

        # Pre-compute token IDs for all user data values
        self.blocked_token_ids = set()
        self.parsed_user_data = self._extract_values(user_data)

        # Get all values from the profile
        values = [
            str(v)
            for v in self.parsed_user_data
            if isinstance(v, (str, int, float, bool))
        ]

        values = [
            [v, " " + v, v.lower(), " " + v.lower(), v.upper(), " " + v.upper()]
            for v in values
        ]
        values = list(set([item for sublist in values for item in sublist]))
        token_ids = [self.tokenizer.encode(v, add_special_tokens=False) for v in values]
        token_ids = list(set([item for sublist in token_ids for item in sublist]))
        self.blocked_token_ids.update(token_ids)

    def _extract_values(self, data):
        """Recursively extract all values from nested dictionaries and lists.

        Parameters
        ----------
        data : dict or list
            The data structure to extract values from.

        Returns
        -------
        list
            A flat list of all values found in the data structure.
        """
        values = []
        if isinstance(data, dict):
            for value in data.values():
                values.extend(self._extract_values(value))
        elif isinstance(data, list):
            for item in data:
                values.extend(self._extract_values(item))
        else:
            values.append(data)
        return values

    def __call__(self, input_ids, logits):
        """Process logits to block user data tokens.

        This method is called at each generation step. It modifies the logits
        to prevent the generation of blocked tokens during the thinking phase.

        Parameters
        ----------
        input_ids : torch.Tensor
            The sequence of input IDs generated so far.
        logits : torch.Tensor
            The logits for the next token.

        Returns
        -------
        torch.Tensor
            The modified logits.
        """
        if (
            self.end_think_token_ids is not None
            and self.is_thinking_phase
            and len(input_ids) > 1
        ):
            last_tokens = input_ids[-len(self.end_think_token_ids) :]
            think_token_match = torch.equal(
                torch.tensor(last_tokens, device=logits.device),
                torch.tensor(self.end_think_token_ids, device=logits.device),
            )
            if think_token_match:
                self.is_thinking_phase = False
                return logits

        # Only block tokens during thinking phase
        if self.is_thinking_phase:
            for token_id in self.blocked_token_ids:
                logits[token_id] = float("-inf")

        return logits


def generate_with_openrouter(
    prompts, model_name, sampling_params, args, end_think_token=None, is_cot=False
):
    """Generate text using the OpenRouter API.

    This function sends prompts to the OpenRouter API for text generation,
    handling parallel requests, retries, and cost calculation. It's designed
    to work with models available through OpenRouter, such as DeepSeek-R1.

    Parameters
    ----------
    prompts : list of list of dict
        A list of prompts, where each prompt is a list of messages in chat format.
    model_name : str
        The name of the model to use on OpenRouter (e.g., 'deepseek/deepseek-chat').
    sampling_params : object
        An object containing sampling parameters like temperature, top_p, max_tokens.
    args : argparse.Namespace
        Command-line arguments, expected to contain `openrouter_settings`.
    end_think_token : str, optional
        The token that separates reasoning from the final answer. If provided,
        the two parts are concatenated. Default is None.
    is_cot : bool, optional
        Flag indicating if it is a Chain-of-Thought prompt. Default is False.

    Returns
    -------
    list of RequestOutputObj
        A list of output objects, each containing the generated text and original prompt.
    """
    print("Starting generation with OpenRouter API, generate_with_openrouter")
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    if (
        model_name == "deepseek/deepseek-chat"
    ):  # for some reason DeepInfra does not take tool outputs
        openrouter_settings["provider"].pop("order")
        openrouter_settings["provider"]["allow_fallbacks"] = True

    all_outputs = [None] * len(prompts)  # Initialize with correct size
    num_workers = min(50, len(prompts))  # Number of parallel workers
    generation_ids = []  # Store all generation IDs
    generation_id_to_prompt_idx = {}  # Map generation IDs to prompt indices

    print(
        f"Generating responses with OpenRouter API for {len(prompts)} prompts using {num_workers} workers..."
    )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt):
        """Make a single API request to OpenRouter with retries."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={**params, "messages": prompt},
        )
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"], response_data["id"]

    def process_single_prompt(prompt, prompt_idx, end_think_token=None, is_cot=False):
        """Process a single prompt to generate `n` samples."""
        batch_outputs = []
        for _ in range(sampling_params.n):
            # Set up generation parameters
            params = {
                "model": model_name,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
            }

            if sampling_params.top_p is not None:
                params["top_p"] = sampling_params.top_p

            if hasattr(sampling_params, "stop") and sampling_params.stop:
                params["stop"] = sampling_params.stop

            # Add OpenRouter settings
            params.update(openrouter_settings)

            # Make API request with retry mechanism
            response_output, gen_id = make_api_request(params, prompt)
            if (
                end_think_token is not None
                and not is_cot
                and "reasoning" in response_output
                and "content" in response_output
                and response_output["reasoning"] is not None
                and response_output["content"] is not None
            ):
                output_text = (
                    response_output["reasoning"]
                    + end_think_token
                    + response_output["content"]
                )
            else:
                output_text = response_output["content"]
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = prompt_idx

            # Create object that mimics VLLM's output structure
            batch_outputs.append(OutputObj(output_text))

        return prompt_idx, batch_outputs

    # Process prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                process_single_prompt, prompt, i, end_think_token, is_cot
            ): i
            for i, prompt in enumerate(prompts)
        }

        # Create progress bar
        progress_bar = tqdm(total=len(prompts), desc="OpenRouter API calls")

        # Process results as they complete
        for future in as_completed(future_to_idx):
            idx, batch_outputs = future.result()
            # Create an object that mimics VLLM's RequestOutput structure and place it at the correct index
            all_outputs[idx] = RequestOutputObj(batch_outputs, prompts[idx])
            # Update progress bar
            progress_bar.update(1)

    print(f"Completed {len(all_outputs)} OpenRouter API calls")

    # Calculate and display total cost
    total_cost, provider_info = calculate_openrouter_cost(generation_ids, api_key)
    cost_console = Console()
    cost_panel = Panel(
        f"[bold white]Total OpenRouter API Cost:[/] [bold green]${total_cost:.2f}[/]",
        title="💰 Cost Summary",
        border_style="green",
    )
    cost_console.print()
    cost_console.print(cost_panel)
    cost_console.print()

    # Add provider info to outputs
    for gen_id, prompt_idx in generation_id_to_prompt_idx.items():
        if not hasattr(all_outputs[prompt_idx], "provider_info"):
            all_outputs[prompt_idx].provider_info = []
        all_outputs[prompt_idx].provider_info.append(provider_info[gen_id])

    return all_outputs


def get_provider_model_name(model_name, provider):
    """Get the correct model name format for the specified provider.

    Different providers (OpenRouter, DeepSeek API, local vLLM) may use
    different identifiers for the same model. This function canonicalizes
    the model name based on the specified provider.

    Parameters
    ----------
    model_name : str
        The generic model name (e.g., 'deepseek-ai/deepseek-r1').
    provider : str
        The provider name ('openrouter', or 'vllm').

    Returns
    -------
    str
        The provider-specific model name.

    Raises
    ------
    ValueError
        If a model is not supported by the specified provider.
    """
    # Handle DeepSeek model naming conventions per provider
    if model_name.lower() in [
        "deepseek-ai/deepseek-r1",
        "deepseek/deepseek-r1",
    ]:
        if provider == "openrouter":
            return "deepseek/deepseek-r1"

        elif provider == "vllm":
            raise ValueError(
                "Cannot use vLLM as provider, as models cannot be run locally. Please use 'openrouter' or 'deepseek' as provider."
            )
    elif model_name.lower() == "deepseek-ai/deepseek-v3":
        if provider == "openrouter":
            return "deepseek/deepseek-chat"

        elif provider == "vllm":
            raise ValueError(
                "Cannot use vLLM as provider, as models cannot be run locally. Please use 'openrouter' as provider."
            )
    elif model_name.lower() == "deepseek-ai/deepseek-v3-0324":
        if provider == "openrouter":
            return "deepseek/deepseek-chat-v3-0324"

        elif provider == "vllm":
            raise ValueError(
                "Cannot use vLLM as provider, as models cannot be run locally. Please use 'openrouter' or 'deepseek' as provider."
            )
    return model_name


def display_generation_config(console, sampling_params):
    """Display the generation configuration in a pretty table.

    Parameters
    ----------
    console : rich.console.Console
        The rich console object for printing.
    sampling_params : object
        An object containing the sampling parameters for generation.

    Returns
    -------
    dict
        A dictionary containing the generation configuration parameters.
    """
    # Save sampling parameters in a gen_conf dictionary
    gen_conf = {
        "temperature": sampling_params.temperature
        if hasattr(sampling_params, "temperature")
        else None,
        "top_p": sampling_params.top_p if hasattr(sampling_params, "top_p") else None,
        "top_k": sampling_params.top_k if hasattr(sampling_params, "top_k") else None,
        "repetition_penalty": sampling_params.repetition_penalty
        if hasattr(sampling_params, "repetition_penalty")
        else None,
        "max_tokens": sampling_params.max_tokens,
        "n": sampling_params.n,
        "seed": sampling_params.seed,
        "stop": sampling_params.stop if hasattr(sampling_params, "stop") else None,
        "skip_special_tokens": sampling_params.skip_special_tokens
        if hasattr(sampling_params, "skip_special_tokens")
        else None,
    }

    # Pretty print the generation configuration using rich
    gen_conf_table = Table(title="Generation Configuration", box=box.ROUNDED)
    gen_conf_table.add_column("Parameter", style="cyan")
    gen_conf_table.add_column("Value", style="green")

    for param, value in gen_conf.items():
        gen_conf_table.add_row(param, str(value))

    console.print()
    console.print(Panel(gen_conf_table, expand=False))
    console.print()

    return gen_conf


_SUMMARIZE_DEBIAS_CODE_SYSTEM = (
    "You analyze text about code, programming.\n"
    "Perform two steps on the INPUT TEXT:\n"
    "Step 1 (Summarize): Summarize the given chain-of-thought into a short, information-dense reasoning"
    "(purpose, behavior, inputs/outputs, APIs, constraints). No speculation.\n"
    "Step 2 (Debias): Remove potentially biased/subjective/opinionated language"
    "objective and verifiable statements about the code.\n"
    "Output ONLY the cleaned reasoning text."
)

def _build_openai_client():
    base_url = os.getenv("OPENAI_BASE_URL", None)
    from openai import OpenAI
    if base_url:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _build_code_user_prompt(thinking_text: str) -> str:
    return (
        f"Original chain-of-thought (between special tags):\n{thinking_text}"
    )

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _summarize_and_debias_code_once(
    client,
    model: str,
    text: str,
    max_tokens: int = 10000
) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SUMMARIZE_DEBIAS_CODE_SYSTEM},
            {"role": "user", "content": _build_code_user_prompt(text)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    # 简单清理 code fence（以防万一）
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def summarize_and_sanitize_reasonings(
    think_texts: List[str],
    openai_model: str = "gpt-4o-mini",
    max_tokens: int = 512,
    parallel: bool = True,
) -> List[str]:
    """
    用 ChatGPT 对思考文本做摘要 + 去bias。默认串行，想并行可把 parallel=True 并用线程池。
    """
    client = _build_openai_client()
    if not os.getenv("OPENAI_API_KEY"):
        # 没有 key 就“无操作”返回，以免流程中断
        raise ValueError("OPENAI_API_KEY not found in .env file")   

    results = []
    if parallel:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=min(8, len(think_texts))) as ex:
            futs = [
                ex.submit(_summarize_and_debias_code_once, client, openai_model, t, max_tokens)
                for t in think_texts
            ]
            for fut in futs:
                try:
                    results.append(fut.result() or "")
                except Exception:
                    results.append("")
    else:
        for t in think_texts:
            try:
                results.append(_summarize_and_debias_code_once(client, openai_model, t, max_tokens))
            except Exception:
                results.append("")
    # 兜底：若某些为空，则回退原文
    cleaned = [r if r.strip() else t for r, t in zip(results, think_texts)]
    return cleaned

# -----------------------------------------------------------
# 自由生成思考（到 </think> 截止）——无 budget 循环
# -----------------------------------------------------------
def _gen_reasoning_free(
    llm,
    prompts,
    sampling_params,
    args,
    start_think_token: str,
    end_think_token: str,
    is_qwen3: bool,
    tokenizer,
    custom_template: Optional[str],
) -> List[str]:
    reason_params = sampling_params.clone()
    # 建议：给思考阶段一个单独上限（不走 budget，但要止损）
    reason_params.max_tokens = 2048
    reason_params.stop = [end_think_token, " " + end_think_token]
    reason_params.skip_special_tokens = False
    reason_params.include_stop_str_in_output = True

    if is_qwen3:
        # render 到 <think> 起点，再用 llm.generate
        rendered = []
        for msgs in prompts:
            s = _apply_template_to_str(tokenizer, msgs, enable_thinking=True, add_generation_prompt=True)
            if s.rstrip().endswith(start_think_token):
                s += "\n"
            rendered.append(s)
        outs = llm.generate(rendered, sampling_params=reason_params)
        texts = [o.outputs[0].text for o in outs]
    else:
        outs = llm.chat(
            prompts,
            sampling_params=reason_params,
            chat_template=custom_template if custom_template is not None else tokenizer.chat_template,
            add_generation_prompt=False,         # reasoning 阶段通常不需要再添加 <assistant> 引导
            continue_final_message=True,         # 让模型继续在最后一条消息上写
        )
        texts = [o.outputs[0].text for o in outs]

    # 强制闭合，保证下游流程不崩
    cleaned = []
    for t in texts:
        cleaned.append(_ensure_close_with_phrase(t, end_think_token))
    return cleaned



# -----------------------------------------------------------
# 主流程：RAnA（自由生成 → 摘要去敏 → 拼回作答）
# -----------------------------------------------------------
def generate_with_rana(
    llm,
    prompts,
    args,
    model_name,
    start_think_token,
    end_think_token,
    sampling_params=None,
):
    """
    自由生成思考（到 </think> 截止），把思考交给 ChatGPT 做“摘要+去敏”，
    再把处理后的 think 拼回本地模型继续生成答案。
    """
    print("Starting RAnA generation process (free reasoning)")

    tokenizer = llm.get_tokenizer()
    is_qwen3 = _is_qwen3_thinker(args)

    # 自定义模板（例如 deepseek 等需要保留思考内容的模板）
    custom_template_path = f"chat_templates/rana/{model_name.replace('/', '_')}.jinja"
    try:
        with open(custom_template_path, "r", encoding="utf-8") as f:
            custom_template = f.read()
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        custom_template = None
        print(f"No custom template for {model_name}. Using default tokenizer.chat_template")

    # ---- Step 1: 自由生成“思考”到 </think> ----
    final_reasonings = _gen_reasoning_free(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        args=args,
        start_think_token=start_think_token,
        end_think_token=end_think_token,
        is_qwen3=is_qwen3,
        tokenizer=tokenizer,
        custom_template=custom_template,
    )

    # （可选）把 </think> 去掉再送给 OpenAI（一般去掉更干净）
    stripped_reasonings = [r.replace(end_think_token, "").strip() for r in final_reasonings]

    # -------------------------------------------------------
    # Step B: 用 ChatGPT 做“摘要 + 去敏/去偏”
    #   - 如未设置 OPENAI_API_KEY，则回退原文
    #   - 这里无 profile 时传 None
    # -------------------------------------------------------
    openai_model = getattr(args, "openai_model", "gpt-4o-mini")
    openai_max_tokens = getattr(args, "openai_max_tokens", 10000)
    cleaned_reasonings = summarize_and_sanitize_reasonings(
        think_texts=stripped_reasonings,
        openai_model=openai_model,
        max_tokens=openai_max_tokens,
        parallel=True,
    )

    # 确保闭合：给清洗后的文本补上 </think>（规范闭合，避免模板截断）
    closed_reasonings = [_ensure_close_with_phrase(cr, end_think_token) for cr in cleaned_reasonings]

    # -------------------------------------------------------
    # Step C: 把清洗后的思考拼回 prompt，让本地模型生成“最终答案”
    #   - Qwen3：渲染到 <think> 起点 → 拼接 cleaned + </think> → llm.generate
    #   - 非 Qwen：直接在 messages 里 append assistant 一条包含 cleaned + </think> → llm.chat
    # -------------------------------------------------------
    answer_params = sampling_params.clone() if sampling_params is not None else None
    if answer_params is None:
        from vllm import SamplingParams
        answer_params = SamplingParams()
    answer_params.max_tokens = args.max_tokens - 2048
    # 答案阶段通常不要再 stop 掐 </think>；只在 reasoning 阶段使用 stop
    answer_params.stop = None
    answer_params.include_stop_str_in_output = False
    answer_params.skip_special_tokens = True

    outputs = []

    if is_qwen3:
        # ---- Qwen3：render 到 <think> 起点，再拼 cleaned + </think> ----
        rendered_with_reason = []
        for i, msgs in enumerate(prompts):
            s = _apply_template_to_str(tokenizer, msgs, enable_thinking=True, add_generation_prompt=True)
            # 如果刚好以 <think> 结尾，补个换行更自然
            if s.rstrip().endswith(start_think_token):
                s += "\n"
            # 拼回清洗后的 reasoning，并强制闭合
            s += closed_reasonings[i]
            rendered_with_reason.append(s)

        ans_outs = llm.generate(rendered_with_reason, sampling_params=answer_params)

        for i, ans in enumerate(ans_outs):
            answer_text = ans.outputs[0].text
            # 防止答案里多余地再复述 </think>
            if answer_text.startswith(end_think_token):
                answer_text = answer_text[len(end_think_token):]
            if answer_text.endswith(end_think_token):
                answer_text = answer_text[: -len(end_think_token)]
            ans.outputs[0].text = closed_reasonings[i] + answer_text
            ans.prompt = stripped_reasonings[i]
            outputs.append(ans)

    else:
        # ---- 非 Qwen：直接在对话最后追加“assistant: <think>... </think>” ----
        answer_prompts = []
        for i, msgs in enumerate(prompts):
            new_msgs = deepcopy(msgs)
            new_msgs[-1]["content"] += closed_reasonings[i]
            
            answer_prompts.append(new_msgs)

        ans_outs = llm.chat(
            answer_prompts,
            sampling_params=answer_params,
            chat_template=custom_template if custom_template is not None else tokenizer.chat_template,
            add_generation_prompt=False,
            continue_final_message=True,      # 继续在 assistant 的最后一条消息后写“答案”
        )

        for i, ans in enumerate(ans_outs):
            answer_text = ans.outputs[0].text
            if answer_text.startswith(end_think_token):
                answer_text = answer_text[len(end_think_token):]
            if answer_text.endswith(end_think_token):
                answer_text = answer_text[: -len(end_think_token)]
            combined = closed_reasonings[i] + answer_text

            output_obj = OutputObj(combined)
            request_output = RequestOutputObj([output_obj], prompts[i])
            outputs.append(request_output)

    return outputs


def generate_with_openrouter_rana(
    prompts,
    data,
    valid_indices,
    model_name,
    sampling_params,
    args,
    start_think_token,
    end_think_token,
):
    """Implement the Reason-Anonymize-Answer (RAnA) approach using the OpenRouter API.

    This function orchestrates the RAnA pipeline with OpenRouter as the backend:
    1. Generate reasoning in parallel for each prompt, stopping at `end_think_token`.
    2. Anonymize the generated reasoning traces to remove PII.
    3. Feed the anonymized reasoning back to the OpenRouter API to generate final answers.

    Parameters
    ----------
    prompts : list
        A list of prompts for the model.
    data : list of dict
        The dataset, containing user profile data for each prompt.
    valid_indices : list of int
        The indices of the prompts/data to be processed.
    model_name : str
        The name of the model to use on OpenRouter.
    sampling_params : object
        An object with sampling parameters (temperature, top_p, etc.).
    args : argparse.Namespace
        Command-line arguments, containing model path and prompt type.
    start_think_token : str
        The token to prepend to the reasoning.
    end_think_token : str
        The token to signal the end of the reasoning phase.

    Returns
    -------
    tuple
        - list of RequestOutputObj: The final generated outputs.
        - list of str: The generation IDs from OpenRouter.
        - dict: A mapping from generation IDs to prompt indices.
    """
    import time
    from copy import deepcopy

    from transformers import AutoTokenizer

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    if model_name == "deepseek/deepseek-chat":
        openrouter_settings["provider"].pop("order")
        openrouter_settings["provider"]["allow_fallbacks"] = True

    # Initialize variables to store generation results
    reasoning_texts = [None] * len(valid_indices)  # Initialize with correct size
    num_workers = min(50, len(valid_indices))  # Number of parallel workers
    generation_ids = []  # Store all generation IDs
    generation_id_to_prompt_idx = {}  # Map generation IDs to prompt indices

    print(
        f"Generating responses with OpenRouter API for {len(valid_indices)} prompts using {num_workers} workers in RAnA mode..."
    )

    # Load tokenizer for applying chat templates
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Path to custom chat template
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"

    # Load custom chat template
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            tokenizer.chat_template = custom_template
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        print(f"Custom template not found for {args.model} at {custom_template_path}")
        print("Using default chat template")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt_text):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }

        # Always use completions endpoint
        response = requests.post(
            url="https://openrouter.ai/api/v1/completions",
            headers=headers,
            json={**params, "prompt": prompt_text},
        )
        response.raise_for_status()
        gen_id = response.json()["id"]
        output = response.json()
        return output, gen_id

    # Step 1: Generate reasoning for each prompt in parallel
    # Max tokens for reasoning is max_tokens - 500
    reasoning_max_tokens = max(sampling_params.max_tokens - 500, 1000)
    print(
        f"Step 1: Generating reasoning in parallel (max tokens: {reasoning_max_tokens})..."
    )

    # Function to process a single reasoning prompt
    def process_reasoning_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        prompt = prompts[idx]

        # Format the prompt using the chat template if it's a list (chat format)
        if isinstance(prompt, list):
            formatted_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=False if "cot" in args.prompt_type else True,
                continue_final_message=True if "cot" in args.prompt_type else False,
            )
        else:
            # For non-chat prompts, use as-is
            formatted_prompt = prompt

        # Set up generation parameters for reasoning
        reasoning_params = {
            "model": model_name,
            "max_tokens": reasoning_max_tokens,
            "temperature": sampling_params.temperature,
        }

        if sampling_params.top_p is not None:
            reasoning_params["top_p"] = sampling_params.top_p

        # Add stop tokens to end at reasoning phase
        if end_think_token is not None:
            reasoning_params["stop"] = [end_think_token, " " + end_think_token]

        # Add OpenRouter settings
        reasoning_params.update(openrouter_settings)

        # Make API request for reasoning
        response_output, gen_id = make_api_request(reasoning_params, formatted_prompt)
        reasoning_key = "reasoning" if "reasoning" in args.prompt_type else "text"
        reasoning_text = response_output["choices"][0][reasoning_key]

        # Add end_think_token if needed
        if (
            end_think_token is not None
            and reasoning_text is not None
            and not reasoning_text.endswith(end_think_token)
        ):
            reasoning_text += end_think_token

        # Store generation ID mapping
        return prompt_idx, idx, reasoning_text, gen_id, formatted_prompt

    # Process reasoning prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all reasoning tasks
        future_to_idx = {
            executor.submit(process_reasoning_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(
            total=len(valid_indices), desc="Step 1: Reasoning generation"
        )

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, reasoning_text, gen_id, formatted_prompt = (
                future.result()
            )
            reasoning_texts[prompt_idx] = reasoning_text

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    # Step 2: Anonymize all reasoning texts in parallel
    print("Step 2: Anonymizing reasoning in parallel...")

    # Get a representative profile for anonymization
    # Using the first valid index's profile as a representative
    sample_profile = data[valid_indices[0]].get("profile", {})

    # Anonymize all reasoning texts in parallel
    anonymized_results = anonymize_reasonings_parallel(reasoning_texts, sample_profile)

    # Store anonymized reasoning and extracted PII in data
    anonymized_reasoning_list = []
    for i, idx in enumerate(valid_indices):
        anonymized_text, extracted_pii = anonymized_results[i]

        # Store original reasoning in data
        data[idx]["original_reasoning"] = reasoning_texts[i]

        # Store extracted PII data
        if "gpt_extractions" not in data[idx]:
            data[idx]["gpt_extractions"] = {}
        data[idx]["gpt_extractions"]["reasoning"] = extracted_pii

        # Add to anonymized list for next step
        anonymized_reasoning_list.append(anonymized_text)

    # Step 3: Generate answers based on anonymized reasoning in parallel
    final_outputs = [None] * len(valid_indices)  # Initialize with correct size
    print("Step 3: Generating answers in parallel (max tokens: 500)...")

    # Max tokens for answer generation is fixed at 500
    answer_max_tokens = 500

    # Function to process a single answer generation
    def process_answer_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        orig_prompt = prompts[idx]
        anonymized_reasoning = anonymized_reasoning_list[prompt_idx]

        # Prepare prompt for answer generation
        # Create a new chat prompt with anonymized reasoning as assistant message
        answer_messages = deepcopy(orig_prompt)
        if "cot" in args.prompt_type:
            answer_messages[1]["content"] += anonymized_reasoning
        else:
            answer_messages.append(
                {
                    "role": "assistant",
                    "content": start_think_token + "\n" + anonymized_reasoning,
                }
            )

        # Format with chat template
        formatted_answer_prompt = tokenizer.apply_chat_template(
            answer_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

        # Set up generation parameters for answer
        answer_params = {
            "model": model_name,
            "max_tokens": answer_max_tokens,
            "temperature": sampling_params.temperature,
        }

        if sampling_params.top_p is not None:
            answer_params["top_p"] = sampling_params.top_p

        # Add OpenRouter settings
        answer_params.update(openrouter_settings)

        # Generate answer
        response_output, gen_id = make_api_request(
            answer_params, formatted_answer_prompt
        )
        answer_text = response_output["choices"][0]["text"]

        # Combine reasoning and answer
        combined_text = anonymized_reasoning + answer_text

        # Create output object
        output_obj = OutputObj(combined_text)
        request_output = RequestOutputObj([output_obj], orig_prompt)

        return prompt_idx, idx, request_output, gen_id, formatted_answer_prompt

    # Process answer prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all answer generation tasks
        future_to_idx = {
            executor.submit(process_answer_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(total=len(valid_indices), desc="Step 3: Answer generation")

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, request_output, gen_id, formatted_answer_prompt = (
                future.result()
            )
            final_outputs[prompt_idx] = request_output

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    print(f"Completed {len(final_outputs)} OpenRouter API calls with RAnA")
    return final_outputs, generation_ids, generation_id_to_prompt_idx


def generate_openrouter_hide_data(
    prompts, data, valid_indices, model_name, sampling_params, args, end_think_token
):
    """Generate text with OpenRouter, preventing PII leakage using logit biasing.

    This function implements the "hide_data" approach. It first generates a
    reasoning trace while using OpenRouter's `logit_bias` feature to prevent the
    model from generating tokens corresponding to user data. It then generates
    the final answer based on this "sanitized" reasoning.

    Parameters
    ----------
    prompts : list
        A list of prompts for the model.
    data : list of dict
        The dataset, containing user profile data for each prompt.
    valid_indices : list of int
        The indices of the prompts/data to be processed.
    model_name : str
        The name of the model to use on OpenRouter.
    sampling_params : object
        An object with sampling parameters (temperature, top_p, etc.).
    args : argparse.Namespace
        Command-line arguments, containing model path and other settings.
    end_think_token : str
        The token to signal the end of the reasoning phase.

    Returns
    -------
    tuple
        - list of RequestOutputObj: The final generated outputs.
        - list of str: The generation IDs from OpenRouter.
        - dict: A mapping from generation IDs to prompt indices.
    """
    import time
    from copy import deepcopy

    from transformers import AutoTokenizer

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    openrouter_settings["provider"].pop("order")
    openrouter_settings["provider"]["allow_fallbacks"] = True

    # Initialize variables to store generation results
    reasoning_texts = [None] * len(valid_indices)  # Initialize with correct size
    num_workers = min(50, len(valid_indices))  # Number of parallel workers
    generation_ids = []  # Store all generation IDs
    generation_id_to_prompt_idx = {}  # Map generation IDs to prompt indices

    print(
        f"Generating responses with OpenRouter API for {len(valid_indices)} prompts using {num_workers} workers with hide_data enabled..."
    )

    # Load tokenizer for applying chat templates and creating logit bias
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Path to custom chat template
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"

    # Load custom chat template
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            tokenizer.chat_template = custom_template
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        print(f"Custom template not found for {args.model} at {custom_template_path}")
        print("Using default chat template")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt_text):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }

        # Always use completions endpoint
        response = requests.post(
            url="https://openrouter.ai/api/v1/completions",
            headers=headers,
            json={**params, "prompt": prompt_text},
        )
        response.raise_for_status()
        gen_id = response.json()["id"]
        output = response.json()
        return output, gen_id

    # Function to extract and tokenize user data for logit biasing
    def extract_user_data_tokens(profile):
        blocked_token_ids = set()

        # Recursively extract all values from the profile
        def extract_values(data):
            values = []
            if isinstance(data, dict):
                for value in data.values():
                    values.extend(extract_values(value))
            elif isinstance(data, list):
                for item in data:
                    values.extend(extract_values(item))
            else:
                values.append(data)
            return values

        # Get all values from the profile
        values = [
            str(v)
            for v in extract_values(profile)
            if isinstance(v, (str, int, float, bool))
        ]

        values = [
            [v, " " + v, v.lower(), " " + v.lower(), v.upper(), " " + v.upper()]
            for v in values
        ]
        values = list(set([item for sublist in values for item in sublist]))

        token_ids = [tokenizer.encode(v, add_special_tokens=False) for v in values]
        token_ids = list(set([item for sublist in token_ids for item in sublist]))
        blocked_token_ids.update(token_ids)

        return blocked_token_ids

    # Step 1: Generate reasoning for each prompt in parallel with hide_data
    # Max tokens for reasoning is max_tokens - 500
    reasoning_max_tokens = max(sampling_params.max_tokens - 500, 1000)
    print(
        f"Step 1: Generating reasoning with hide_data (max tokens: {reasoning_max_tokens})..."
    )

    # Function to process a single reasoning prompt
    def process_reasoning_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        prompt = prompts[idx]
        profile = data[idx].get("profile", {})

        # Get token IDs to block from the user's profile
        blocked_token_ids = extract_user_data_tokens(profile)

        # Create logit_bias dictionary for OpenRouter API
        logit_bias = {token_id: -100 for token_id in blocked_token_ids}

        # Format the prompt using the chat template if it's a list (chat format)
        if isinstance(prompt, list):
            formatted_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=False if "cot" in args.prompt_type else True,
                continue_final_message=True if "cot" in args.prompt_type else False,
            )
        else:
            # For non-chat prompts, use as-is
            formatted_prompt = prompt

        # Set up generation parameters for reasoning
        reasoning_params = {
            "model": model_name,
            "max_tokens": reasoning_max_tokens,
            "temperature": sampling_params.temperature,
            "logit_bias": logit_bias,  # Add logit bias to block user data tokens
        }

        if sampling_params.top_p is not None:
            reasoning_params["top_p"] = sampling_params.top_p

        # Add stop tokens to end at reasoning phase
        if end_think_token is not None:
            reasoning_params["stop"] = [end_think_token, " " + end_think_token]

        # Add OpenRouter settings
        reasoning_params.update(openrouter_settings)

        # Make API request for reasoning
        response_output, gen_id = make_api_request(reasoning_params, formatted_prompt)
        reasoning_key = "reasoning" if "reasoning" in args.prompt_type else "text"
        reasoning_text = response_output["choices"][0][reasoning_key]

        # Add end_think_token if needed
        if (
            end_think_token is not None
            and reasoning_text is not None
            and not reasoning_text.endswith(end_think_token)
        ):
            reasoning_text += end_think_token

        # Store generation ID mapping
        return prompt_idx, idx, reasoning_text, gen_id, formatted_prompt

    print(process_reasoning_prompt(0))
    # Process reasoning prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all reasoning tasks
        future_to_idx = {
            executor.submit(process_reasoning_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(
            total=len(valid_indices), desc="Step 1: Reasoning generation with hide_data"
        )

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, reasoning_text, gen_id, formatted_prompt = (
                future.result()
            )
            reasoning_texts[prompt_idx] = reasoning_text

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    # Step 2: Generate answers based on reasoning in parallel
    final_outputs = [None] * len(valid_indices)  # Initialize with correct size
    print("Step 2: Generating answers in parallel (max tokens: 500)...")

    # Max tokens for answer generation is fixed at 500
    answer_max_tokens = 500

    # Function to process a single answer generation
    def process_answer_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        orig_prompt = prompts[idx]
        reasoning_text = reasoning_texts[prompt_idx]

        # Prepare prompt for answer generation
        # Create a new chat prompt with reasoning as assistant message
        answer_messages = deepcopy(orig_prompt)
        if "cot" in args.prompt_type:
            answer_messages[1]["content"] += reasoning_text
        else:
            answer_messages.append(
                {
                    "role": "assistant",
                    "content": reasoning_text,
                }
            )

        # Format with chat template
        formatted_answer_prompt = tokenizer.apply_chat_template(
            answer_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

        # Set up generation parameters for answer
        answer_params = {
            "model": model_name,
            "max_tokens": answer_max_tokens,
            "temperature": sampling_params.temperature,
        }

        if sampling_params.top_p is not None:
            answer_params["top_p"] = sampling_params.top_p

        # Add OpenRouter settings
        answer_params.update(openrouter_settings)

        # Generate answer
        response_output, gen_id = make_api_request(
            answer_params, formatted_answer_prompt
        )
        answer_text = response_output["choices"][0]["text"]

        # Combine reasoning and answer
        combined_text = reasoning_text + answer_text

        # Create output object
        output_obj = OutputObj(combined_text)
        request_output = RequestOutputObj([output_obj], orig_prompt)

        # Store the prompt with reasoning for debugging
        data[idx]["prompt_with_reasoning"] = formatted_answer_prompt

        return prompt_idx, idx, request_output, gen_id

    # Process answer prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all answer generation tasks
        future_to_idx = {
            executor.submit(process_answer_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(total=len(valid_indices), desc="Step 2: Answer generation")

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, request_output, gen_id = future.result()
            final_outputs[prompt_idx] = request_output

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    print(f"Completed {len(final_outputs)} OpenRouter API calls with hide_data")
    return final_outputs, generation_ids, generation_id_to_prompt_idx

def _is_qwen3_thinker(args) -> bool:
    """仅用模型名判断是否走 Qwen3 思考分支。"""
    name = (getattr(args, "model", "") or "").lower()
    return "qwen3" in name
def _is_deepseek_r1_thinker(args) -> bool:
    """仅用模型名判断是否走 DeepSeek R1 思考分支。"""
    name = (getattr(args, "model", "") or "").lower()
    return "deepseek-r1" in name

def _apply_template_to_str(tokenizer, messages, enable_thinking: bool = True, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

def _ensure_close_with_phrase(s: str, end_think_token: str) -> str:
    """
    若结尾已有 </think>，保持不变；
    若没有，则在末尾追加：` Okay, I think I have finished thinking.\n</think>`
    """
    s = s.rstrip()
    if s.endswith(end_think_token):
        return s  # 已闭合，什么也不做
    # 没闭合 -> 追加收尾句 + 闭合标签
    closing = f" Okay, I think I have finished thinking.\n{end_think_token}"
    if not s.endswith("\n"):
        s += "\n"
    return s + closing


# # ========= 统一的带预算思考生成 =========
# def generate_with_budget(
#     llm, prompts, sampling_params, args, start_think_token: str, end_think_token: str
# ):
#     """
#     针对 Qwen3（思考模型）与 非 Qwen3（通用）统一的“思考预算生成”：
#       - 非 Qwen3：维持原先 llm.chat([...]) 的对话式增量写法；
#       - Qwen3：先用 tokenizer.apply_chat_template -> 字符串，再 llm.generate(prompt_str) 按预算迭代续写 <think>。
#     """
#     tokenizer = llm.get_tokenizer()
#     is_qwen3 = _is_qwen3_thinker(args)
    
#     # 选择 chat_template（若提供了自定义模板则优先）
#     custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"
#     try:
#         with open(custom_template_path, "r") as f:
#             custom_template = f.read()
#             print(f"Using custom chat template from {custom_template_path}")
#     except FileNotFoundError:
#         print(f"Custom template not found for {args.model} at {custom_template_path}, using default")
#         custom_template = tokenizer.chat_template

#     base_params = sampling_params.clone()
#     ignore_strs = ["Oh wait", "Wait", "But wait,"]
#     outputs = []
    
#     # 统一 answer 阶段 token 预算（加速 & 简化）
#     max_total = getattr(args, "max_tokens", 10000)
#     answer_budget = max(32, max_total - getattr(args, "budget_thinking", 0))

#     # ===================== Qwen3 分支：字符串 prompt + generate =====================
#     if is_qwen3:
#         prompts_with_reasoning = []
#         only_reasonings = []

#         for messages in tqdm(prompts, desc="Processing prompts (reasoning)"):
#         # for messages in prompts:
#             # 1) 到 assistant 起点（模板已自动插入 <think>），我们只需要接着写思考
#             full_prompt_str = _apply_template_to_str(tokenizer, messages, enable_thinking=True, add_generation_prompt=True)
#             # 确保最后有 "<think>" 打开（通常模板已添加）
#             if not full_prompt_str.rstrip().endswith("<think>") and start_think_token in full_prompt_str:
#                 # 若模板放在中间或末尾已不是 <think> 结尾，这里追加一个换行更安全
#                 full_prompt_str = full_prompt_str + "\n"
#             elif full_prompt_str.rstrip().endswith("<think>"):
#                 full_prompt_str += "\n"

#             remaining = args.budget_thinking
            
#             reasoning_text = ""

#             while remaining > 0:

#                 think_params = base_params.clone()
#                 think_params.max_tokens = remaining
#                 think_params.min_tokens = 1
#                 think_params.stop = [end_think_token, f" {end_think_token}"]
#                 think_params.skip_special_tokens = False
#                 think_params.include_stop_str_in_output = True

#                 think_out = llm.generate(full_prompt_str, sampling_params=think_params, use_tqdm=False)[0]
#                 text = think_out.outputs[0].text
                
#                 reasoning_text += text
                
#                 token_ids = getattr(think_out.outputs[0], "token_ids", None)
#                 tokens_used = len(token_ids) if token_ids else len(tokenizer.encode(text))

#                 remaining -= tokens_used

#                 if text.endswith(end_think_token):
#                     if remaining > 0:
#                         # 提前闭合，替换掉 </think> 继续写（防止早停）
#                         trimmed = text[: -len(end_think_token)] + random.choice(ignore_strs)
#                         full_prompt_str += trimmed
#                         continue
#                     else:
#                         # 正好到预算，直接接受
#                         full_prompt_str += text
#                         break
#                 else:
#                     full_prompt_str += text
#                     if remaining > 0:
#                         continue
#                     else:
#                         full_prompt_str += (f" Okay, I think I have finished thinking.\n{end_think_token}")
#                         break
                    
#             reasoning_text += (f" Okay, I think I have finished thinking.\n{end_think_token}")
#             # 2) 思考结束，进入答案阶段
#             if not full_prompt_str.rstrip().endswith(end_think_token):
#                 full_prompt_str += (f" Okay, I think I have finished thinking.\n{end_think_token}")
#                 reasoning_text += (f" Okay, I think I have finished thinking.\n{end_think_token}")
            
#             prompts_with_reasoning.append(full_prompt_str)
            
#             only_reasonings.append(reasoning_text)
            
#         # 批量生成答案（加速）
#         answer_params = base_params.clone()
#         answer_params.max_tokens = answer_budget

#         ans_outs = llm.generate(
#             prompts_with_reasoning,
#             sampling_params=answer_params,
#             use_tqdm=False
#         )
        
#         for i, ans in enumerate(ans_outs):
#             answer_text = ans.outputs[0].text
#             # 让输出包含“思考+Answer 提示+答案”
#             reasoning_text = only_reasonings[i]
#             if answer_text.endswith("</think>"):
#                 answer_text = answer_text[:-len("</think>")]
#             ans.outputs[0].text = reasoning_text + answer_text
#             ans.prompt = full_prompt_str
#             outputs.append(ans)
            
#         return outputs

#     # ===================== 非 Qwen3 分支：messages + chat =====================
#     prompts_with_reasoning = []
#     only_reasonings = []
#     rendered_prompts = []
#     for prompt in tqdm(prompts, desc="Processing prompts (reasoning)"):
#         full_prompt = deepcopy(prompt)
#         # 显式在 assistant 内容中打开 “思考”
#         # full_prompt.append({"role": "assistant", "content": start_think_token + "\n"})

#         remaining = args.budget_thinking
#         reasoning_text = ""

#         while remaining > 0:
#             think_params = base_params.clone()
#             think_params.max_tokens = remaining
#             think_params.min_tokens = 1
#             think_params.stop = [end_think_token, f" {end_think_token}"]
#             think_params.skip_special_tokens = False
#             think_params.include_stop_str_in_output = True

#             outs = llm.chat(
#                 [full_prompt],
#                 sampling_params=think_params,
#                 chat_template=custom_template,
#                 add_generation_prompt=False,
#                 continue_final_message=True,
#                 use_tqdm=False,
#             )
#             out = outs[0]
#             text = out.outputs[0].text
#             reasoning_text += text
            
#             token_ids = getattr(out.outputs[0], "token_ids", None)
#             tokens_used = len(token_ids) if token_ids else len(tokenizer.encode(text))
#             remaining -= tokens_used

#             if text.endswith(end_think_token):
#                 if remaining > 0:
#                     trimmed = text[: -len(end_think_token)] + random.choice(ignore_strs)
#                     full_prompt[-1]["content"] += trimmed
#                     continue
#                 else:
#                     full_prompt[-1]["content"] += text
#                     break
#             else:
#                 full_prompt[-1]["content"] += text
#                 if remaining > 0:
#                     continue
#                 else:
#                     full_prompt[-1]["content"] += (f" Okay, I think I have finished thinking.\n{end_think_token}")
#                     break
        
#         reasoning_text += (f" Okay, I think I have finished thinking.\n{end_think_token}")

#         # 闭合思考并提示回答
#         if not full_prompt[-1]["content"].rstrip().endswith(end_think_token):
#             full_prompt[-1]["content"] += (f" Okay, I think I have finished thinking.\n{end_think_token}")
#             reasoning_text += (f" Okay, I think I have finished thinking.\n{end_think_token}")

#         prompts_with_reasoning.append(full_prompt)
#         only_reasonings.append(reasoning_text)
#         rendered = tokenizer.apply_chat_template(
#             full_prompt,
#             chat_template=custom_template,
#             tokenize=False,
#             add_generation_prompt=False,
#             continue_final_message=True,
#         )
#         rendered_prompts.append(rendered)
    
#     answer_params = base_params.clone()
#     answer_params.max_tokens = answer_budget

#     ans_outs = llm.chat(
#         prompts_with_reasoning,
#         sampling_params=answer_params,
#         chat_template=custom_template,
#         add_generation_prompt=False,
#         continue_final_message=True,
#         use_tqdm=False,
#     )
    
#     for i, ans in enumerate(ans_outs):
#         answer_text = ans.outputs[0].text
#         # 输出 = 渲染后的“思考+Answer: ” + 答案
#         if answer_text.endswith("</think>"):
#             answer_text = answer_text[:-len("</think>")]
#         ans.outputs[0].text = only_reasonings[i] + answer_text
#         ans.prompt = rendered_prompts[i]
#         outputs.append(ans)        
        # 
    # return outputs
    
    
def generate_with_budget(
    llm, prompts, sampling_params, args, start_think_token: str, end_think_token: str
):
    tokenizer = llm.get_tokenizer()
    is_qwen3 = _is_qwen3_thinker(args)
    is_deep_r1 = _is_deepseek_r1_thinker(args)
    IGNORE_STRS = ["Oh wait", "Wait", "But wait,"]  # 用于提前闭合后继续写

    # 选模板
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        custom_template = tokenizer.chat_template

    base_params = sampling_params.clone()
    N = len(prompts)

    # -------------------------- Qwen3  and DeepSeek R1分支 --------------------------
    if is_qwen3 or is_deep_r1:
        # 初始化：渲染到 <think> 起点
        raw_prompt_strs = []
        for messages in prompts:
            s = _apply_template_to_str(tokenizer, messages, enable_thinking=True, add_generation_prompt=True)
            if s.rstrip().endswith("<think>"):
                s += "\n"
            raw_prompt_strs.append(s)

        final_reasonings = [""] * N        # 仅“思考”文本
        final_prompt_strs = [""] * N       # 闭合后的完整 prompt（如果你还要存）
        remaining = [args.budget_thinking] * N

        active_indices = list(range(N))
        active_prompts = raw_prompt_strs[:]  # 和 active_indices 一一对应

        while active_indices:
            # 每轮用该批里的最小剩余，避免超预算（若你用 per-request SamplingParams 就更优了）
            step = max(1, max(remaining[i] for i in active_indices))
            print(f"The new iteration max step {step}")
            think_params = base_params.clone()
            think_params.max_tokens = step
            think_params.min_tokens = 1
            think_params.stop = [end_think_token, f" {end_think_token}"]
            think_params.skip_special_tokens = False
            think_params.include_stop_str_in_output = True

            outs = llm.generate(active_prompts, sampling_params=think_params)
            
            next_idx, next_prompts = [], []

            for k, out in enumerate(outs):
                #idx: 原位置
                idx = active_indices[k]
                text = out.outputs[0].text
                token_ids = getattr(out.outputs[0], "token_ids", None)
                
                # —— 新增：按样本的剩余预算截断 ——
                if token_ids is not None:
                    # 截到本样本的剩余预算
                    if len(token_ids) > remaining[idx]:
                        token_ids = token_ids[: remaining[idx]]
                        text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    used = len(token_ids)
                else:
                    # 没有 token_ids 可退化用 encode 估计（可能与真实切分略有出入）
                    est_ids = tokenizer.encode(text, add_special_tokens=False)
                    if len(est_ids) > remaining[idx]:
                        est_ids = est_ids[: remaining[idx]]
                        text = tokenizer.decode(est_ids, skip_special_tokens=True)
                    used = len(est_ids)                
                
                
                remaining[idx] -= used

                closed = text.rstrip().endswith(end_think_token)
                
                if remaining[idx] > 0:
                    if closed:
                        # 提前闭合：删掉 </think>，接 filler 继续写
                        trimmed = text[: -len(end_think_token)] + random.choice(IGNORE_STRS)
                        final_reasonings[idx] += trimmed
                        next_prompts.append(active_prompts[k] + trimmed)
                    else:
                        next_prompts.append(active_prompts[k] + text)
                        final_reasonings[idx] += text
                    next_idx.append(idx)
                else:
                    # 预算用完，规范闭合+收尾句
                    suffix = _ensure_close_with_phrase(text, end_think_token)
                    final_reasonings[idx] += suffix
                    final_prompt_strs[idx] = active_prompts[k] + suffix
                    
            active_indices, active_prompts = next_idx, next_prompts

        # 规范化“思考文本”的闭合
        for i in range(N):
            final_reasonings[i] = _ensure_close_with_phrase(final_reasonings[i], end_think_token)

        # 统一 answer 阶段 token 预算，加速
        answer_params = base_params.clone()
        max_total = getattr(args, "max_tokens", 10000)
        answer_budget = max(32, max_total - getattr(args, "budget_thinking", 0))
        answer_params.max_tokens = answer_budget

        # 批量生成“答案阶段”
        ans_outs = llm.generate(
            final_prompt_strs,         # 直接用字符串 prompt（Qwen3 分支）；非 Qwen3 用 chat 版同理
            sampling_params=answer_params,
        )

        # 把“思考 + 答案”拼回到同一个 outputs[i].text 里，保持顺序不变
        outputs = []
        for i, ans in enumerate(ans_outs):
            answer_text = ans.outputs[0].text
            # 万一模型把 </think> 又复述了一次，做个稳妥的清理（只清理末尾的多余闭合）
            if answer_text.endswith(end_think_token):
                answer_text = answer_text[:-len(end_think_token)]
            if answer_text.startswith(end_think_token):
                answer_text = answer_text[len(end_think_token):]
                
            ans.outputs[0].text = final_reasonings[i] + answer_text
            ans.prompt = raw_prompt_strs[i]        # 记录用于生成答案的最终 prompt
            outputs.append(ans)

        return outputs

    # -------------------------- 非 Qwen3 分支 --------------------------
    final_reasonings = [""] * N
    final_rendered_prompts = [""] * N
    final_msgs = [None] * N               # ★ 新增：保存每条样本思考阶段结束后的 messages
    remaining = [args.budget_thinking] * N

    active_indices = list(range(N))
    active_msgs = [deepcopy(m) for m in prompts]


    while active_indices:
        step = max(1, max(remaining[i] for i in active_indices))
        print(f"The new iteration max step {step}")

        think_params = base_params.clone()
        think_params.max_tokens = step
        think_params.min_tokens = 1
        think_params.stop = [end_think_token, f" {end_think_token}"]
        think_params.skip_special_tokens = False
        think_params.include_stop_str_in_output = True

        outs = llm.chat(
            active_msgs,
            sampling_params=think_params,
            chat_template=custom_template,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        next_idx, next_msgs = [], []

        for k, out in enumerate(outs):
            idx = active_indices[k]
            text = out.outputs[0].text
            token_ids = getattr(out.outputs[0], "token_ids", None)
            
            # —— 新增：按样本的剩余预算截断 ——
            if token_ids is not None:
                # 截到本样本的剩余预算
                if len(token_ids) > remaining[idx]:
                    token_ids = token_ids[: remaining[idx]]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                used = len(token_ids)
            else:
                # 没有 token_ids 可退化用 encode 估计（可能与真实切分略有出入）
                est_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(est_ids) > remaining[idx]:
                    est_ids = est_ids[: remaining[idx]]
                    text = tokenizer.decode(est_ids, skip_special_tokens=True)
                used = len(est_ids)                
            
            
            remaining[idx] -= used 

            closed = text.rstrip().endswith(end_think_token)

            if remaining[idx] > 0:
                if closed:
                    trimmed = text[: -len(end_think_token)] + random.choice(IGNORE_STRS)
                    final_reasonings[idx] += trimmed
                    active_msgs[k][-1]["content"] += trimmed
                else:
                    active_msgs[k][-1]["content"] += text
                    final_reasonings[idx] += text
                next_idx.append(idx)
                next_msgs.append(active_msgs[k])
            else:
                # 预算用尽：规范闭合+收尾句
                suffix = _ensure_close_with_phrase(text, end_think_token)
                
                active_msgs[k][-1]["content"] += suffix
                
                final_reasonings[idx] += suffix

                final_msgs[idx] = deepcopy(active_msgs[k])          # ★ 新增：关键一步
                
                rendered = tokenizer.apply_chat_template(
                    active_msgs[k],
                    chat_template=custom_template,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
                final_rendered_prompts[idx] = rendered

        active_indices, active_msgs = next_idx, next_msgs

    # 规范化“思考文本”的闭合
    for i in range(N):
        final_reasonings[i] = _ensure_close_with_phrase(final_reasonings[i], end_think_token)

    # ★★★ 这里就是 prompts_with_reasoning（答案阶段的输入）
    prompts_with_reasoning = final_msgs
    
    answer_params = base_params.clone()
    max_total = getattr(args, "max_tokens", 10000)
    answer_budget = max(32, max_total - getattr(args, "budget_thinking", 0))
    answer_params.max_tokens = answer_budget

    ans_outs = llm.chat(
        prompts_with_reasoning,
        sampling_params=answer_params,
        chat_template=custom_template,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    outputs = []
    for i, ans in enumerate(ans_outs):
        answer_text = ans.outputs[0].text
        # 模型有时会把 </think> 再复述，做个稳妥清理（首尾都处理一下）
        if answer_text.rstrip().endswith(end_think_token):
            answer_text = answer_text[:-len(end_think_token)]
        if answer_text.rstrip().startswith(end_think_token):
            answer_text = answer_text[len(end_think_token):]
            
        ans.outputs[0].text = final_reasonings[i] + answer_text
        ans.prompt = tokenizer.apply_chat_template(
                    prompts[i],
                    chat_template=custom_template,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
        
        outputs.append(ans)        
        
    return outputs    
    


def generate_with_swap(
    llm,
    prompts,
    data,
    valid_indices,
    args,
    model_name,
    start_think_token,
    end_think_token,
    sampling_params=None,
):
    """Implement the Reason-Swap-Answer (RSwA) approach with a local model.

    This function orchestrates the RSwA pipeline:
    1. Generate an initial reasoning trace from the model.
    2. Swap sensitive values in the reasoning with alternatives based on reference data.
    3. Feed the swapped reasoning back to the model to generate the final answer.

    Parameters
    ----------
    llm : vllm.LLM
        The vLLM object for generation.
    prompts : list
        The list of prompts.
    data : list of dict
        The dataset, containing original and alternative references for swapping.
    valid_indices : list of int
        Indices of the prompts/data to process.
    args : argparse.Namespace
        Command-line arguments.
    model_name : str
        The name of the model being used.
    start_think_token : str
        The token to prepend to the reasoning.
    end_think_token : str
        The token signaling the end of the reasoning phase.
    sampling_params : vllm.SamplingParams, optional
        Sampling parameters for generation.

    Returns
    -------
    list of RequestOutputObj
        A list of final outputs with swapped reasoning and answers.
    """
    from copy import deepcopy

    print("Starting RSwA generation process")

    # Step 1: Generate reasoning (stop at end_think_token)
    reasoning_sampling_params = deepcopy(sampling_params)
    if end_think_token is not None:
        reasoning_sampling_params.stop = [end_think_token, " " + end_think_token]

    # Path to custom chat template for swap flow
    custom_template_path = f"chat_templates/rana/{model_name.replace('/', '_')}.jinja"
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
    except FileNotFoundError:
        custom_template = llm.get_tokenizer().chat_template

    # Set max tokens to max_tokens - 500 for reasoning
    original_max_tokens = reasoning_sampling_params.max_tokens
    reasoning_sampling_params.max_tokens = max(original_max_tokens - 500, 1000)

    print(
        f"Step 1: Generating initial reasoning (max tokens: {reasoning_sampling_params.max_tokens})..."
    )

    reasoning_outputs = llm.chat(
        prompts,
        sampling_params=reasoning_sampling_params,
        chat_template=custom_template,
        add_generation_prompt=True,
        continue_final_message=False,
    )

    # Step 2: Collect and prepare reasoning for swapping
    reasoning_texts = []
    for i in range(len(reasoning_outputs)):
        reasoning_text = reasoning_outputs[i].outputs[0].text
        if (
            end_think_token is not None
            and reasoning_text is not None
            and not reasoning_text.endswith(end_think_token)
        ):
            reasoning_text += end_think_token
        reasoning_texts.append(reasoning_text)

    # Step 2: Swap values in reasoning in parallel
    print("Step 2: Swapping reasoning values in parallel...")
    swapped_results = swap_reasonings_parallel(reasoning_texts, data, valid_indices)

    # Store swapped reasoning in data
    swapped_reasoning_list = []
    for i, idx in enumerate(valid_indices):
        reasoning_text = reasoning_texts[i]
        swapped_text, mapping = swapped_results[i]

        data[idx]["original_reasoning"] = reasoning_text
        data[idx]["swap_mapping"] = mapping

        swapped_reasoning_list.append(swapped_text)

    # Step 3: Create new prompts with swapped reasoning
    print("Step 3: Generating answers based on swapped reasoning...")
    answer_prompts = []

    for i, idx in enumerate(valid_indices):
        prefix = start_think_token + "\n" + swapped_reasoning_list[i]
        new_prompt = deepcopy(prompts[i])
        new_prompt.append(
            {
                "role": "assistant",
                "content": prefix,
            }
        )
        answer_prompts.append(new_prompt)

    answer_sampling_params = deepcopy(sampling_params)
    answer_sampling_params.max_tokens = 500

    print(f"Generating answers with max_tokens: {answer_sampling_params.max_tokens}")

    # Generate answers based on swapped reasoning
    answer_outputs = llm.chat(
        answer_prompts,
        sampling_params=answer_sampling_params,
        chat_template=custom_template,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Step 4: Combine reasoning and answers
    print("Step 4: Combining reasoning and answers...")
    final_outputs = []
    for i, idx in enumerate(valid_indices):
        answer_text = answer_outputs[i].outputs[0].text
        combined_text = swapped_reasoning_list[i] + answer_text

        output_obj = OutputObj(combined_text)
        request_output = RequestOutputObj(
            [output_obj],
            llm.get_tokenizer().apply_chat_template(
                answer_prompts[i],
                tokenize=False,
                chat_template=custom_template,
                add_generation_prompt=False,
                continue_final_message=True,
            ),
        )
        final_outputs.append(request_output)

    return final_outputs


def generate_with_openrouter_swap(
    prompts,
    data,
    valid_indices,
    model_name,
    sampling_params,
    args,
    start_think_token,
    end_think_token,
):
    """Implement the Reason-Swap-Answer (RSwA) approach using the OpenRouter API.

    This function orchestrates the RSwA pipeline with OpenRouter as the backend:
    1. Generate reasoning in parallel for each prompt.
    2. Swap sensitive values in the reasoning with alternatives from reference data.
    3. Feed the swapped reasoning back to the OpenRouter API to generate final answers.

    Parameters
    ----------
    prompts : list
        The list of prompts.
    data : list of dict
        The dataset, containing references for swapping.
    valid_indices : list of int
        Indices of prompts/data to process.
    model_name : str
        The name of the model on OpenRouter.
    sampling_params : object
        An object with sampling parameters.
    args : argparse.Namespace
        Command-line arguments.
    start_think_token : str
        Token to prepend to the reasoning.
    end_think_token : str
        Token to signal the end of the reasoning phase.

    Returns
    -------
    tuple
        - list of RequestOutputObj: The final generated outputs.
        - list of str: The generation IDs from OpenRouter.
        - dict: A mapping from generation IDs to prompt indices.
    """
    import time
    from copy import deepcopy

    from transformers import AutoTokenizer

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    if model_name == "deepseek/deepseek-chat":
        openrouter_settings["provider"].pop("order")
        openrouter_settings["provider"]["allow_fallbacks"] = True

    # Step 1: Generate reasoning for each prompt in parallel
    reasoning_texts = [None] * len(valid_indices)
    num_workers = min(50, len(valid_indices))
    generation_ids = []
    generation_id_to_prompt_idx = {}

    print(
        f"Generating responses with OpenRouter API for {len(valid_indices)} prompts using {num_workers} workers in RSwA mode..."
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load custom chat template for swap flow
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            tokenizer.chat_template = custom_template
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        print(f"Custom template not found for {args.model} at {custom_template_path}")
        print("Using default chat template")
        custom_template = tokenizer.chat_template

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt_text):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }
        response = requests.post(
            url="https://openrouter.ai/api/v1/completions",
            headers=headers,
            json={**params, "prompt": prompt_text},
        )
        response.raise_for_status()
        data_json = response.json()
        return data_json, data_json.get("id")

    # Function to process a single reasoning prompt
    def process_reasoning_prompt(idx):
        i = valid_indices[idx]
        prompt = prompts[i]
        if isinstance(prompt, list):
            formatted = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
        else:
            formatted = prompt
        params = {
            "model": model_name,
            "max_tokens": max(sampling_params.max_tokens - 500, 1000),
            "temperature": sampling_params.temperature,
        }
        if sampling_params.top_p is not None:
            params["top_p"] = sampling_params.top_p
        if end_think_token is not None:
            params["stop"] = [end_think_token, " " + end_think_token]
        params.update(openrouter_settings)
        output_json, gen_id = make_api_request(params, formatted)
        key = "reasoning" if "reasoning" in args.prompt_type else "text"
        text = output_json["choices"][0][key]
        if end_think_token and not text.endswith(end_think_token):
            text += end_think_token
        generation_ids.append(gen_id)
        generation_id_to_prompt_idx[gen_id] = i
        return idx, text

    # Generate reasonings in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_reasoning_prompt, idx): idx
            for idx in range(len(valid_indices))
        }
        for future in as_completed(futures):
            idx, text = future.result()
            reasoning_texts[idx] = text

    # Step 2: Swap reasoning values in parallel
    print("Step 2: Swapping reasoning values in parallel...")
    swapped_results = swap_reasonings_parallel(reasoning_texts, data, valid_indices)
    swapped_reasoning_list = []
    for i, idx in enumerate(valid_indices):
        swap_text, mapping = swapped_results[i]
        data[idx]["original_reasoning"] = reasoning_texts[i]
        data[idx]["swap_mapping"] = mapping
        swapped_reasoning_list.append(swap_text)

    # Step 3: Generate answers based on swapped reasoning
    final_outputs = [None] * len(valid_indices)
    num_workers_ans = min(50, len(valid_indices))

    def process_answer(idx):
        i = valid_indices[idx]
        orig_prompt = prompts[i]
        swap_text = swapped_reasoning_list[idx]
        reasoning_with_start = start_think_token + "\n" + swap_text
        if isinstance(orig_prompt, list):
            messages = deepcopy(orig_prompt)
            messages.append({"role": "assistant", "content": reasoning_with_start})
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            formatted = orig_prompt + reasoning_with_start
        params = {
            "model": model_name,
            "max_tokens": 500,
            "temperature": sampling_params.temperature,
        }
        if sampling_params.top_p is not None:
            params["top_p"] = sampling_params.top_p
        params.update(openrouter_settings)
        output_json, gen_id = make_api_request(params, formatted)
        answer = output_json["choices"][0].get("text")
        combined = reasoning_with_start + answer
        output_obj = OutputObj(combined)
        request_output = RequestOutputObj([output_obj], formatted)
        generation_ids.append(gen_id)
        generation_id_to_prompt_idx[gen_id] = i
        return idx, request_output

    with ThreadPoolExecutor(max_workers=num_workers_ans) as executor:
        futures = {
            executor.submit(process_answer, idx): idx
            for idx in range(len(valid_indices))
        }
        for future in as_completed(futures):
            idx, out = future.result()
            final_outputs[idx] = out

    print(f"Completed {len(final_outputs)} OpenRouter API calls with RSwA")
    return final_outputs, generation_ids, generation_id_to_prompt_idx
