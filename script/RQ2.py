
import os
import subprocess
import time

MODEL_PROVIDER = {
    "/home/tangxinran/QueryAttack/models/Llama-3.1-8B-Instruct": "vllm",
    "/home/tangxinran/QueryAttack/models/Qwen2.5-Coder-7B-Instruct": "vllm",
    "/home/tangxinran/QueryAttack/models/Qwen3-8B": "vllm",
    "/home/tangxinran/QueryAttack/models/Llama2-13B": "vllm",
    "/home/tangxinran/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1": "vllm",
    "/home/tangxinran/QueryAttack/models/Qwen3-14B": "vllm",
    "/home/tangxinran/QueryAttack/models/qwen3-32B": "vllm",
    "/home/tangxinran/QueryAttack/models/deepseek-coder-7b-instruct-v1.5": "vllm",
    "/home/tangxinran/QueryAttack/models/CodeLlama-7b-Instruct-hf": "vllm",
    "/home/tangxinran/QueryAttack/models/DeepSeek-R1-Distill-Qwen-7B": "vllm"
}
PY_SCRIPT = "eval_cp_privacy_V1.py"
MAX_RETRIES = 10  # 最多重试次数

# 你可以根据需要调整推理长度
REASONING_LENGTHS = [64, 128, 256, 512, 1024, 2048]

def run_reasoning_length_experiment():
    SCENARIORS = ["func_edu", "func_job", "multi_med"]
    MODEL = [
    # "/home/tangxinran/QueryAttack/models/Llama-3.1-8B-Instruct",
    # "/home/tangxinran/QueryAttack/models/Qwen2.5-Coder-7B-Instruct",
    # "/home/tangxinran/QueryAttack/models/Qwen3-8B",
    # "/home/tangxinran/QueryAttack/models/Llama2-13B",
    # "/home/tangxinran/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1",
    # "/home/tangxinran/QueryAttack/models/Qwen3-14B",
    # "/home/tangxinran/QueryAttack/models/qwen3-32B",
    # "/home/tangxinran/QueryAttack/models/deepseek-coder-7b-instruct-v1.5",
    # "/home/tangxinran/QueryAttack/models/CodeLlama-7b-Instruct-hf",
    "/home/tangxinran/QueryAttack/models/DeepSeek-R1-Distill-Qwen-7B"
    ]
    PROMPT_TYPES = ["cot"]
    for model in MODEL:
        for scen in SCENARIORS:
            scenarior_path = os.path.join(".", scen)
            if not os.path.isdir(scenarior_path):
                continue
            for cat in os.listdir(scenarior_path):
                category_path = os.path.join(scenarior_path, cat)
                if not os.path.isdir(category_path):
                    continue
                for attr_file in os.listdir(category_path):
                    if not attr_file.endswith(".jsonl"):
                        continue
                    attr = attr_file[:-6]
                    for prompt_type in PROMPT_TYPES:
                        for reasoning_length in REASONING_LENGTHS:
                            print(f"Processing: Model={model}, Scenario={scen}, Category={cat}, Attribute={attr}, PromptType={prompt_type}, ReasoningLength={reasoning_length}")
                            time.sleep(2)
                            cmd = [
                                "python", PY_SCRIPT,
                                "--model", model,
                                "--scenarior", scen,
                                "--category", cat,
                                "--attribute", attr,
                                "--prompt_type", prompt_type,
                                "--model_provider", f"{MODEL_PROVIDER[model]}",
                                "--max_tokens", "8000" if "llama2" not in model.lower() and "deepseek-coder-7b-instruct-v1.5" not in model.lower() else "4096",
                                "--budget_thinking", str(reasoning_length)
                            ]
                            # 输出文件名建议在eval_cp_privacy_V1.py里自动带上reasoning_length
                            print("Running:", " ".join(cmd))
                            retries = 0
                            while retries < MAX_RETRIES:
                                try:
                                    result = subprocess.run(cmd, env=os.environ.copy(), check=True)
                                    break  # 成功就退出循环
                                except subprocess.CalledProcessError as e:
                                    retries += 1
                                    print(f"Error running command (attempt {retries}/{MAX_RETRIES}): {e}")
                                    if retries >= MAX_RETRIES:
                                        print(f"Command failed after {MAX_RETRIES} retries, skipping...")
                                    else:
                                        print("Retrying...")
                                        time.sleep(60)  # 等待 5 秒再重试
                                
if __name__ == "__main__":
    run_reasoning_length_experiment()