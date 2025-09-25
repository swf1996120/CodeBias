import os
import subprocess
import time


# PROMPT_TYPES = ["vanilla"]
MODEL_PROVIDER = {
  "meta-llama/Llama-3.1-8B-Instruct": "vllm",
  "Qwen/Qwen2.5-Coder-7B-Instruct": "vllm",
  "Qwen/Qwen3-8B": "vllm",
  "/home/tangxinran/QueryAttack/models/Llama2-13B": "vllm",
  "/home/tangxinran/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1": "vllm",
  "Qwen/Qwen3-14B": "vllm",
  "/home/tangxinran/QueryAttack/models/qwen3-32B": "vllm",
  "/home/tangxinran/QueryAttack/models/deepseek-coder-7b-instruct-v1.5": "vllm",
  "/home/tangxinran/QueryAttack/models/CodeLlama-7b-Instruct-hf": "vllm",
}
PY_SCRIPT = "eval_cp_privacy_V1.py"

def run_function_evaluation_vllm(scenarior=None, category=None, attribute=None):
    SCENARIORS = ["func_edu", "func_job", "multi_med"]
    for model in MODEL:
        # 如果指定了scenarior/category/attribute，则只运行指定路径
        if scenarior and category and attribute:
            attr_file = f"./{scenarior}/{category}/{attribute}.jsonl"
            if not os.path.isfile(attr_file):
                print(f"File not found: {attr_file}")
                return
            for prompt_type in PROMPT_TYPES:
                cmd = [
                    "python", PY_SCRIPT,
                    "--model", model,
                    "--scenarior", scenarior,
                    "--category", category,
                    "--attribute", attribute,
                    "--prompt_type", prompt_type,
                    "--model_provider", f"{MODEL_PROVIDER[model]}",
                    "--max_tokens", "8000" if "llama2" not in model.lower() and "deepseek-coder-7b-instruct-v1.5" not in model.lower() else "4096",
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, env=os.environ.copy())
            return  # 只跑一次指定路径后直接返回
        
        elif (scenarior and category):
            scenarior_path = os.path.join(".", scenarior)
            if not os.path.isdir(scenarior_path):
                print(f"Directory not found: {scenarior_path}")
                return
            category_path = os.path.join(scenarior_path, category)
            if not os.path.isdir(category_path):
                print(f"Directory not found: {category_path}")
                return
            for attr_file in os.listdir(category_path):
                if not attr_file.endswith(".jsonl"):
                    continue
                attr = attr_file[:-6]
                print(f"Processing: Model={model}, Scenario={scenarior}, Category={category}, Attribute={attr}")
                time.sleep(2)  # 避免过快启动多个进程
                for prompt_type in PROMPT_TYPES:
                    cmd = [
                        "python", PY_SCRIPT,
                        "--model", model,
                        "--scenarior", scenarior,
                        "--category", category,
                        "--attribute", attr,
                        "--prompt_type", prompt_type,
                        "--model_provider", f"{MODEL_PROVIDER[model]}",
                        "--max_tokens", "8000" if "llama2" not in model.lower() and "deepseek-coder-7b-instruct-v1.5" not in model.lower() else "4096",
                    ]
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, env=os.environ.copy())
            return  # 只跑一次指定路径后直接返回

        elif scenarior:
            scenarior_path = os.path.join(".", scenarior)
            if not os.path.isdir(scenarior_path):
                print(f"Directory not found: {scenarior_path}")
                return
            for cat in os.listdir(scenarior_path):
                category_path = os.path.join(scenarior_path, cat)
                if not os.path.isdir(category_path):
                    continue
                for attr_file in os.listdir(category_path):
                    if not attr_file.endswith(".jsonl"):
                        continue
                    attr = attr_file[:-6]
                    print(f"Processing: Model={model}, Scenario={scenarior}, Category={cat}, Attribute={attr}")
                    time.sleep(2)  # 避免过快启动多个进程
                    for prompt_type in PROMPT_TYPES:
                        cmd = [
                            "python", PY_SCRIPT,
                            "--model", model,
                            "--scenarior", scenarior,
                            "--category", cat,
                            "--attribute", attr,
                            "--prompt_type", prompt_type,
                            "--model_provider", f"{MODEL_PROVIDER[model]}",
                            "--max_tokens", "8000" if "llama2" not in model.lower() and "deepseek-coder-7b-instruct-v1.5" not in model.lower() else "4096",
                        ]
                        print("Running:", " ".join(cmd))
                        subprocess.run(cmd, env=os.environ.copy())
            return  # 只跑一次指定路径后直接返回

        # 否则全量遍历
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
                    print(f"Processing: Model={model}, Scenario={scen}, Category={cat}, Attribute={attr}")
                    time.sleep(2)  # 避免过快启动多个进程
                    for prompt_type in PROMPT_TYPES:
                        cmd = [
                            "python", PY_SCRIPT,
                            "--model", model,
                            "--scenarior", scen,
                            "--category", cat,
                            "--attribute", attr,
                            "--prompt_type", prompt_type,
                            "--model_provider", f"{MODEL_PROVIDER[model]}",
                            "--max_tokens", "10000" if "llama2" not in model.lower() and "deepseek-coder-7b-instruct-v1.5" not in model.lower() else "4096",
                        ]
                        print("Running:", " ".join(cmd))
                        subprocess.run(cmd, env=os.environ.copy())

if __name__ == "__main__":
    # 用法1：全量遍历
    MODEL = ["/home/tangxinran/QueryAttack/models/qwen3-32B"]
    PROMPT_TYPES = ["vanilla", "cot"]
    run_function_evaluation_vllm(scenarior="func_job", category="race", attribute="asian")
    # 用法2：只运行指定路径
    # run_function_evaluation(scenarior="func_edu", category="gender", attribute="male"2)