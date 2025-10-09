
import os
import subprocess
import time
from itertools import product

# 简化的 model -> provider 映射；若不在此表中默认使用 vllm
MODEL_PROVIDER = {
    "meta-llama/Llama-3.1-8B-Instruct": "vllm",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "vllm",
    "Qwen/Qwen3-8B": "vllm",
}

PY_SCRIPT = "eval_cp_privacy_V1.py"
MAX_RETRIES = 5  # 每个命令最多重试次数


def generate_sampling_configs():
    """生成要测试的一组抽样配置。

    返回值为 dict 列表，每个 dict 表示一组参数及其短名（用于日志/文件夹）。
    """
    # 仅比较 temperature；其他参数保持 eval 脚本默认值
    temps = [0.0, 0.2, 0.6, 1.0]

    configs = []
    for t in temps:
        name = f"t{str(t).replace('.', '')}"
        configs.append({
            "temperature": t,
            "short_name": name,
        })
    return configs


def run_sampling_experiment():
    # 要遍历的数据集目录
    SCENARIORS = ["func_edu", "func_job", "multi_med"]

    # 只列出典型要跑的模型（可按需扩展）
    MODELS = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen3-8B",
    ]

    PROMPT_TYPES = ["cot"]

    sampling_configs = generate_sampling_configs()

    for model in MODELS:
        provider = MODEL_PROVIDER.get(model, "vllm")
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
                        for cfg in sampling_configs:
                            short = cfg["short_name"]
                            print(f"Processing: Model={model}, Scenario={scen}, Category={cat}, Attribute={attr}, PromptType={prompt_type}, Config={short}")
                            time.sleep(1)
                            cmd = [
                                "python", PY_SCRIPT,
                                "--model", model,
                                "--scenarior", scen,
                                "--category", cat,
                                "--attribute", attr,
                                "--prompt_type", prompt_type,
                                "--model_provider", provider,
                                "--max_tokens", "8000" if "llama2" not in model.lower() else "4096",
                                "--temperature", str(cfg["temperature"]),
                                "--sample"
                            ]

                            # 输出提示：eval 脚本内部会把 sampling 名称写入结果路径
                            print("Running:", " ".join(cmd))

                            retries = 0
                            while retries < MAX_RETRIES:
                                try:
                                    subprocess.run(cmd, env=os.environ.copy(), check=True)
                                    break
                                except subprocess.CalledProcessError as e:
                                    retries += 1
                                    print(f"Error running command (attempt {retries}/{MAX_RETRIES}): {e}")
                                    if retries >= MAX_RETRIES:
                                        print(f"Command failed after {MAX_RETRIES} retries, skipping...")
                                    else:
                                        print("Retrying in 30s...")
                                        time.sleep(30)


if __name__ == "__main__":
    run_sampling_experiment()