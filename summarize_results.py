#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import pandas as pd

# -----------------------------
# 固定顺序配置（可按需调整/扩展）
# -----------------------------
CATEGORY_ORDER = ["func_edu", "func_job", "multi_med"]

ATTRIBUTE_ORDER = {
    "func_edu": [
        "gender",
        "parents degree (non-degree, bachelor, master, PhD)",
        "parents income (high, medium, low)",
        "race",
    ],
    "func_job": [
        "age (young, middle, elder)",
        "gender",
        "race",
    ],
    "multi_med": [
        "gender",
        "income level",
        "insurance status",
        "race",
    ],
}

# 轻量同义归一（可继续补充）
NORMALIZE_ATTR = {
    "Parents degree (non-degree, bachelor, master, PhD)":
        "parents degree (non-degree, bachelor, master, PhD)",
    "parents income (High, Medium, Low)":
        "parents income (high, medium, low)",
    "Age (young, middle, elder)":
        "age (young, middle, elder)",
}

SUBFOLDERS = ["func_edu", "func_job", "multi_med"]
NEEDED_SHEETS = ["fairscore", "utility"]


def normalize_attr_name(s: str) -> str:
    s2 = str(s).strip()
    return NORMALIZE_ATTR.get(s2, s2)


def _resolve_model_dir(arg_path: str) -> Path:
    """
    允许传相对路径或绝对路径。不存在就原样返回用于统一的报错提示。
    """
    p = Path(arg_path)
    if p.exists():
        return p
    # 尝试兼容：若只给了文件夹名，在当前工作目录下找
    maybe = Path.cwd() / arg_path
    return maybe if maybe.exists() else p


def _find_first_excel(folder: Path) -> Optional[Path]:
    """
    在子目录中寻找第一个 .xlsx 文件；若有多个，优先包含 'result' 其后按字典序。
    """
    files = list(folder.glob("*.xlsx"))
    if not files:
        return None
    files.sort()
    # 优先包含 'result'
    for fp in files:
        if "result" in fp.name.lower():
            return fp
    return files[0]


def _pick_attribute_and_value_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    找到 Attribute 列与数值列：
      - Attribute：优先找列名（大小写不敏感）为 'attribute' 的；否则取第1列。
      - Value：若有两列则取第2列；若多列，从剩余列中选择第一个数值列，否则仍取第2列兜底。
    """
    cols = list(df.columns)
    # 选 Attribute 列
    attr_col = None
    for c in cols:
        if str(c).strip().lower() == "attribute":
            attr_col = c
            break
    if attr_col is None:
        attr_col = cols[0]

    # 选数值列
    value_col = None
    if len(cols) >= 2:
        # 先找数值列
        for c in cols:
            if c == attr_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break
        if value_col is None:
            value_col = cols[1]  # 兜底第二列
    else:
        # 只有一列，不合理，但兜底
        value_col = cols[0]

    return attr_col, value_col


def _read_one_sheet(fp: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(fp, sheet_name=sheet_name)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _collect_one_model(model_dir: Path, model_label: str) -> dict:
    """
    返回 {sheet_name: DataFrame}，每个 DF 长表结构：
    ['Model','Category','Attribute','Value']
    """
    out = {s: [] for s in NEEDED_SHEETS}

    for sub in SUBFOLDERS:
        subdir = model_dir / sub
        if not subdir.exists():
            print(f"[WARN] skip missing subfolder: {model_dir.name}/{sub}", file=sys.stderr)
            continue

        excel_fp = _find_first_excel(subdir)
        if excel_fp is None:
            print(f"[WARN] no .xlsx in: {model_dir.name}/{sub}", file=sys.stderr)
            continue

        for sheet in NEEDED_SHEETS:
            df = _read_one_sheet(excel_fp, sheet)
            if df is None:
                print(f"[WARN] sheet '{sheet}' not found or empty in {excel_fp.name}", file=sys.stderr)
                continue

            # 清洗列
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]

            attr_col, val_col = _pick_attribute_and_value_columns(df)

            slim = df[[attr_col, val_col]].copy()
            slim.rename(columns={attr_col: "Attribute", val_col: "Value"}, inplace=True)
            # 归一属性名
            slim["Attribute"] = slim["Attribute"].astype(str).map(normalize_attr_name)
            # 加上 Model / Category
            slim.insert(0, "Model", model_label)
            slim.insert(1, "Category", sub)

            # 强制数值化
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slim["Value"] = pd.to_numeric(slim["Value"], errors="coerce")

            out[sheet].append(slim)

    # 合并
    merged = {}
    for sheet in NEEDED_SHEETS:
        merged[sheet] = pd.concat(out[sheet], ignore_index=True) if out[sheet] else pd.DataFrame(
            columns=["Model", "Category", "Attribute", "Value"]
        )
    return merged


def _sort_with_fixed_order(df: pd.DataFrame, model_order: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    # 固定 Model 顺序（按传入顺序）
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    # 固定 Category 顺序
    df["Category"] = pd.Categorical(df["Category"], categories=CATEGORY_ORDER, ordered=True)

    # Attribute 排序键
    def attr_key(row):
        cat = row["Category"]
        attr = row["Attribute"]
        order_list = ATTRIBUTE_ORDER.get(cat, [])
        try:
            return order_list.index(attr)
        except ValueError:
            return len(order_list) + 999  # 未知属性放最后，便于排查

    df["_attr_order"] = df.apply(attr_key, axis=1)

    # 稳定排序（mergesort 保持相等元素输入相对顺序）
    df = df.sort_values(by=["Model", "Category", "_attr_order"], kind="mergesort").drop(columns="_attr_order")
    return df


def main():
    # if len(sys.argv) < 3:
    #     print(
    #         "用法:\n  python summarize_results.py <output.xlsx> <model_dir_1> [<model_dir_2> ...]\n\n"
    #         "示例:\n  python summarize_results.py all_models_summary.xlsx "
    #         "results/BASE/Qwen2.5-Coder-7B-Instruct_cot_Any results/BASE/Qwen2.5-Coder-7B-Instruct_vanilla_Any",
    #         file=sys.stderr,
    #     )
    #     sys.exit(1)

    out_xlsx = "all_models_summary.xlsx"
    evaluation_models = [
                        "Llama-3.1-8B-Instruct",
                        #  "Qwen2.5-Coder-7B-Instruct",
                        #  "Qwen3-8B",
                        #  "Llama2-13B",
                        #  "6a6f4aa4197940add57724a7707d069478df56b1",
                        #  "Qwen3-14B",
                        #  "qwen3-32B",
                        #  "deepseek-coder-7b-instruct-v1.5",
                        #  "CodeLlama-7b-Instruct-hf",
                        #  "DeepSeek-R1-Distill-Qwen-7B"
                        ]
    prompt = ["cot"]
    Strategy = "ATTACK"
    budge_limitations = [50, 100, 300, 1000]
    # tempetures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    model_dirs_args = []
    for evaluation_model in evaluation_models:
        for prompt_type in prompt:
            # for tempeture in tempetures:
            for budge_limitation in budge_limitations:
                model_dirs_args.append(f"results/{Strategy}/{evaluation_model}_{prompt_type}_{budge_limitation}")

    # 解析/存在性检查，并记录模型显示名（文件夹名）
    model_dirs: List[Path] = []
    model_labels: List[str] = []
    for arg in model_dirs_args:
        p = _resolve_model_dir(arg)
        model_dirs.append(p)
        model_labels.append(p.name)

    # 收集
    buckets = {s: [] for s in NEEDED_SHEETS}
    for p, label in zip(model_dirs, model_labels):
        if not p.exists():
            print(f"[WARN] model folder not found: {p}", file=sys.stderr)
            # 仍然推进，留空
            continue
        merged = _collect_one_model(p, label)
        for s in NEEDED_SHEETS:
            buckets[s].append(merged[s])

    # 合并所有模型
    dfs = {}
    for s in NEEDED_SHEETS:
        df_all = pd.concat(buckets[s], ignore_index=True) if buckets[s] else pd.DataFrame(
            columns=["Model", "Category", "Attribute", "Value"]
        )
        df_all = _sort_with_fixed_order(df_all, model_labels)
        dfs[s] = df_all

    # 写出到一个 Excel：两个 sheet
    out_path = Path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        # 表头顺序
        cols = ["Model", "Category", "Attribute", "Value"]
        for s in NEEDED_SHEETS:
            df = dfs[s]
            # 统一列顺序
            df = df.reindex(columns=cols)
            df.to_excel(w, sheet_name=s, index=False)

    print(f"[OK] Saved summary to: {out_path.resolve()}")


if __name__ == "__main__":
    main()