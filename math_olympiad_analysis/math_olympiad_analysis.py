import json, io, zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

EVAL_ZIP = Path(r"logs")

def read_json_from_zip(zp: zipfile.ZipFile, name: str) -> Any:
    with zp.open(name) as f:
        return json.load(io.TextIOWrapper(f, encoding="utf-8"))

def parse_accuracy(item: Dict[str, Any]) -> Optional[float]:
    # From your data: scores -> olympiadbench_scorer -> value in {"C","I"}
    try:
        v = item["scores"]["olympiadbench_scorer"]["value"]
    except Exception:
        return None
    if isinstance(v, str):
        v = v.strip().upper()
        if v == "C":  # correct
            return 1.0
        if v == "I":  # incorrect
            return 0.0
    # Fallbacks if schema differs
    score = item.get("score")
    if isinstance(score, (int, float)):
        return float(np.clip(score, 0.0, 1.0))
    return None

def parse_t_human(item: Dict[str, Any]) -> Optional[float]:
    md = item.get("metadata") or {}
    th = md.get("T_human")
    if isinstance(th, (int, float)):
        return float(th)
    if isinstance(th, str):
        try:
            return float(th)
        except ValueError:
            return None
    return None

def load_df_from_eval_zip(eval_dir_path: Path) -> pd.DataFrame:
    # Find the most recent .eval file in the directory
    eval_files = sorted(Path(eval_dir_path).glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not eval_files:
        raise FileNotFoundError("No .eval files found in the specified directory.")
    eval_zip_path = eval_files[0]

    with zipfile.ZipFile(eval_zip_path, "r") as zp:
        names = set(zp.namelist())
        if "summaries.json" not in names:
            raise FileNotFoundError("summaries.json not found in .eval ZIP.")
        data = read_json_from_zip(zp, "summaries.json")
        if not isinstance(data, list):
            raise ValueError("summaries.json did not contain a list of items.")
    rows: List[Dict[str, Any]] = []
    for it in data:
        rows.append({
            "id": it.get("id"),
            "epoch": it.get("epoch"),
            "source": (it.get("metadata") or {}).get("source"),
            "subfield": (it.get("metadata") or {}).get("subfield"),
            "accuracy": parse_accuracy(it),
            "t_human": parse_t_human(it),
            "support_auto_scoring": (it.get("metadata") or {}).get("support_auto_scoring"),
            "exam": (it.get("metadata") or {}).get("exam"),
            "answer_type": (it.get("metadata") or {}).get("answer_type"),
        })
    df = pd.DataFrame(rows)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["t_human"] = pd.to_numeric(df["t_human"], errors="coerce")
    return df


# ---- load + basic checks
df = load_df_from_eval_zip(EVAL_ZIP)
print(df.head())
print("Rows with both accuracy & t_human:", df.dropna(subset=["accuracy","t_human"]).shape[0])

# Optional: exclude rows where support_auto_scoring == False
plot_df = df[df["t_human"].notna() & df["accuracy"].notna() & (df["support_auto_scoring"] != "False")].copy()
"""
# ---- Plot: accuracy vs t_human
plt.figure(figsize=(8,5))
sns.regplot(data=plot_df, x="t_human", y="accuracy", scatter_kws={"alpha":0.35})
plt.title("Accuracy vs T_human")
plt.xlabel("T_human (minutes)")
plt.ylabel("Accuracy (0/1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Optional: log(1 + t_human) if long-tailed
plt.figure(figsize=(8,5))
sns.regplot(data=plot_df, x=np.log1p(plot_df["t_human"]), y="accuracy", scatter_kws={"alpha":0.35})
plt.title("Accuracy vs log(1 + T_human)")
plt.xlabel("log(1 + T_human)")
plt.ylabel("Accuracy (0/1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""

import pandas as pd

# Example: df is your DataFrame with multiple epochs
# You already have: id, epoch, accuracy, t_human, etc.

# 1️⃣ Define aggregation rules
agg_df = (
    df.groupby(["id", "source", "subfield", "exam", "answer_type", "support_auto_scoring"], dropna=False)
      .agg({
          "accuracy": "mean",   # average accuracy across epochs
          "t_human": "mean",    # usually constant, but safe to average
          "epoch": "count"      # how many epochs were seen
      })
      .rename(columns={"epoch": "num_epochs"})
      .reset_index()
)

# 2️⃣ Optional: keep only auto-scored items
agg_df = agg_df[agg_df["support_auto_scoring"] != "False"]

print(agg_df.head())


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,5))
sns.scatterplot(data=agg_df, x="t_human", y="accuracy", s=60, alpha=0.6)
#plt.plot(agg_df["t_human"], agg_df["accuracy"], linewidth=1.2, alpha=0.7)

plt.title("Average Accuracy per Question vs T_human")
plt.xlabel("T_human (minutes)")
plt.ylabel("Average Accuracy (0–1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()