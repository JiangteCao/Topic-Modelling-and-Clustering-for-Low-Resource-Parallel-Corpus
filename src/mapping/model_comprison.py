import os
import pandas as pd

def load_and_merge_model_summaries(base_path: str) -> pd.DataFrame:
    paths = {
        "LaBSE": os.path.join(base_path, "labse", "labse_k_summary.csv"),
        "Glot500": os.path.join(base_path, "glot500", "glot500_k_summary.csv"),
        "XLM-R": os.path.join(base_path, "xlm-r", "xlmr_k_summary.csv"),
        "Laser": os.path.join(base_path, "laser", "laser_k_summary.csv"),
    }

    df_all = pd.DataFrame()
    for model_name, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Model"] = model_name
            df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all


def normalize_and_score_across_models(df: pd.DataFrame, alpha=0.5, beta=0.5) -> pd.DataFrame:
    df = df.copy()

    # normalize top mapping ratio
    top_min, top_max = df["avg_top_mapping_ratio"].min(), df["avg_top_mapping_ratio"].max()
    df["TopRatio_norm"] = (df["avg_top_mapping_ratio"] - top_min) / (top_max - top_min + 1e-12)

    # normalize entropy
    ent_min, ent_max = df["avg_entropy"].min(), df["avg_entropy"].max()
    df["Entropy_norm"] = (df["avg_entropy"] - ent_min) / (ent_max - ent_min + 1e-12)

    # combined score (higher is better)
    df["Combined_Score"] = alpha * df["TopRatio_norm"] + beta * (1 - df["Entropy_norm"])

    return df.sort_values(by=["k", "Combined_Score"], ascending=[True, False])
