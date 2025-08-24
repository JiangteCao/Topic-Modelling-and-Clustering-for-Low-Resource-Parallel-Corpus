from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import pandas as pd


def compare_cross_lingual_cluster_consistency_full(
    hsb_labels: List[int],
    de_labels: List[int],
    hsb_sentences: List[str] = None,
    de_sentences: List[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Build a mapping report from HSB clusters to DE clusters.
    Returns, for each HSB cluster id:
      - top_german_cluster: the DE cluster with highest overlap
      - consistency_ratio : top overlap / total in that HSB cluster (rounded to 2 decimals)
      - distribution_full : raw counts per DE cluster
      - distribution_percent : percentages as strings per DE cluster
    """
    if len(hsb_labels) != len(de_labels):
        raise ValueError("hsb_labels and de_labels must have the same length.")

    cluster_map: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(hsb_labels):
        cluster_map[label].append(idx)

    report: Dict[int, Dict[str, Any]] = {}

    for cluster_id, indices in cluster_map.items():
        german_cluster_ids = [de_labels[i] for i in indices]
        german_counts = Counter(german_cluster_ids)
        total = len(german_cluster_ids)

        top_cluster, top_count = german_counts.most_common(1)[0]
        ratio = top_count / total if total > 0 else 0.0

        percent_dist = {k: f"{(v / total) * 100:.1f}%" for k, v in german_counts.items()} if total > 0 else {}

        report[cluster_id] = {
            "top_german_cluster": int(top_cluster),
            "consistency_ratio": round(ratio, 2),
            "distribution_full": {int(k): int(v) for k, v in german_counts.items()},
            "distribution_percent": {int(k): v for k, v in percent_dist.items()},
        }

    return report


def describe_crosslingual_cluster_mapping_full(report: Dict[int, Dict[str, Any]]) -> None:
    """
    Pretty-print the mapping report to stdout.
    """
    for cid in sorted(report.keys()):
        info = report[cid]
        print(f"\nUpper Sorbian Cluster {cid}:")
        print(
            f"  → Most mapped to German Cluster {info['top_german_cluster']} "
            f"({info['consistency_ratio']*100:.1f}% of sentences)"
        )
        print(f"  → Full German cluster distribution (percentage): {info['distribution_percent']}")
        print(f"  → Raw distribution: {info['distribution_full']}")


def convert_cluster_report_to_dataframe_full(report: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the mapping report to a flat DataFrame.
    Columns include:
      - Upper_Sorbian_Cluster
      - Top_German_Cluster
      - Top_Cluster_Ratio
      - German_Cluster_{id}_%
      - German_Cluster_{id}_count
    """
    rows: List[Dict[str, Any]] = []
    for cid in sorted(report.keys()):
        info = report[cid]
        row: Dict[str, Any] = {
            "Upper_Sorbian_Cluster": cid,
            "Top_German_Cluster": info["top_german_cluster"],
            "Top_Cluster_Ratio": info["consistency_ratio"],
        }

        for gcid, percent in info.get("distribution_percent", {}).items():
            row[f"German_Cluster_{gcid}_%"] = percent

        for gcid, count in info.get("distribution_full", {}).items():
            row[f"German_Cluster_{gcid}_count"] = count

        rows.append(row)

    return pd.DataFrame(rows)


def save_report_csv(report: Dict[int, Dict[str, Any]], csv_path: str) -> None:
    """
    Save the report as a CSV via the DataFrame view.
    """
    df = convert_cluster_report_to_dataframe_full(report)
    df.to_csv(csv_path, index=False)
