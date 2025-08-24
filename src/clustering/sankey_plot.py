import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any


def report_to_alluvial_df(report: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a cross-lingual cluster mapping report to an alluvial DataFrame
    for Sankey visualization.
    """
    rows = []
    for sorb_cluster, info in report.items():
        for ger_cluster, count in info["distribution_full"].items():
            rows.append({
                "source": f"HSB_{sorb_cluster}",
                "target": f"DE_{ger_cluster}",
                "value": count
            })
    return pd.DataFrame(rows)


def plot_sankey_from_report(
    report: Dict[int, Dict[str, Any]],
    title: str = "Cross-lingual Cluster Mapping",
    html_path: str = None
) -> go.Figure:
    """
    Create a Sankey diagram from the report and return a plotly Figure.
    If html_path is provided, also write to HTML.
    """
    df_alluvial = report_to_alluvial_df(report)

    labels = list(pd.unique(df_alluvial['source'].tolist() + df_alluvial['target'].tolist()))
    label_indices = {label: idx for idx, label in enumerate(labels)}

    df_alluvial['source_idx'] = df_alluvial['source'].map(label_indices)
    df_alluvial['target_idx'] = df_alluvial['target'].map(label_indices)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=df_alluvial['source_idx'],
            target=df_alluvial['target_idx'],
            value=df_alluvial['value'],
            label=[str(v) for v in df_alluvial['value']],
            customdata=[str(v) for v in df_alluvial['value']],
            hovertemplate='Value: %{value}<extra></extra>'
        )
    )])

    fig.update_layout(title_text=title, font_size=10)

    if html_path:
        fig.write_html(html_path)

    return fig
