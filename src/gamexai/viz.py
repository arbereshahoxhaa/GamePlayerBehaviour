"""Shared Plotly styling for the dashboard.

Colours come from the validated data-viz reference palette (categorical slots in
fixed order; a sequential blue ramp for ordered magnitude; a blue<->red diverging
pair for signed SHAP). Engagement Low<Medium<High is *ordered*, so it uses the
ordinal blue ramp, not categorical hues.
"""
import plotly.graph_objects as go

# Categorical slots 1-5 (light surface), fixed order.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, ORANGE, AQUA, YELLOW = CAT[0], CAT[1], CAT[2], CAT[3]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"            # status (reserved)

# Ordered engagement classes -> ordinal blue (light->dark = low->high).
CLASS_COLORS = {"Low": "#86b6ef", "Medium": "#3987e5", "High": "#184f95"}
SEQ_BLUE = [[0, "#eef5fe"], [0.5, "#5598e7"], [1, "#184f95"]]
DIVERGING = {"pos": BLUE, "neg": CRITICAL}       # SHAP supports / opposes

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig, height=360, legend=True, margin_t=40):
    fig.update_layout(
        font=dict(family=FONT, color=INK2, size=13),
        title=dict(font=dict(color=INK, size=16), x=0, xanchor="left"),
        margin=dict(l=10, r=16, t=margin_t, b=10),
        height=height, paper_bgcolor=SURF, plot_bgcolor=SURF,
        hoverlabel=dict(font_size=13, font_family=FONT, bgcolor="white"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=12), title_text=""),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, color=MUTED, linecolor=AXIS, ticks="outside",
                     tickcolor=AXIS, title_font=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, color=MUTED, linecolor=AXIS,
                     zeroline=False, title_font=dict(size=12))
    return fig


def hbar(labels, values, color=BLUE, title="", suffix="", height=360):
    """Horizontal bar, largest at top, rounded data-ends, direct value labels."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=color,
        marker_line_width=0,
        text=[f"{v:.3f}{suffix}" for v in values], textposition="outside",
        textfont=dict(color=INK2, size=12),
        hovertemplate="%{y}: %{x:.3f}" + suffix + "<extra></extra>"))
    style(fig, height=height, legend=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=title, xaxis_title="", yaxis_title="")
    return fig
