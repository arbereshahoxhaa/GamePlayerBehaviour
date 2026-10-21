"""Shared Plotly styling for the dashboard (dark theme).

Colours come from the validated data-viz reference palette, dark-surface steps
(categorical slots in fixed order; an ordinal blue ramp for ordered engagement;
a blue<->red diverging pair for signed SHAP). Chart backgrounds are transparent
so they sit on the Streamlit dark surface.
"""
import plotly.graph_objects as go

# Categorical slots 1-5, dark-surface steps, fixed order.
CAT = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
BLUE, ORANGE, AQUA, YELLOW = CAT[0], CAT[1], CAT[2], CAT[3]
GOOD, CRITICAL = "#0ca30c", "#e06666"            # status (reserved)

# Ordered engagement classes -> ordinal blue, bright enough for a dark surface.
CLASS_COLORS = {"Low": "#9ec5f4", "Medium": "#5598e7", "High": "#2a78d6"}
SEQ_BLUE = [[0, "#1b2740"], [1, "#2f7fda"]]       # heatmap, dark -> bright blue
DIVERGING = {"pos": BLUE, "neg": CRITICAL}        # SHAP supports / opposes

INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#8f8d87"
GRID, AXIS, SURF = "#2c2c2a", "#44443f", "#1a1a19"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig, height=360, legend=True, margin_t=40):
    fig.update_layout(
        font=dict(family=FONT, color=INK2, size=13),
        title=dict(font=dict(color=INK, size=16), x=0, xanchor="left"),
        margin=dict(l=10, r=16, t=margin_t, b=10),
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=13, font_family=FONT, bgcolor="#26261f",
                        font_color=INK, bordercolor=AXIS),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=12, color=INK2), title_text=""),
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
