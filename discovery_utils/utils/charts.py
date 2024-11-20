"""
discovery_utils.utils.charts

Utils and building blocks for generating charts
"""

from typing import List

import altair as alt
import pandas as pd


NESTA_COLOURS = [
    "#0000FF",
    "#FDB633",
    "#18A48C",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    "#FFFFFF",
    "#000000",
]
DEFAULT_COLOUR = NESTA_COLOURS[0]

FONTSIZE_TITLE = 16
FONTSIZE_SUBTITLE = 13
FONTSIZE_NORMAL = 13
TITLE_FONT = "Averta"
FONT = "Averta"


def test_chart() -> alt.Chart:
    """Generates a simple test chart"""
    return (
        alt.Chart(
            pd.DataFrame(
                {
                    "labels": ["A", "B", "C"],
                    "values": [10, 15, 30],
                    "label": ["This is A", "This is B", "And this is C"],
                }
            ),
            width=400,
            height=200,
        )
        .mark_bar()
        .encode(
            alt.Y("labels:O", title="Vertical axis"),
            alt.X("values:Q", title="Horizontal axis"),
            tooltip=["label", "values"],
            color="labels",
        )
        .properties(
            title={
                "anchor": "start",
                "text": ["Chart title"],
                "subtitle": ["Longer descriptive subtitle"],
                "subtitleFont": FONT,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )


def configure_plots(fig: alt.Chart, chart_title: str = "", chart_subtitle: str = "") -> alt.Chart:
    """Add titles, subtitles and configure font sizes"""
    return (
        fig.properties(
            title={
                "anchor": "start",
                "text": chart_title,
                "fontSize": FONTSIZE_TITLE,
                "subtitle": chart_subtitle,
                "subtitleFont": FONT,
                "subtitleFontSize": FONTSIZE_NORMAL,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
            labelFontSize=FONTSIZE_NORMAL,
            titleFontSize=FONTSIZE_NORMAL,
        )
        .configure_legend(
            titleFontSize=FONTSIZE_NORMAL,
            labelFontSize=FONTSIZE_NORMAL,
        )
        .configure_view(strokeWidth=0)
    )


def configure_titles(fig: alt.Chart, chart_title: str, chart_subtitle: str = "") -> alt.Chart:
    """Add titles and subtitles"""
    return fig.properties(
        title={
            "anchor": "start",
            "text": chart_title,
            "subtitle": chart_subtitle,
            "subtitleFont": FONT,
            "subtitleFontSize": FONTSIZE_SUBTITLE,
        },
    )


_line_width = 3
_stroke_dash_none = [0]
_stroke_dash_default = [5, 5]


def ts_smooth(
    ts: pd.DataFrame,
    variable: str,
    variable_title: str,
    time_column: str = "year",
    time_column_title: str = "",
    categories_to_show: List[str] = None,
    category_column: str = None,
    category_label: str = "",
    width: int = 400,
    height: int = 150,
    stroke_dash: list = _stroke_dash_none,
    tooltip: bool = True,
    line_width: float = _line_width,
    line_point_filled: bool = True,
    interpolation: str = "monotone",
) -> alt.Chart:
    """Smoothed time series plot"""
    if not isinstance(categories_to_show, list):
        ts = ts.copy().assign(**{category_column: "category"})
        categories_to_show = ["category"]
        _legend = None
    else:
        _legend = alt.Legend(orient="top", title=f"{category_label}")

    if variable in ["n_projects", "n_rounds"]:
        _format = ".0f"
    else:
        _format = ".3f"

    if tooltip:
        tooltip = [
            alt.Tooltip(f"{time_column}:O", title=time_column_title),
            alt.Tooltip(f"{category_column}:N"),
            alt.Tooltip(f"{variable}:Q", title=variable_title, format=_format),
        ]
    else:
        tooltip = []

    return (
        alt.Chart(
            (
                ts
                # Subselect time series
                .query(f"`{category_column}` in @categories_to_show")
            ),
            width=width,
            height=height,
        )
        .mark_line(
            interpolate=interpolation,
            size=line_width,
            strokeDash=stroke_dash,
            point=alt.OverlayMarkDef(size=30, filled=line_point_filled),
        )
        .encode(
            x=alt.X("year:O", title=""),
            y=alt.Y(f"{variable}:Q", title=variable_title),
            color=alt.Color(f"{category_column}:N", legend=_legend),
            tooltip=tooltip,
        )
    )


def ts_smooth_incomplete(
    ts: pd.DataFrame,
    variable: str,
    variable_title: str,
    time_column: str = "year",
    time_column_title: str = "",
    categories_to_show: List[str] = None,
    category_column: str = None,
    category_label: str = "",
    width: int = 400,
    height: int = 150,
    max_complete_year: int = 2021,
) -> alt.Chart:
    """Smoothed time series plot with incomplete years shown with dashed lines"""
    fig_solid = ts_smooth(
        ts=ts,
        variable=variable,
        variable_title=variable_title,
        time_column=time_column,
        time_column_title=time_column_title,
        categories_to_show=categories_to_show,
        category_column=category_column,
        category_label=category_label,
        width=width,
        height=height,
    ).transform_filter(f"datum.year <= {max_complete_year}")

    fig_solid_stroke = ts_smooth(
        ts=ts,
        variable=variable,
        variable_title=variable_title,
        time_column=time_column,
        time_column_title=time_column_title,
        categories_to_show=categories_to_show,
        category_column=category_column,
        category_label=category_label,
        width=width,
        height=height,
        line_point_filled=False,
    ).transform_filter(f"datum.year <= {max_complete_year}")

    fig_dashed = ts_smooth(
        ts=ts,
        variable=variable,
        variable_title=variable_title,
        time_column=time_column,
        time_column_title=time_column_title,
        categories_to_show=categories_to_show,
        category_column=category_column,
        category_label=category_label,
        width=width,
        height=height,
        stroke_dash=_stroke_dash_default,
        line_width=2,
        line_point_filled=False,
    ).transform_filter(f"datum.year >= {max_complete_year}")

    return fig_solid_stroke + fig_solid + fig_dashed


def ts_bar(
    ts: pd.DataFrame,
    variable: str,
    variable_title: str,
    time_column: str = "year",
    time_column_title: str = "",
    categories_to_show: List[str] = None,
    category_column: str = None,
    category_label: str = "",
    width: int = 400,
    height: int = 150,
    tooltip: bool = True,
    filled: bool = True,
    fillOpacity: float = 1,
) -> alt.Chart:
    """Plot grouped bar plot showing time series"""
    if not isinstance(categories_to_show, list):
        ts = ts.copy().assign(**{category_column: "category"})
        categories_to_show = ["category"]
        _legend = None
    else:
        _legend = alt.Legend(orient="top", title=f"{category_label}")
    # Tooltips
    if tooltip:
        tooltip = [
            alt.Tooltip(f"{category_column}:N", title=category_label),
            alt.Tooltip(f"{time_column}:O", title=time_column_title),
            alt.Tooltip(f"{variable}:Q", format=",.3f", title=variable_title),
        ]
    else:
        tooltip = []

    return (
        alt.Chart(
            (ts.query(f"`{category_column}` in @categories_to_show")),
            width=width,
            height=height,
        )
        .mark_bar(filled=filled, fillOpacity=fillOpacity, strokeWidth=1.5, strokeOpacity=1)
        .encode(
            alt.X(f"{time_column}:O", title=""),
            alt.Y(
                f"{variable}:Q",
                title=variable_title,
            ),
            xOffset=f"{category_column}",
            tooltip=tooltip,
            color=alt.Color(
                f"{category_column}:N",
                legend=_legend,
            ),
        )
    )


def ts_bar_incomplete(
    ts: pd.DataFrame,
    variable: str,
    variable_title: str,
    time_column: str = "year",
    time_column_title: str = "",
    categories_to_show: List[str] = None,
    category_column: str = None,
    category_label: str = "",
    width: int = 400,
    height: int = 150,
    tooltip: bool = True,
    max_complete_year: int = 2021,
) -> alt.Chart:
    """Plot grouped bar plot showing time series with incomplete years shown with fainter colours"""
    fig_solid = ts_bar(
        ts=ts,
        variable=variable,
        variable_title=variable_title,
        time_column=time_column,
        time_column_title=time_column_title,
        categories_to_show=categories_to_show,
        category_column=category_column,
        category_label=category_label,
        width=width,
        height=height,
        tooltip=tooltip,
        filled=True,
    ).transform_filter(f"datum.year <= {max_complete_year}")

    fig_stroke = ts_bar(
        ts=ts,
        variable=variable,
        variable_title=variable_title,
        time_column=time_column,
        time_column_title=time_column_title,
        categories_to_show=categories_to_show,
        category_column=category_column,
        category_label=category_label,
        width=width,
        height=height,
        tooltip=False,
        filled=False,
        fillOpacity=1,
    ).transform_filter(f"datum.year >= {max_complete_year}")

    fig_faint = ts_bar(
        ts=ts,
        variable=variable,
        variable_title=variable_title,
        time_column=time_column,
        time_column_title=time_column_title,
        categories_to_show=categories_to_show,
        category_column=category_column,
        category_label=category_label,
        width=width,
        height=height,
        tooltip=tooltip,
        filled=True,
        fillOpacity=0.25,
    ).transform_filter(f"datum.year >= {max_complete_year}")

    return fig_solid + fig_stroke + fig_faint


def nestafont() -> dict:
    """Define Nesta fonts"""
    return {
        "config": {
            "title": {"font": TITLE_FONT, "anchor": "start"},
            "subtitle": {"font": FONT},
            "axis": {"labelFont": FONT, "titleFont": FONT},
            "header": {"labelFont": FONT, "titleFont": FONT},
            "legend": {"labelFont": FONT, "titleFont": FONT},
            "range": {
                "category": NESTA_COLOURS,
                "ordinal": {"scheme": NESTA_COLOURS},  # this will interpolate the colors
            },
        }
    }


alt.themes.register("nestafont", nestafont)
alt.themes.enable("nestafont")
