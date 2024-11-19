"""
discovery_utils.utils.charts

Utils for making charts
"""

import altair as alt


DEF_COLOUR = "#0000FF"
FONTSIZE_TITLE = 16
FONTSIZE_SUBTITLE = 13
FONTSIZE_NORMAL = 13
FONT = "Arial"


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
