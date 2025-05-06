from enum import Enum

import pandas as pd
import plotly.graph_objects as go


class AggregationEnum(str, Enum):
    sum = "sum"
    mean = "mean"


def ts_plot(  # noqa: PLR0912
    df: pd.DataFrame,
    x: str,
    y: list[str],
    agg_func: AggregationEnum = AggregationEnum.sum,
    freq: str | None = None,
    label: str | None = None,
    closed: str | None = None,
    dropdown: list[str] | str | None = None,
    limit_dropdown: int | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    color: list[str] | None = None,
    sort: bool = True,
) -> go.Figure:
    """Plot one ore more timeseries in plotly html chart.

    Args:
        df (pd.DataFrame): Timeseries data.
        x (str): Date
        y (List[str]): Values
        agg_func (AggregationEnum): Aggregation function.
        freq (str | None, optional): Timeseries aggregation. Defaults to None.
        label (str | None, optional): _description_. Defaults to None.
        closed (str | None, optional): _description_. Defaults to None.
        dropdown (str | None, optional): Timeseries groups. Defaults to None.
        limit_dropdown (int | None, optional): Limit the numbers of the dropdown.
        skipna (bool, optional): Skip na's in groupby sum. Defaults to False.
        title (str | None, optional): Plot title.
        xaxis_title (str | None, optional): Plot xaxis title.
        yaxis_title (str | None, optional): Plot yaxis title.
        legend_title (str | None, optional): Plot legend title.
        color (List[str] | None, optional): Line color.
        sort (bool, optional): Sort dropdown list by values by total amount of x.

    Returns:
        go.Figure: Ploty figure
    """
    fig = go.Figure(layout=go.Layout(colorway=color) if color else None)

    updatemenu = []
    buttons = []

    n_lines = len(y)

    if dropdown:
        fig.update_layout(
            annotations=[
                dict(
                    text=", ".join(dropdown)
                    if isinstance(dropdown, list)
                    else dropdown,
                    x=1,
                    xref="paper",
                    y=1.3,
                    yref="paper",
                    showarrow=False,
                )
            ]
        )

        if freq:
            df = (
                df.set_index(x)
                .groupby(dropdown, sort=False)[y]
                .resample(freq, closed=closed, label=label)
                .apply(agg_func)  # type: ignore
                .reset_index()
            )

        if sort:
            df["total"] = df.groupby(dropdown)[y[0]].transform(lambda x: x.sum())
            df.sort_values(by=["total"], ascending=False, inplace=True)

        if limit_dropdown:
            df = df.merge(
                df[dropdown].drop_duplicates().head(limit_dropdown),
                how="inner",
                on=dropdown,
            )

        n_groups = df.groupby(dropdown).ngroups

        for i, group in zip(
            range(0, n_groups * n_lines, n_lines),
            df.groupby(dropdown, sort=False),
        ):
            visibility = [False] * (n_groups * n_lines)
            visibility[i : i + n_lines] = [True] * n_lines

            group[1].sort_values(x, inplace=True)

            for _y in y:
                fig.add_trace(
                    go.Scatter(
                        x=group[1][x].values,
                        y=group[1][_y].values,
                        visible=True if len(buttons) == 0 else False,
                        name=_y,
                    )
                )

            buttons.append(
                dict(
                    method="restyle",
                    label=", ".join(group[0])
                    if isinstance(group[0], tuple)
                    else group[0],
                    args=[
                        {"visible": visibility},
                    ],
                )
            )

        updatemenu.append(
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1,
                "y": 1.2,
            }
        )

    else:
        if freq:
            df = df.set_index(x)[y].resample(freq).apply(agg_func).reset_index()

        for _y in y:
            fig.add_trace(
                go.Scatter(x=df[x].values, y=df[_y].values, visible=True, name=_y)
            )

    updatemenu.append(
        dict(
            type="buttons",
            direction="left",
            buttons=list(
                [
                    dict(args=["type", "line"], label="Line", method="restyle"),
                    dict(args=["type", "bar"], label="Bar", method="restyle"),
                ]
            ),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0,
            xanchor="left",
            y=-0.5,
            yanchor="top",
        )
    )

    fig.update_layout(
        showlegend=True if len(y) > 1 else False,
        updatemenus=updatemenu,
    )

    if title:
        fig.update_layout(title_text=title)

    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)

    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)

    if legend_title:
        fig.update_layout(legend_title=legend_title)

    if width:
        fig.update_layout(width=width)

    if height:
        fig.update_layout(height=height)

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=14, label="14d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )

    return fig
