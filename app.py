from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from peristaltic_tip_pump_data import (
    DATA_DIR,
    DEFAULT_DATA_FILE,
    TIMESTAMP_OUTPUT_FORMAT,
    apply_moving_average,
    build_stats_frame,
    downsample_frame,
    filter_by_period,
    format_timestamp,
    get_available_column_groups,
    get_display_name,
    get_default_plot_columns,
    get_numeric_columns,
    insert_gap_breaks,
    list_workspace_csv_files,
    load_data,
)


def parse_timestamp_input(value: str) -> datetime:
    normalized_value = value.strip()
    for pattern in (TIMESTAMP_OUTPUT_FORMAT[:-3], TIMESTAMP_OUTPUT_FORMAT):
        try:
            return datetime.strptime(normalized_value, pattern)
        except ValueError:
            continue

    raise ValueError(
        "Timestamp must be in YYYY/MM/DD HH:MM:SS or YYYY/MM/DD HH:MM:SS.mmm format."
    )


@st.cache_data(show_spinner=False)
def load_cached_data(
    file_path: str | None = None,
    file_mtime_ns: int | None = None,
    file_bytes: bytes | None = None,
    row_limit: int | None = None,
):
    # Keep file_mtime_ns in the cache signature so updated local CSVs invalidate cached data.
    del file_mtime_ns

    if file_path is not None:
        return load_data(path=Path(file_path), row_limit=row_limit)

    if file_bytes is not None:
        return load_data(file_bytes=file_bytes, row_limit=row_limit)

    raise ValueError("Either file_path or file_bytes must be provided.")


def normalize_series_values(y_values: list[float | None]) -> list[float | None]:
    non_null_values = [value for value in y_values if value is not None]
    if not non_null_values:
        return y_values

    min_value = min(non_null_values)
    max_value = max(non_null_values)
    if max_value == min_value:
        return [0.0 if value is not None else None for value in y_values]

    return [
        None if value is None else (value - min_value) / (max_value - min_value)
        for value in y_values
    ]


def get_default_selected_groups(group_options: dict[str, list[str]]) -> list[str]:
    if "State_Main" in group_options:
        return ["State_Main"]

    return list(group_options)[:1]


def get_default_columns(
    selected_groups: list[str],
    group_options: dict[str, list[str]],
    all_columns: list[str],
) -> list[str]:
    grouped_columns = []
    for group_name in selected_groups:
        grouped_columns.extend(group_options.get(group_name, []))

    unique_grouped_columns = list(dict.fromkeys(grouped_columns))
    return unique_grouped_columns or get_default_plot_columns(all_columns)


def get_column_options(
    all_columns: list[str],
    default_columns: list[str],
    column_search: str,
) -> list[str]:
    if not column_search:
        return all_columns

    lowered_search = column_search.lower()
    return [
        column_name
        for column_name in all_columns
        if lowered_search in column_name.lower() or column_name in default_columns
    ]


@st.cache_data(show_spinner=False)
def build_plot_html_bytes(figure_json: str) -> bytes:
    return pio.from_json(figure_json).to_html(
        include_plotlyjs="cdn",
        full_html=True,
    ).encode("utf-8")


@st.cache_data(show_spinner=False)
def build_plot_png_bytes(figure_json: str) -> bytes:
    return pio.from_json(figure_json).to_image(format="png", scale=2)


def build_figure(
    df,
    selected_columns: list[str],
    normalize_values: bool,
):
    figure = go.Figure()
    x_values = df["timestamp_dt"].to_list()

    for column_name in selected_columns:
        y_values = df[column_name].to_list()
        if normalize_values:
            y_values = normalize_series_values(y_values)

        x_values_with_gaps, y_values_with_gaps = insert_gap_breaks(x_values, y_values)

        figure.add_trace(
            go.Scatter(
                x=x_values_with_gaps,
                y=y_values_with_gaps,
                mode="lines",
                name=get_display_name(column_name),
                connectgaps=False,
            )
        )

    figure.update_layout(
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis_title="timestamp",
        yaxis_title="normalized value" if normalize_values else "value",
    )
    figure.update_xaxes(
        rangeslider_visible=True,
        showgrid=True,
        gridcolor="rgba(120, 120, 120, 0.18)",
        gridwidth=0.5,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="rgba(120, 120, 120, 0.12)",
        gridwidth=0.5,
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="Peristaltic Tip Pump Viewer",
        layout="wide",
    )
    st.title("Peristaltic Tip Pump Viewer")
    st.caption("Plotly and Streamlit viewer for Peristaltic Tip Pump CSV logs.")

    with st.sidebar:
        st.header("Controls")

        with st.expander("1. Data Load", expanded=True):
            available_files = list_workspace_csv_files(DATA_DIR)
            source_mode = st.radio(
                "Data source",
                options=["Workspace CSV", "Upload CSV"],
                index=0 if available_files else 1,
            )
            row_limit_input = st.number_input(
                "Row limit (0 = all rows)",
                min_value=0,
                value=0,
                step=1000,
            )
            row_limit = int(row_limit_input) or None

            if st.button("Clear cache"):
                st.cache_data.clear()

            if source_mode == "Workspace CSV":
                if not available_files:
                    st.info("No workspace CSV files were found. Switch to Upload CSV.")
                    st.stop()

                default_index = 0
                for index, file_path in enumerate(available_files):
                    if file_path == DEFAULT_DATA_FILE:
                        default_index = index
                        break

                selected_file = st.selectbox(
                    "Workspace file",
                    options=available_files,
                    index=default_index,
                    format_func=lambda path: path.name,
                )
                df = load_cached_data(
                    file_path=str(selected_file),
                    file_mtime_ns=selected_file.stat().st_mtime_ns,
                    row_limit=row_limit,
                )
                source_label = selected_file.name
            else:
                uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
                if uploaded_file is None:
                    st.info("Upload a CSV file to continue.")
                    st.stop()

                df = load_cached_data(
                    file_bytes=uploaded_file.getvalue(),
                    row_limit=row_limit,
                )
                source_label = uploaded_file.name

    if df.is_empty():
        st.warning("The selected CSV did not contain any rows.")
        st.stop()

    min_timestamp = df["timestamp_dt"].min()
    max_timestamp = df["timestamp_dt"].max()
    all_columns = get_numeric_columns(df)
    group_options = get_available_column_groups(all_columns)

    with st.sidebar:
        with st.expander("2. Time Range", expanded=True):
            preset = st.selectbox(
                "Range preset",
                options=["All", "Custom"],
            )

            if preset == "All":
                start_timestamp = min_timestamp
                end_timestamp = max_timestamp
            else:
                st.caption("Format: YYYY/MM/DD HH:MM:SS.mmm")
                start_input = st.text_input(
                    "Start timestamp",
                    value=format_timestamp(min_timestamp),
                )
                end_input = st.text_input(
                    "End timestamp",
                    value=format_timestamp(max_timestamp),
                )

                try:
                    start_timestamp = parse_timestamp_input(start_input)
                    end_timestamp = parse_timestamp_input(end_input)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

            if start_timestamp > end_timestamp:
                st.error("Start timestamp must be earlier than or equal to end timestamp.")
                st.stop()

        with st.expander("3. Columns", expanded=True):
            selected_groups = st.multiselect(
                "Quick groups",
                options=list(group_options.keys()),
                default=get_default_selected_groups(group_options),
            )
            column_search = st.text_input("Filter column names", value="")

            default_columns = get_default_columns(
                selected_groups,
                group_options,
                all_columns,
            )
            column_options = get_column_options(
                all_columns,
                default_columns,
                column_search,
            )

            selected_columns = st.multiselect(
                "Plot columns",
                options=column_options,
                default=[column for column in default_columns if column in column_options],
            )

        with st.expander("4. Display", expanded=True):
            normalize_values = st.checkbox("Normalize values", value=False)
            show_legend = st.checkbox("Show legend", value=True)
            downsample_step = st.slider("Plot every Nth row", min_value=1, max_value=50, value=1)
            moving_average_window = st.slider(
                "Moving average window",
                min_value=1,
                max_value=20,
                value=1,
            )

    filtered_df = filter_by_period(df, start_timestamp, end_timestamp)
    plotted_df = downsample_frame(filtered_df, downsample_step)
    plotted_df = apply_moving_average(plotted_df, selected_columns, moving_average_window)

    if not selected_columns:
        st.warning("Select at least one column to plot.")
        st.stop()

    if plotted_df.is_empty():
        st.warning("No rows matched the selected time range.")
        st.stop()

    figure = build_figure(
        plotted_df,
        selected_columns,
        normalize_values=normalize_values,
    )
    if not show_legend:
        figure.update_layout(showlegend=False)

    top_metrics = st.columns([2.2, 1, 1, 1, 1])
    top_metrics[0].caption("Source")
    top_metrics[0].markdown(source_label)
    top_metrics[1].metric("Total rows", f"{df.height:,}")
    top_metrics[2].metric("Filtered rows", f"{filtered_df.height:,}")
    top_metrics[3].metric("Plotted rows", f"{plotted_df.height:,}")
    top_metrics[4].metric("Selected columns", len(selected_columns))

    st.caption(
        f"Timestamp range: {format_timestamp(min_timestamp)} "
        f"to {format_timestamp(max_timestamp)}"
    )

    plot_tab, preview_tab, stats_tab = st.tabs(["Plot", "Data Preview", "Stats"])

    with plot_tab:
        st.plotly_chart(figure, width="stretch")

    with preview_tab:
        preview_rows = st.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)
        st.dataframe(
            filtered_df.drop("timestamp_dt").head(preview_rows),
            width="stretch",
            hide_index=True,
        )

    with stats_tab:
        stats_df = build_stats_frame(filtered_df, selected_columns)
        st.dataframe(stats_df, width="stretch", hide_index=True)

    with st.sidebar:
        with st.expander("5. Export", expanded=True):
            filtered_csv = filtered_df.drop("timestamp_dt").write_csv().encode("utf-8")
            st.download_button(
                "Download filtered CSV",
                data=filtered_csv,
                file_name="filtered_peristaltic_tip_pump.csv",
                mime="text/csv",
            )

            figure_json = figure.to_json()
            figure_html = build_plot_html_bytes(figure_json)
            st.download_button(
                "Download plot HTML",
                data=figure_html,
                file_name="peristaltic_tip_pump_plot.html",
                mime="text/html",
            )

            try:
                figure_png = build_plot_png_bytes(figure_json)
            except Exception as exc:
                st.info(f"PNG export is unavailable in the current environment: {exc}")
            else:
                st.download_button(
                    "Download plot PNG",
                    data=figure_png,
                    file_name="peristaltic_tip_pump_plot.png",
                    mime="image/png",
                )


if __name__ == "__main__":
    main()
