import csv
import io
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = DATA_DIR / "test_data.csv"
TIMESTAMP_INPUT_FORMAT = "%Y/%m/%d %H:%M:%S"
TIMESTAMP_OUTPUT_FORMAT = "%Y/%m/%d %H:%M:%S.%f"

ASSIGNED_COLUMNS = [
    "Emergency_ds",
    "State_AutoRun",
    "State_ManualRunPM",
    "State_manualRunSM",
    "Approv_PIDCTL",
    "Approv_PID",
    "Approv_LevelSen_PID",
    "Approv_ForceGauge_PID",
    "CapacitiveSensor1_ds",
    "CapacitiveSensor2_ds",
    "CapacitiveSensor3_ds",
    "ProximitySensor_ds",
    "State_drivePM",
    "State_driveSM",
    "WarningCondition",
    "InformationCondition",
    "Timeout_delivery",
    "Alert_Pressure_PM",
    "Alert_Pressure_TN",
    "MbusErrorDetection_PM",
    "MbusErrorDetection_SM",
    "Timeout_RunPM",
    "Timeout_RunSM",
    "Timeout_RemCTL",
    "Pressure_PM",
    "Pressure_TN",
    "Level_Btank",
    "Force_Btank",
    "S_ServoPID",
    "D1_ServoPID_0",
    "D1_ServoPID_5",
    "D1_ServoPID_6",
    "D1_ServoPID_7",
    "D2_ServoPID",
    "Work_PID",
    "RotationFrequency",
    "Freq_Inverter_PM",
    "RotVelStd_SM",
    "S_ServoPID_FG",
    "D1_ServoPID_FG_0",
    "D1_ServoPID_FG_5",
    "D1_ServoPID_FG_6",
    "D1_ServoPID_FG_7",
    "D2_ServoPID_FG",
    "NC4TP_CH0",
    "NC4TP_CH1",
]

COLUMN_GROUPS = {
    "State_Main": [
        "Emergency_ds",
        "State_AutoRun",
        "State_ManualRunPM",
        "State_manualRunSM",
    ],
    "State_All": [
        "Emergency_ds",
        "State_AutoRun",
        "State_ManualRunPM",
        "State_manualRunSM",
        "State_drivePM",
        "State_driveSM",
        "WarningCondition",
        "InformationCondition",
        "Timeout_delivery",
        "Alert_Pressure_PM",
        "Alert_Pressure_TN",
        "MbusErrorDetection_PM",
        "MbusErrorDetection_SM",
        "Timeout_RunPM",
        "Timeout_RunSM",
        "Timeout_RemCTL",
    ],
    "Sensors": [
        "CapacitiveSensor1_ds",
        "CapacitiveSensor2_ds",
        "CapacitiveSensor3_ds",
        "ProximitySensor_ds",
        "Pressure_PM",
        "Pressure_TN",
        "Level_Btank",
        "Force_Btank",
        "NC4TP_CH0",
        "NC4TP_CH1",
    ],
    "PID_LevelSensor": [
        "Approv_PIDCTL",
        "Approv_PID",
        "Approv_LevelSen_PID",
        "Work_PID",
        "S_ServoPID",
        "D1_ServoPID_0",
        "D1_ServoPID_5",
        "D1_ServoPID_6",
        "D1_ServoPID_7",
        "D2_ServoPID",
        "RotationFrequency",
        "Freq_Inverter_PM",
        "RotVelStd_SM",
    ],
    "PID_ForceGauge": [
        "Approv_PIDCTL",
        "Approv_PID",
        "Approv_ForceGauge_PID",
        "Work_PID",
        "S_ServoPID_FG",
        "D1_ServoPID_FG_0",
        "D1_ServoPID_FG_5",
        "D1_ServoPID_FG_6",
        "D1_ServoPID_FG_7",
        "D2_ServoPID_FG",
        "RotationFrequency",
        "Freq_Inverter_PM",
        "RotVelStd_SM",
    ],
}

DEFAULT_PLOT_COLUMNS = [
    # "Pressure_PM",
    # "Pressure_TN",
    # "Level_Btank",
    # "Force_Btank",
]
GAP_BREAK_THRESHOLD = timedelta(seconds=2)

COLUMN_SCALE_FACTORS = {
    "Pressure_PM": 4 / 3200,  # Convert to MPa.
    "Pressure_TN": 4 / 3200,  # Convert to MPa.
    "Level_Btank": 253 / 3200 / 10,  # Convert to cm.
    "Force_Btank": 500 / 800 * 0.101971,  # Convert to kgf.
    "NC4TP_CH0": 1 / 10,  # Convert to Celsius degrees.
    "NC4TP_CH1": 1 / 10,  # Convert to Celsius degrees.
    "S_ServoPID": 253 / 3200 / 10,  # Convert to cm.
    "D1_ServoPID_0": 253 / 3200 / 10,  # Convert to cm.
    "S_ServoPID_FG": 500 / 800 * 0.101971,  # Convert to kgf.
    "D1_ServoPID_FG_0": 500 / 800 * 0.101971,  # Convert to kgf.
    "Freq_Inverter_PM": 1 / 100,  # Convert to Hz.
    "RotVelStd_SM": 1 / 4000 * 300,  # Convert to rpm.
}

COLUMN_METADATA = {
    "Pressure_PM": {"unit": "MPa"},
    "Pressure_TN": {"unit": "MPa"},
    "Level_Btank": {"unit": "cm"},
    "Force_Btank": {"unit": "kgf"},
    "NC4TP_CH0": {"unit": "degC"},
    "NC4TP_CH1": {"unit": "degC"},
    "S_ServoPID": {"unit": "cm"},
    "D1_ServoPID_0": {"unit": "cm"},
    "S_ServoPID_FG": {"unit": "kgf"},
    "D1_ServoPID_FG_0": {"unit": "kgf"},
    "Freq_Inverter_PM": {"unit": "Hz"},
    "RotVelStd_SM": {"unit": "rpm"},
}


def list_workspace_csv_files(directory: Path | None = None) -> list[Path]:
    search_dir = directory or DATA_DIR
    return sorted(path for path in search_dir.glob("*.csv") if path.is_file())


def format_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime(TIMESTAMP_OUTPUT_FORMAT)[:-3]


def decode_uploaded_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return file_bytes.decode("utf-8", errors="replace")


def is_data_row(row: list[str]) -> bool:
    if len(row) < 2:
        return False

    try:
        datetime.strptime(f"{row[0].strip()} {row[1].strip()}", TIMESTAMP_INPUT_FORMAT)
    except ValueError:
        return False

    return True


def read_first_rows(
    path: Path | None = None,
    file_bytes: bytes | None = None,
) -> tuple[list[str], list[str], int]:
    if path is not None:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            first_row = next(reader)
            second_row = next(reader, [])
    else:
        decoded_text = decode_uploaded_bytes(file_bytes or b"")
        reader = csv.reader(io.StringIO(decoded_text))
        first_row = next(reader)
        second_row = next(reader, [])

    if is_data_row(first_row):
        return first_row, first_row, 0

    if second_row and is_data_row(second_row):
        return first_row, second_row, 1

    raise ValueError("Unable to detect a valid data row in the selected CSV.")


def get_present_value_columns(df: pl.DataFrame) -> list[str]:
    present_columns = []
    for column_name in ASSIGNED_COLUMNS:
        if df[column_name].drop_nulls().len() > 0:
            present_columns.append(column_name)

    return present_columns


def load_data(
    path: Path | None = None,
    file_bytes: bytes | None = None,
    row_limit: int | None = None,
) -> pl.DataFrame:
    if path is None and file_bytes is None:
        raise ValueError("Either path or file_bytes must be provided.")

    if path is not None and file_bytes is not None:
        raise ValueError("Provide only one of path or file_bytes.")

    if path is not None and not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    _, _, skip_rows = read_first_rows(path=path, file_bytes=file_bytes)
    # Some logs change column count mid-file, so read the full supported schema and keep
    # only columns that actually contain values after parsing.
    column_schema = {
        "date": pl.String,
        "time": pl.String,
        **{column_name: pl.String for column_name in ASSIGNED_COLUMNS},
    }
    read_kwargs = {
        "has_header": False,
        "schema": column_schema,
        "n_rows": row_limit,
        "skip_rows": skip_rows,
        "truncate_ragged_lines": True,
    }

    if path is not None:
        df = pl.read_csv(path, **read_kwargs)
    else:
        df = pl.read_csv(io.BytesIO(file_bytes or b""), **read_kwargs)

    df = df.with_columns(pl.all().str.strip_chars()).with_columns(
        [
            pl.when(pl.col(column_name) == "")
            .then(None)
            .otherwise(pl.col(column_name))
            .alias(column_name)
            for column_name in column_schema
        ]
    )
    value_columns = get_present_value_columns(df)
    if not value_columns:
        raise ValueError("CSV must contain date, time, and at least one value column.")

    duplicate_counts: dict[str, int] = defaultdict(int)
    adjusted_timestamps = []
    adjusted_datetimes = []
    for date_value, time_value in zip(
        df["date"].to_list(),
        df["time"].to_list(),
        strict=True,
    ):
        base_timestamp = f"{date_value} {time_value}"
        adjusted_timestamp = datetime.strptime(
            base_timestamp, TIMESTAMP_INPUT_FORMAT
        ) + timedelta(milliseconds=100 * duplicate_counts[base_timestamp])
        adjusted_datetimes.append(adjusted_timestamp)
        adjusted_timestamps.append(format_timestamp(adjusted_timestamp))
        duplicate_counts[base_timestamp] += 1

    return (
        df.with_columns(
            pl.Series("timestamp", adjusted_timestamps),
            pl.Series("timestamp_dt", adjusted_datetimes),
        )
        .drop(["date", "time"])
        .with_columns(pl.col(value_columns).cast(pl.Float64, strict=False))
        .with_columns(
            [
                (pl.col(column_name) * scale_factor).alias(column_name)
                for column_name, scale_factor in COLUMN_SCALE_FACTORS.items()
                if column_name in value_columns
            ]
        )
        .select(["timestamp", "timestamp_dt", *value_columns])
    )


def get_numeric_columns(df: pl.DataFrame) -> list[str]:
    return [
        column_name
        for column_name, dtype in df.schema.items()
        if column_name not in {"timestamp", "timestamp_dt"} and dtype.is_numeric()
    ]


def get_available_column_groups(columns: list[str]) -> dict[str, list[str]]:
    available_groups = {}
    for group_name, group_columns in COLUMN_GROUPS.items():
        present_columns = [column for column in group_columns if column in columns]
        if present_columns:
            available_groups[group_name] = present_columns
    return available_groups


def get_default_plot_columns(columns: list[str]) -> list[str]:
    default_columns = [column for column in DEFAULT_PLOT_COLUMNS if column in columns]
    if default_columns:
        return default_columns
    return columns[: min(5, len(columns))]


def get_display_name(column_name: str) -> str:
    metadata = COLUMN_METADATA.get(column_name, {})
    label = metadata.get("label", column_name)
    unit = metadata.get("unit")
    return f"{label} [{unit}]" if unit else label


def filter_by_period(
    df: pl.DataFrame,
    start_timestamp: datetime,
    end_timestamp: datetime,
) -> pl.DataFrame:
    return df.filter(
        pl.col("timestamp_dt").is_between(
            start_timestamp,
            end_timestamp,
            closed="both",
        )
    )


def downsample_frame(df: pl.DataFrame, step: int) -> pl.DataFrame:
    if step <= 1 or df.is_empty():
        return df

    return (
        df.with_row_index("__row_index")
        .filter(pl.col("__row_index") % step == 0)
        .drop("__row_index")
    )


def apply_moving_average(
    df: pl.DataFrame,
    columns: list[str],
    window_size: int,
) -> pl.DataFrame:
    if window_size <= 1 or not columns:
        return df

    return df.with_columns(
        [
            pl.col(column_name)
            .rolling_mean(window_size=window_size, min_samples=1)
            .alias(column_name)
            for column_name in columns
        ]
    )


def build_stats_frame(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    rows = []
    for column_name in columns:
        column = df[column_name]
        rows.append(
            {
                "column": column_name,
                "count": df.height - column.null_count(),
                "null_count": column.null_count(),
                "min": column.min(),
                "max": column.max(),
                "mean": column.mean(),
                "std": column.std(),
            }
        )

    return pl.DataFrame(rows)


def insert_gap_breaks(
    x_values: list[datetime],
    y_values: list[float | None],
    gap_threshold: timedelta = GAP_BREAK_THRESHOLD,
) -> tuple[list[datetime], list[float | None]]:
    if not x_values:
        return [], []

    x_values_with_gaps = [x_values[0]]
    y_values_with_gaps = [y_values[0]]

    for index in range(1, len(x_values)):
        previous_x = x_values[index - 1]
        current_x = x_values[index]
        current_y = y_values[index]

        if current_x - previous_x > gap_threshold:
            x_values_with_gaps.append(previous_x + (current_x - previous_x) / 2)
            y_values_with_gaps.append(None)

        x_values_with_gaps.append(current_x)
        y_values_with_gaps.append(current_y)

    return x_values_with_gaps, y_values_with_gaps
