"""
Data loader for telemetry log files
"""

import collections
from matplotlib import pyplot
import os
import pandas as pd
import statistics

WINDOW_SIZE = 150  # Size of the sliding window for throw detection (ms)
WINDOW_THRESHOLD = 30  # Abs value mean to threshold

SeriesValue = collections.namedtuple("SeriesValue", ["t", "value"])


LOG_COLUMNS = [
    "timeMS",
    "accelX",
    "accelY",
    "accelZ",
    "gyroX",
    "gyroY",
    "gyroZ",
    "magX",
    "magY",
    "magZ",
]


def isolate_throws(data: pd.DataFrame):
    start_t = None
    throws = []
    window = []
    for t in data.index:
        next_value = SeriesValue(t, data["gyroZ"][t])
        window.append(next_value)
        while next_value.t - window[0].t > WINDOW_SIZE:
            window.pop(0)
        avg = statistics.mean([abs(sv.value) for sv in window])
        if start_t is None and avg >= WINDOW_THRESHOLD:
            start_t = window[0].t
        if start_t is not None and avg < WINDOW_THRESHOLD:
            end_t = window[-1].t
            throw_gyroZ = data["gyroZ"].iloc[data.index.get_loc(start_t): data.index.get_loc(end_t)]
            max_gyroZ = max([abs(throw_gyroZ.max()), abs(throw_gyroZ.min())])
            if max_gyroZ > 100:
                throws.append((start_t, end_t))
            start_t = None
    return throws


def loadf(filename: str):
    df = pd.read_csv(filename, names=LOG_COLUMNS, index_col="timeMS")
    df["dt"] = pd.Series(df.index, index=df.index).diff()
    for col in LOG_COLUMNS[1:]:
        df[f"d_{col}"] = df[col].diff()
    throws = isolate_throws(df)
    for start_t, end_t in throws:
        if not os.path.exists(f"graphs_{start_t}_{end_t}"):
            os.mkdir(f"graphs_{start_t}_{end_t}")
        throw_data = df.iloc[df.index.get_loc(start_t): df.index.get_loc(end_t)]
        for col in LOG_COLUMNS[1:]:
            pyplot.suptitle(col)
            pyplot.plot(throw_data[col], linewidth=1)
            pyplot.plot(throw_data[f"d_{col}"], linewidth=1)
            pyplot.savefig(f"graphs_{start_t}_{end_t}/{col}.png", dpi=300, format="png")
            pyplot.clf()
