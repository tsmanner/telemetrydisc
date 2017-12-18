"""
Data loader for telemetry log files
"""

import collections
from functools import reduce
from matplotlib import pyplot
import os
import pandas as pd
import statistics
from typing import List, NamedTuple, Optional, Tuple, Union

ANGULAR_VELOCITY_WINDOW_SIZE = 150  # Size of the sliding window for throw detection (ms)
ANGULAR_VELOCITY_WINDOW_THRESHOLD = 50  # Abs value mean to threshold
ANGULAR_ACCELERATION_WINDOW_SIZE = 50  # Size of the sliding window for flight detection (ms)
ANGULAR_ACCELERATION_WINDOW_THRESHOLD = 2  # Abs value mean to threshold

SeriesValue = collections.namedtuple("SeriesValue", ["t", "value"])
TimeSlice = collections.namedtuple("TimeSlice", ["start", "end"])
Throw = collections.namedtuple("Throw", ["start", "flight_start", "flight_end", "end"])


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
    start = None
    start_flight = None
    end_flight = None
    flight_candidates = []
    throws = []
    angular_velocity_window = []
    angular_acceleration_window = []
    for t in data.index:
        angular_velocity_window.append(SeriesValue(t, data["gyroZ"][t]))
        while angular_velocity_window[-1].t - angular_velocity_window[0].t > ANGULAR_VELOCITY_WINDOW_SIZE:
            angular_velocity_window.pop(0)
        angular_velocity_avg = statistics.mean([abs(sv.value) for sv in angular_velocity_window])

        angular_acceleration_window.append(SeriesValue(t, data["d_gyroZ"][t]))
        while angular_acceleration_window[-1].t - angular_acceleration_window[0].t > ANGULAR_ACCELERATION_WINDOW_SIZE:
            angular_acceleration_window.pop(0)
        angular_acceleration_avg = statistics.mean([abs(sv.value) for sv in angular_acceleration_window])

        if start is None and angular_velocity_avg >= ANGULAR_VELOCITY_WINDOW_THRESHOLD:
            start = angular_velocity_window[0].t
        if start is not None:

            if start_flight is None and angular_acceleration_avg <= ANGULAR_ACCELERATION_WINDOW_THRESHOLD:
                start_flight = angular_acceleration_window[0].t
            if start_flight is not None and angular_acceleration_avg > ANGULAR_ACCELERATION_WINDOW_THRESHOLD:
                end_flight = angular_acceleration_window[-1].t
                flight_candidates.append((start_flight, end_flight))
                start_flight = None

            if angular_velocity_avg < ANGULAR_VELOCITY_WINDOW_THRESHOLD:
                end = angular_velocity_window[-1].t
                throw_gyroZ = data["gyroZ"].iloc[data.index.get_loc(start): data.index.get_loc(end)]
                max_gyroZ = max([abs(throw_gyroZ.max()), abs(throw_gyroZ.min())])
                if max_gyroZ > 100:
                    if len(flight_candidates) != 0:
                        flight = reduce(lambda fca, fcb: fca if fca[1] - fca[0] > fcb[1] - fcb[0] else fcb, flight_candidates)
                    throws.append(Throw(start, flight[0], flight[1], end))
                start = None
                start_flight = None
                flight_candidates = []
    return throws


def find_idle(data: pd.Series, window_size: int, threshold: Union[float, int]):
    idles = []  # type: List[TimeSlice]
    window = []  # type: List[SeriesValue]
    start = None
    for t in data.index:
        window.append(SeriesValue(t, data[t]))
        while window[-1].t - window[0].t > window_size:
            window.pop(0)
        window_avg = statistics.mean([abs(sv.value) for sv in window])
        if start is None and window_avg < threshold:
            start = window[-1].t
        if start is not None and window_avg > threshold:
            idles.append(TimeSlice(start, window[0].t))
            start = None
    return idles


def create_plot(data: pd.DataFrame, column: str, throw_dir: str, plot_derivative: bool = False):
    pyplot.suptitle(column)
    pyplot.plot(data[column], linewidth=1)
    if plot_derivative:
        pyplot.plot(data[f"d_{column}"], linewidth=1)
    pyplot.savefig(f"{throw_dir}/{column}.png", dpi=300, format="png")
    pyplot.clf()


def find_crossings(series, cross: int = 0, direction: str = 'cross'):
    """
    Given a Series returns all the index values where the data values equal
    the 'cross' value.

    Direction can be 'rising' (for rising edge), 'falling' (for only falling
    edge), or 'cross' for both edges
    """
    # Find if values are above or bellow yvalue crossing:
    above = series.values > cross
    below = series.values <= cross
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    # x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    # Calculate x crossings with interpolation using formula for a line:
    x1 = series.index.values[idxs]
    x2 = series.index.values[idxs+1]
    y1 = series.values[idxs]
    y2 = series.values[idxs+1]
    return pd.Series((cross-y1)*(x2-x1)/(y2-y1) + x1)


def get_slice(data: Union[pd.DataFrame, pd.Series],
              slc: Union[int, Tuple[int, int]],
              slc_end: Optional[int] = None):
    if isinstance(slc, tuple):
        start = slc[0]
        end = slc[1]
    else:
        start = slc
        end = slc_end
    return data[data.index.get_loc(start): data.index.get_loc(end)]


def loadf(filename: str):
    print(f"Processing '{filename}'...")
    df = pd.read_csv(filename, names=LOG_COLUMNS, index_col="timeMS")
    df["dt"] = pd.Series(df.index, index=df.index).diff()
    for col in LOG_COLUMNS[1:]:
        df[f"d_{col}"] = df[col].diff()
    flights = [flight for flight in find_idle(df["d_gyroZ"], 150, 5)
               if get_slice(df["gyroZ"], flight).mean() > 500]
    print(f"{len(flights)} flights detected.")
    [print(f"    {ts}") for ts in flights]
    create_plot(df, "gyroZ", "../tests", True)

    pyplot.suptitle("composite")
    fig, gyro_axis = pyplot.subplots()
    gyro_axis.plot(df["gyroZ"], linewidth=1)
    accel_axis = gyro_axis.twinx()
    accel_axis.plot(df["d_accelX"], color="purple", linewidth=1)
    accel_axis.plot(df["d_accelY"], color="orange",linewidth=1)
    pyplot.savefig(f"../tests/composite.png", dpi=300, format="png")
    pyplot.clf()

    for flight in flights:
        flight_data = get_slice(df, flight)
        pyplot.suptitle("composite")
        fig, gyro_axis = pyplot.subplots()
        gyro_axis.plot(flight_data["gyroZ"], linewidth=1)
        accel_axis = gyro_axis.twinx()
        accel_axis.plot(flight_data["d_accelX"], color="purple", linewidth=1)
        accel_axis.plot(flight_data["d_accelY"], color="orange", linewidth=1)
        pyplot.savefig(f"../tests/composite_{flight.start}_{flight.end}.png", dpi=300, format="png")
        pyplot.clf()

    exit(0)

    for flight in flights:
        flight_data = get_slice(df["d_accelX"], flight)
        # normalized_flight_data = (flight_data - flight_data.mean()) / (flight_data.max() - flight_data.min())
        x_zeros = find_crossings(flight_data)
        pyplot.suptitle("accel")
        pyplot.plot(flight_data, linewidth=1)
        flight_data = get_slice(df["d_accelY"], flight)
        # normalized_flight_data = (flight_data - flight_data.mean()) / (flight_data.max() - flight_data.min())
        y_zeros = find_crossings(flight_data)
        pyplot.plot(flight_data, linewidth=1)
        pyplot.savefig(f"../tests/accel_{flight.start}_{flight.end}.png", dpi=300, format="png")
        pyplot.clf()
        rpm_est = None
        rpm_est_x = None
        rpm_est_y = None
        if len(x_zeros) > 1:
            rpm_est_x = statistics.mean([60 / (item * 2 / 1000) for item in x_zeros.diff()][1:])
        if len(y_zeros) > 1:
            rpm_est_y = statistics.mean([60 / (item * 2 / 1000) for item in y_zeros.diff()][1:])
        if rpm_est_x is not None and rpm_est_x is not None:
            rpm_est = statistics.mean([rpm_est_x, rpm_est_y])
        elif rpm_est_x is None:
            rpm_est = rpm_est_y
        elif rpm_est_y is None:
            rpm_est = rpm_est_x
        print(rpm_est)

    exit(0)

    throws = isolate_throws(df)
    out_dir = os.path.join(os.path.dirname(filename), filename.split(".")[0])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for throw in throws:
        print(throw)
        throw_dir = os.path.join(out_dir, f"graphs_{throw.start}_{throw.end}")
        if not os.path.exists(throw_dir):
            os.mkdir(throw_dir)
        throw_data = df.iloc[df.index.get_loc(throw.start): df.index.get_loc(throw.end)]
        for col in ["gyroX", "gyroY", "gyroZ"]:
            create_plot(throw_data, col, throw_dir, True)
        for col in ["accelX", "accelY", "accelZ"]:
            create_plot(throw_data, col, throw_dir)
        pyplot.suptitle("composite")
        fig, gyro_axis = pyplot.subplots()
        gyro_axis.plot(throw_data["gyroZ"], linewidth=1)

        # accel_axis.plot(throw_data["d_gyroZ"], color="orange", linewidth=1)
        accel_axis = gyro_axis.twinx()
        flight_data = df.iloc[df.index.get_loc(throw.flight_start): df.index.get_loc(throw.flight_end)]
        normalized_flight_data = (flight_data - flight_data.mean()) / (flight_data.max() - flight_data.min())
        crossings = [x for x in find_crossings(normalized_flight_data["accelY"])]
        d_crossings = [y - x for x, y in zip(crossings[:-1], crossings[1:])]
        rpms = [60 / (item * 2 / 1000) for item in d_crossings]
        if len(crossings):
            print(f"    Crossings:   {[round(x) for x in crossings]}")
        if len(d_crossings) != 0:
            print(f"    dCrossings:  {[round(x) for x in d_crossings]}")
            print(f"    Raw RPMs:    {[round(x) for x in rpms]}")
            print(f"    Average RPM: {round(60 / (statistics.mean(d_crossings) * 2 / 1000))}")
        # accel_axis.plot(crossings[1:], rpms, color="orange", linewidth=1)
        accel_axis.plot(normalized_flight_data["accelX"], color="orange", linewidth=1)
        accel_axis.plot(normalized_flight_data["accelY"], color="purple", linewidth=1)
        pyplot.savefig(f"{throw_dir}/composite.png", dpi=300, format="png")
        pyplot.clf()
        print()
