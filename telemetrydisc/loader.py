"""
Data loader for telemetry log files
"""

import collections
from functools import partial, reduce
import math
from matplotlib import pyplot
import pandas as pd
from scipy.optimize import curve_fit
import statistics
from typing import List, Optional, Tuple, Union

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


def smooth_data(data: Union[pd.DataFrame, pd.Series], window_size: int):
    if isinstance(data, pd.DataFrame):
        smoothed_data = pd.DataFrame(index=data.index)
        for column in data:
            smoothed_data[column] = smooth_data(data[column], window_size)
        return smoothed_data
    else:
        window = []  # type: List[SeriesValue]
        smoothed_data = pd.Series(index=data.index)
        for t in data.index:
            window.append(SeriesValue(t, data[t]))
            while window[-1].t - window[0].t > window_size:
                window.pop(0)
            if len(window) > (window_size / 10):
                window_avg = statistics.mean([sv.value for sv in window])
                smoothed_data[t] = window_avg
            else:
                smoothed_data[t] = data[t]
        return smoothed_data


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
    crosses = (cross-y1)*(x2-x1)/(y2-y1) + x1
    return pd.Series(crosses, crosses)


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
    raw_data = pd.read_csv(filename, names=LOG_COLUMNS, index_col="timeMS")
    df = raw_data
    df["dt"] = pd.Series(df.index, index=df.index).diff()
    for col in LOG_COLUMNS[1:]:
        df[f"d_{col}"] = df[col].diff()
    flights = [flight for flight in find_idle(df["d_gyroZ"], 150, 5)
               if get_slice(df["gyroZ"], flight).mean() > 500]
    print(f"{len(flights)} flights detected.")
    create_plot(df, "gyroZ", "../tests", True)

    for flight in flights:
        flight_data = get_slice(df, flight)
        ideal_index = [i for i in range(flight_data.index.min(), flight_data.index.max(), 1)]

        for axis in ["accelX", "accelY"]:
            pyplot.suptitle("composite")
            fig, accel_axis = pyplot.subplots()
            rpm_axis = accel_axis.twinx()

            a_est = flight_data[axis].max()
            b_est = 1 / (pd.Series(find_crossings(flight_data[axis].diff()[1:])).diff().mean() * 2)
            d_est = flight_data[axis].mean()

            ArgTup = collections.namedtuple("ArgTup", ["a", "b", "theta", "d"])
            b_ArgTup = collections.namedtuple("ArgTup", ["c"])

            def func(xs, a, b, theta, d):
                return [a * math.sin(b * 2 * math.pi * x + theta) + d for x in xs]

            popt, pcov = curve_fit(
                func,
                [x for x in flight_data[axis].index],
                flight_data[axis],
                # p0=[a_est, b_est, 0, d_est],
                p0=[a_est, b_est, 0, d_est],
            )
            popt = ArgTup(*popt)

            def biased_func(xs, c):
                return [popt.a * math.sin(popt.b * 2 * math.pi * x ** (1 - c) + popt.theta) + popt.d for x in xs]

            b_popt, b_pcov = curve_fit(
                biased_func,
                [x for x in flight_data[axis].index],
                flight_data[axis],
                p0=[0],
            )
            b_popt = b_ArgTup(*b_popt)

            ideal_ys = pd.Series([x - popt.b for x in biased_func(ideal_index, *b_popt)], ideal_index)
            crossings = find_crossings(ideal_ys.diff()[1:])
            periods = crossings.diff().iloc[1:]
            rpms = periods.apply(lambda x: 60 / (x * 2 / 1000))

            period_est = (periods.mean() * 2) / 1000  # Seconds
            rpm_est = 60 / period_est
            print(f"    {flight}  Avg RPM: {round(rpm_est)}")

            raw_line = accel_axis.plot(flight_data[axis] - popt.b, color="red", linewidth=1, label="Raw")
            fit_line = accel_axis.plot(ideal_index, ideal_ys, color="purple", linewidth=1, label="Fit")
            rpm_line = rpm_axis.plot(rpms, color="blue", linewidth=1, label="RPMs")

            lns = raw_line + fit_line + rpm_line
            labs = [l.get_label() for l in lns]
            accel_axis.legend(lns, labs, loc=0)

            pyplot.savefig(f"../tests/composite_{axis}_{flight.start}_{flight.end}.png", dpi=300, format="png")
            pyplot.clf()
