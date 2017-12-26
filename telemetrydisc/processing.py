"""
Data loader for telemetry log files
"""

from functools import reduce
import math
from matplotlib import pyplot
import pandas as pd
from scipy.optimize import curve_fit
import statistics
from typing import Iterable, List, Optional, Tuple, Union

from telemetrydisc.database import get_logs_table, get_raw_data
from telemetrydisc.util import *

ANGULAR_VELOCITY_WINDOW_SIZE = 150  # Size of the sliding window for throw detection (ms)
ANGULAR_VELOCITY_WINDOW_THRESHOLD = 50  # Abs value mean to threshold
ANGULAR_ACCELERATION_WINDOW_SIZE = 50  # Size of the sliding window for flight detection (ms)
ANGULAR_ACCELERATION_WINDOW_THRESHOLD = 2  # Abs value mean to threshold


def process_all():
    logs = get_logs_table()
    for crc in logs.index:
        process_log(crc)



import itertools


class sliding_window:
    def __init__(self, collection: Iterable, window: int, post_window: Optional[int] = None):
        # if len(collection) < (window * 2 + 1):
        #     raise ValueError("sliding_window collection must be at least (window * 2 + 1) in size")
        self._iterator = iter(collection)
        self._pre_window = window
        self._post_window = window if post_window is None else post_window
        self._pre = None
        self._now = None
        self._post = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._pre is None:
            self._pre = list(itertools.islice(self._iterator, self._pre_window))
        else:
            self._pre.pop(0)
            self._pre.append(self._now)
        if self._now is None:
            self._now = self._iterator.__next__()
        else:
            self._now = self._post[0]
        if self._post is None:
            self._post = list(itertools.islice(self._iterator, self._post_window))
        else:
            self._post.pop(0)
            self._post.append(self._iterator.__next__())
        return self._pre, self._now, self._post


def smooth(data: pd.Series, *args, window: Optional[int] = 15, iterations: Optional[int] = None):
    if iterations is not None:
        smoothed = data.copy()
        for i in range(iterations):
            smoothed = smooth(smoothed, window=window)
        return smoothed
    smoothed = pd.Series()
    for pre, now, post in sliding_window(data.iteritems(), window):
        # Do Stuff
        pre_mean = statistics.mean([item[1] for item in pre])
        post_mean = statistics.mean([item[1] for item in post])
        if pre_mean > now[1] and post_mean > now[1] or pre_mean < now[1] and post_mean < now[1]:
            smoothed.set_value(now[0], statistics.mean([pre_mean, post_mean]))
        else:
            smoothed.set_value(now[0], now[1])
    return smoothed


def find_releases(data: pd.DataFrame):
    releases = []  # type: List[List[Tuple[int, int]]]
    for pre, now, post in sliding_window(data["gyroZ"].iteritems(), 10):
        if now[1] - statistics.mean([item[1] for item in pre]) >= 500 and\
                now[1] - statistics.mean([item[1] for item in post]) <= 250:
            if len(releases) and len(releases[-1]) and pre[-1][0] == releases[-1][-1][0]:
                releases[-1].append(now)
            else:
                releases.append([now])
    return releases


def find_ends(data: pd.DataFrame):
    ends = []  # type: List[List[Tuple[int, int]]]
    for pre, now, post in sliding_window(data["gyroZ"].iteritems(), 10):
        if now[1] - statistics.mean([item[1] for item in pre]) <= 500 and\
                now[1] - statistics.mean([item[1] for item in post]) >= 250:
            if len(ends) and len(ends[-1]) and pre[-1][0] == ends[-1][-1][0]:
                ends[-1].append(now)
            else:
                ends.append([now])
    return ends


def process_log(log_crc: int):
    log_data = get_raw_data(log_crc)
    s_log_data = pd.DataFrame()
    s_log_data["gyroZ"] = smooth(log_data["gyroZ"], window=10, iterations=3)
    s_log_data["accelX"] = smooth(log_data["accelX"])
    s_log_data["accelY"] = smooth(log_data["accelY"])
    releases = [item[-1][0] for item in find_releases(s_log_data)]

    flights = []
    for n, release_range in enumerate(zip(releases, releases[1:] + [None])):
        ends = [item[0][0] for item in find_ends(s_log_data.loc[release_range[0]:release_range[1]])]
        print(f"Flight Candidate {n+1:>2}: {release_range[0]}-{ends[0]}")
        # print(f"Release Candidate {n+1:>2}: {release_range[0]}")
        # print(f"    End Candidate {n+1:>2}: {ends[0]}")
        flights.append((release_range[0], ends[0]))

    # exit()

    for flight in flights:
        output_directory = os.path.join(LOCAL_DATA_PATH, f"{log_crc}")
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        fig, l_axis = pyplot.subplots()
        r_axis = l_axis.twinx()
        pyplot.suptitle("gyroZ")
        l_axis.plot(s_log_data["gyroZ"].loc[flight[0]:flight[1]], linewidth=1)
        l_axis.plot(s_log_data["gyroZ"].diff().loc[flight[0]:flight[1]], linewidth=1)
        r_axis.plot(log_data["accelX"].loc[flight[0]:flight[1]], linewidth=1, color="g")
        r_axis.plot(log_data["accelY"].loc[flight[0]:flight[1]], linewidth=1, color="brown")
        fig.savefig(os.path.join(output_directory, f"gyroZ_{flight[0]}_{flight[1]}.png"), dpi=300, format="png")
        pyplot.close(fig)
        # pyplot.clf()


def isolate_flights(data: pd.DataFrame):
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
                        flight = reduce(lambda fca, fcb: fca if fca[1] - fca[0] > fcb[1] - fcb[0] else fcb,
                                        flight_candidates)
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
    x2 = series.index.values[idxs + 1]
    y1 = series.values[idxs]
    y2 = series.values[idxs + 1]
    crosses = (cross - y1) * (x2 - x1) / (y2 - y1) + x1
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


def process_data(output_dir: str):
    # df = pd.read_sql("SELECT * FROM raw_data", )
    df = pd.DataFrame()
    df["dt"] = pd.Series(df.index, index=df.index).diff()
    for col in LOG_COLUMNS[1:]:
        df[f"d_{col}"] = df[col].diff()
    # n_gyroZ = df["d_gyroZ"]
    # n_gyroZ = n_gyroZ[~((n_gyroZ-n_gyroZ.mean()).abs() > 3*n_gyroZ.std())]
    # df["n_d_gyroZ"] = n_gyroZ
    flights = [flight for flight in find_idle(df["d_gyroZ"], 150, 5)
               if get_slice(df["gyroZ"], flight).mean() > 500 and len(get_slice(df, flight)) > 50]
    # flights = [flight for flight in find_idle(n_gyroZ, 150, 5)
    #            if get_slice(df["gyroZ"], flight).mean() > 500 and len(get_slice(n_gyroZ, flight)) > 50]
    print(f"{len(flights)} flights detected.")
    create_plot(df, "gyroZ", output_dir, True)
    # create_plot(df, "n_d_gyroZ", log_dir)
    exit(0)

    for flight in flights:
        flight_data = get_slice(df, flight)
        ideal_index = [i for i in range(flight_data.index.min(), flight_data.index.max(), 1)]

        for axis in ["accelX", "accelY"]:
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
            print(f"    {flight}  Avg RPM: {round(rpm_est) if not math.isnan(rpm_est) else '---'}")

            raw_line = accel_axis.plot(flight_data[axis] - popt.b, color="red", linewidth=1, label="Raw")
            fit_line = accel_axis.plot(ideal_index, ideal_ys, color="purple", linewidth=1, label="Fit")
            lns = raw_line + fit_line
            if math.isnan(rpm_est):
                rpm_line = rpm_axis.plot(rpms, color="blue", linewidth=1, label="RPMs")
                lns += rpm_line

            labs = [l.get_label() for l in lns]
            accel_axis.legend(lns, labs, loc=0)

            out_dir = os.path.join(log_dir, f"flight_{flight.start}_{flight.end}")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            pyplot.savefig(os.path.join(out_dir, f"composite_{axis}.png"), dpi=300, format="png")
            pyplot.clf()
