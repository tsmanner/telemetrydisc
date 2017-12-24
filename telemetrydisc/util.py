import collections
import os

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

LOCAL_DATA_PATH = "../data"
LOCAL_DB_PATH = os.path.join(LOCAL_DATA_PATH, "local.db")
