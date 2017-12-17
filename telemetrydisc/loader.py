"""
Data loader for telemetry log files
"""

from matplotlib import pyplot
import pandas as pd


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


def loadf(filename: str):
    df = pd.read_csv(filename, names=LOG_COLUMNS, index_col="timeMS")
    # diffs = df.diff()
    df["dt"] = pd.Series(df.index, index=df.index).diff()
    # df["d_gyroX"] = df["gyroX"].diff()
    # df["d_gyroY"] = df["gyroY"].diff()
    # df["d_gyroZ"] = df["gyroZ"].diff()
    # print(df)
    for col in ["gyroZ"]:  # LOG_COLUMNS[1:]:
        df[f"d_{col}"] = df[col].diff()
        print(df[f"d_{col}"].min())
        # [print(z) for z in zip(df.index, df[f"d_{col}"])]
        pyplot.suptitle(col)
        pyplot.plot(df[col], linewidth=1)
        pyplot.plot(df[f"d_{col}"], linewidth=1)
        # pyplot.savefig(f"{col}.png", dpi=300, format="png")
        # pyplot.show()
        pyplot.clf()
