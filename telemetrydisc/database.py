import binascii
import datetime
from functools import lru_cache
import os
import pandas as pd
import sqlalchemy
import sqlalchemy.engine.url
import sys
from telemetrydisc.util import LOCAL_DATA_PATH, LOCAL_DB_PATH, LOG_COLUMNS


@lru_cache()
def get_engine(db: str):
    if db == "local":
        return sqlalchemy.create_engine(f"sqlite:///{LOCAL_DB_PATH }")
    elif db == "remote":
        server = 'telemetry-disc-data.postgres.database.azure.com'
        username = 'tsmanner@telemetry-disc-data'
        password = 'aTH3n588a'
        url = sqlalchemy.engine.url.URL("postgresql", username, password, server, 5432, "postgres")
        return sqlalchemy.create_engine(url, connect_args={"sslmode": "require"})
    raise ValueError(f"unrecognized database '{db}'")


def get_logs_table():
    return pd.read_sql(
        "SELECT * FROM logs",
        get_engine("local").connect(),
        index_col="log_crc",
        parse_dates=["log_date"]
    )


def get_raw_data(log_crc: int):
    return pd.read_sql(
        f"SELECT * FROM raw_data WHERE log_crc={log_crc}",
        get_engine("local").connect(),
        index_col="timeMS"
        # index_col=["timeMS", "log_crc"]
    )


def init_db():
    # If the local database doesn't exist yet, create it
    if not os.path.exists(LOCAL_DB_PATH):
        if not os.path.exists(LOCAL_DATA_PATH):
            print(f"Creating local data directory '{LOCAL_DATA_PATH}'.")
            os.mkdir(LOCAL_DATA_PATH)
        print(f"Creating local database '{LOCAL_DB_PATH}'.")
        open(LOCAL_DB_PATH, "x").close()
    remote_engine = get_engine("remote")
    remote_con = remote_engine.connect()
    local_engine = get_engine("local")
    local_con = local_engine.connect()
    for con in [remote_con, local_con]:
        con.execute("CREATE TABLE IF NOT EXISTS logs(log_crc INTEGER PRIMARY KEY, log_name TEXT, log_date TIMESTAMP)")
        con.execute(
            'CREATE TABLE IF NOT EXISTS raw_data('
            'log_crc BIGINT, '
            '"timeMS" BIGINT, '
            '"accelX" FLOAT, '
            '"accelY" FLOAT, '
            '"accelZ" FLOAT, '
            '"gyroX" FLOAT, '
            '"gyroY" FLOAT, '
            '"gyroZ" FLOAT, '
            '"magX" FLOAT, '
            '"magY" FLOAT, '
            '"magZ" FLOAT, '
            'PRIMARY KEY("log_crc", "timeMS")'
            ')'
        )


def reset_db():
    engine = get_engine("remote")
    con = engine.connect()
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)
    [con.execute(f"DROP TABLE {table}") for table in meta.tables]


def calculate_crc(filename):
    data = pd.read_csv(filename, names=LOG_COLUMNS, index_col="timeMS")
    crc_data = pd.DataFrame(data.index, index=data.index)
    for column in LOG_COLUMNS[1:]:
        crc_data[column] = data[column]
    crc = 0
    for i in data.index:
        crc = binascii.crc32(str(list(data.loc[i])).encode(), crc)
    return crc


def load_new_log(filename: str):
    local_engine = get_engine("local")
    local_con = local_engine.connect()
    log_crc = calculate_crc(filename)
    crc_table = pd.read_sql("SELECT * FROM logs", local_con, index_col="log_crc", parse_dates=["log_date"])
    if log_crc not in crc_table.index:
        print(f"Processing '{filename}({log_crc})'...", end="")
        crc_table.loc[log_crc] = pd.Series(
            {
                "log_name": os.path.basename(filename),
                "log_date": datetime.datetime.now()
            },
        )
        crc_table.to_sql("logs", local_con, if_exists="replace")
        data = pd.read_csv(filename, names=LOG_COLUMNS)
        data["log_crc"] = log_crc
        data.set_index(["log_crc", "timeMS"], inplace=True)
        sys.stdout.flush()
        data.to_sql("raw_data", local_con, if_exists="append")
        print("done.")


def sync_log_data():
    remote_engine = get_engine("remote")
    remote_con = remote_engine.connect()
    local_engine = get_engine("local")
    local_con = local_engine.connect()

    meta = sqlalchemy.MetaData()
    meta.reflect(bind=remote_engine)
    remote_crc_table = pd.read_sql("SELECT * FROM logs", remote_con, index_col="log_crc")
    local_crc_table = pd.read_sql("SELECT * FROM logs", local_con, index_col="log_crc")
    # print(f"Tables: {sorted(meta.tables)}")
    # print(pd.read_sql("SELECT * FROM logs", local_con, index_col="log_crc"))
    # Download missing log data
    for crc in set(remote_crc_table.index) - set(local_crc_table.index):
        print(f"Downloading log {remote_crc_table.loc[crc]['log_name']}({crc})...", end="")
        sys.stdout.flush()
        remote_data = pd.read_sql(
            f"SELECT * FROM raw_data WHERE log_crc={crc}",
            remote_con,
            index_col=["timeMS", "log_crc"]
        )
        remote_data.to_sql("raw_data", local_con, if_exists="append")
        local_crc_table.loc[crc] = remote_crc_table.loc[crc]
        print("done.")
    # Upload local log data
    for crc in set(local_crc_table.index) - set(remote_crc_table.index):
        print(f"Uploading log {local_crc_table.loc[crc]['log_name']}({crc})...", end="")
        sys.stdout.flush()
        local_data = pd.read_sql(
            f"SELECT * FROM raw_data WHERE log_crc={crc}",
            local_con,
            index_col=["timeMS", "log_crc"]
        )
        local_data.to_sql("raw_data", remote_con, if_exists="append")
        remote_crc_table.loc[crc] = local_crc_table.loc[crc]
        print("done.")
    remote_crc_table.to_sql("logs", remote_con, if_exists="replace")
    local_crc_table.to_sql("logs", local_con, if_exists="replace")
