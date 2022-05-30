import numpy as np
import pandas as pd
import torch
import yaml

input_label = [
    "LAT",
    "LONG",
    "TRACK_ALTITUDE",
    "UNIX_TIME",
]

WIND_SPEED = "WIND_SPEED"
WIND_ANGLE = "WIND_ORIGIN_ANGLE_DEG"
FLIGHT_ID = "FLIGHT_ID"
UNIX_TIME = "UNIX_TIME"
DAY = "DAY"
VX = "VX"
VY = "VY"
speed_label = [VX, VY]
label = input_label + speed_label


def select_day(df, day):
    df = df[df["DAY"] == day].copy()
    df.loc[:, "DAY"] = 0
    return df


def process_df(df):
    df = filter_invalid_flights(df)
    df[UNIX_TIME] = add_weektime(df[UNIX_TIME], df[DAY])
    df[VX], df[VY] = polar2carthesian(df[WIND_SPEED], df[WIND_ANGLE])
    df = df.sort_values(by="UNIX_TIME")
    print("Reset index.")
    df = df.reset_index(drop=True)

    fid2indices = create_fid2indices(df)

    return torch.tensor(df[label].values).float(), fid2indices


def create_fid2indices(df):
    return {
        str(fid): torch.tensor((df[df["FLIGHT_ID"] == fid]).index)
        for fid in df["FLIGHT_ID"].unique()
    }


def filter_invalid_flights(df):
    SPEED_OUTLIER_LIMIT = 250
    invalid = df[df[WIND_SPEED] > SPEED_OUTLIER_LIMIT][FLIGHT_ID].unique()
    df = df.dropna(axis=0)
    df = df[~(df[FLIGHT_ID].isin(invalid))]
    return df


def add_weektime(time, day):
    duration_day = 60 * 60 * 24
    time = time + day * duration_day
    return time


def polar2carthesian(r, angle_starting_north_clockwise):
    x = r * np.cos((90 - angle_starting_north_clockwise) / 360 * 2 * np.pi)
    y = r * np.sin((90 - angle_starting_north_clockwise) / 360 * 2 * np.pi)

    return x, y


def save_count(data_list, filename):
    name_and_count = [(str(i), len(d)) for i, d in enumerate(data_list)]
    pd.DataFrame(name_and_count, columns=["name", "count"]).to_csv(filename)


def save_means_stds(means, std, filename):
    columns = [la + "_mean" for la in label] + [la + "_std" for la in label]
    means_and_std = torch.cat([means, std]).squeeze().tolist()
    d = dict(zip(columns, means_and_std))
    with open(filename, "w+") as file:
        yaml.dump(d, file)


def load_means_stds(filename):
    with filename.open("r") as file:
        d = yaml.load(file, Loader=yaml.FullLoader)

    means = torch.tensor([d[L + "_mean"] for L in label])
    stds = torch.tensor([d[L + "_std"] for L in label])

    return means, stds

