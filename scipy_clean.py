import numpy as np
from scipy.signal import butter, filtfilt

INPUT_FOLDER = "data_first_run"
OUTPUT_FOLDER = "data_clean"

TRIAL_FILES = [
    "0_to_6_trial_01.csv",
    "0_to_6_trial_03.csv",
    "0_to_6_trial_04.csv",
    "4_to_0_trial_01.csv",
    "6_to_0_trial_01.csv",
    "6_to_0_trial_02.csv",
    "6_to_3_trial_01.csv",
    "6_to_4_trial_01.csv",
]


FILTER_ORDER = 4
CUTOFF_HZ = 5
EDGE_S = 2
MOVE_THRESHOLD = 0.1


def load_trial(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1) # loading data
    time, ax, ay, az, pressure, pressure_change = data.T
    return time, az, pressure


def filter_acceleration(time, az):
    rate = 1 / np.median(np.diff(time))  # measurements per second
    b, a = butter(FILTER_ORDER, CUTOFF_HZ, fs=rate) # creating weights for filter
    return filtfilt(b, a, az) # filtered values


def flag_pressure_updates(pressure):
    pressure_updates = np.zeros(len(pressure), dtype=bool) #selects unrepeated values of pressure
    pressure_updates[1:] = np.diff(pressure) != 0
    return pressure_updates


def save_trial(path, time, az_f, pressure, pressure_updates):
    table = np.column_stack([time, az_f, pressure, pressure_updates]) #creates table with updated data
    header = "elapsed_s,az_filt_m_s2,pressure_Pa,pressure_updated"
    formats = ["%.3f", "%.6f", "%.6f", "%d"]
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt=formats)

def main():
    for name in TRIAL_FILES:
        time, az, pressure = load_trial(INPUT_FOLDER + "/" + name)
        az_f = filter_acceleration(time, az)
        pressure_updates = flag_pressure_updates(pressure)
        save_trial(OUTPUT_FOLDER + "/" + name, time, az_f, pressure, pressure_updates)
        print("saved", name)

main()