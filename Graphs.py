# =====================================================================
# Lift experiment - graphs
#
# A phone (Physics Toolbox app) rode a lift between floors 8 times.
# It recorded acceleration (about 100 times a second) and air pressure
# (about once a second). Floors are 3.2 m apart, so we know the true
# height of every ride.
#
# Question: can the phone measure how far the lift travelled?
# We work out the height in two ways and compare with the true height:
#   1. from pressure:      height = -(pressure change) / (rho * g)
#   2. from acceleration:  integrate to get velocity, integrate again
#                          to get height
#
# To run: save this file in the same folder as the CSV files and press F5.
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress

# ------------------------------------------------------------ settings
RHO = 1.20           # density of air (kg/m^3)
G = 9.81             # gravity (m/s^2)
FLOOR_HEIGHT = 3.2   # height of one floor (m)

# graph style: ticks pointing inwards, grid on
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#e5e5e5"

# Each ride: [file name, start floor, end floor, label for graphs]
# Note: two file names are wrong. "6_to_0_trial_01" is really a ride UP
# from 0 to 6, and "0_to_6_trial_03" is really a ride DOWN from 6 to 0.
# (You can see this in the pressure data and in ride_summary.csv.)
RIDES = [
    ["0_to_6_trial_01.csv", 0, 6, "Floor 0 → 6 (trial 1)"],
    ["0_to_6_trial_04.csv", 0, 6, "Floor 0 → 6 (trial 2)"],
    ["6_to_0_trial_01.csv", 0, 6, "Floor 0 → 6 (trial 3)"],
    ["0_to_6_trial_03.csv", 6, 0, "Floor 6 → 0 (trial 1)"],
    ["6_to_0_trial_02.csv", 6, 0, "Floor 6 → 0 (trial 2)"],
    ["6_to_3_trial_01.csv", 6, 3, "Floor 6 → 3"],
    ["6_to_4_trial_01.csv", 6, 4, "Floor 6 → 4"],
    ["4_to_0_trial_01.csv", 4, 0, "Floor 4 → 0"],
]


# ------------------------------------------------------------ functions
def load_ride(filename, start_floor, end_floor, label):
    """Read one ride and work out its height in both ways.
    Returns a dictionary with everything we need for the graphs."""

    # Columns in the file: time, ax, ay, az, pressure, pressure change
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    # Cut off the first second (the first pressure reading is often
    # out of date) and the last 0.2 s (someone picking up the phone)
    data = data[100:-20]

    t = data[:, 0]      # time (s)
    az = data[:, 3]     # vertical acceleration (m/s^2)
    p = data[:, 4]      # pressure (Pa)

    # The lift doesn't move for the first 2 s (200 samples), so we use
    # that part as our "starting" values.

    # ---- Method 1: pressure ----
    dp = p - np.mean(p[:200])          # pressure change from the start
    h_pressure = -dp / (RHO * G)       # convert to height

    # ---- Method 2: acceleration ----
    # The sensor reads slightly above 0 even when still, so subtract that
    az = az - np.mean(az[:200])
    v = cumulative_trapezoid(az, t, initial=0)          # velocity
    h_accel = cumulative_trapezoid(v, t, initial=0)     # height

    # Smooth the acceleration for plotting (average over 50 samples = 0.5 s)
    az_smooth = np.convolve(az, np.ones(50) / 50, mode="same")

    # A lift always ends the ride stopped, so the velocity should come
    # back to about 0. If it doesn't, the phone moved during the ride.
    accel_ok = abs(v[-1]) < 0.3
    if not accel_ok:
        print("Warning:", label, "- accelerometer data is unreliable")

    ride = {
        "label": label,
        "t": t,
        "az": az_smooth,
        "v": v,
        "h_accel": h_accel,
        "dp": dp,
        "h_pressure": h_pressure,
        "true_h": (end_floor - start_floor) * FLOOR_HEIGHT,  # + up, - down
        "accel_ok": accel_ok,
    }
    return ride


def graph1_vertical_acceleration(rides):
    """Vertical acceleration for one ride up and one ride down."""
    up = rides[0]
    down = rides[3]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(up["t"], up["az"], color="tab:blue")
    ax[0].set_title(up["label"])
    ax[1].plot(down["t"], down["az"], color="tab:orange")
    ax[1].set_title(down["label"])

    for a in ax:
        a.axhline(0, color="black", linewidth=0.8)
        a.set_ylabel("Acceleration (m/s²)")
    ax[1].set_xlabel("Time (s)")
    fig.suptitle("Vertical acceleration of the lift")
    fig.tight_layout()
    plt.savefig("graph1_vertical_acceleration.png", dpi=200)


def graph2_velocity(rides):
    """Velocity of every ride (from integrating the acceleration)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ride in rides:
        if ride["accel_ok"]:
            ax.plot(ride["t"], ride["v"], label=ride["label"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)   [up = +]")
    ax.set_title("Lift velocity (from integrating acceleration)")
    ax.legend(fontsize=8)
    plt.savefig("graph2_velocity.png", dpi=200)


def graph3_motion_summary(ride):
    """Acceleration, velocity and height for one ride, stacked."""
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(ride["t"], ride["az"])
    ax[0].set_ylabel("Acceleration (m/s²)")

    ax[1].plot(ride["t"], ride["v"])
    ax[1].set_ylabel("Velocity (m/s)")

    ax[2].plot(ride["t"], ride["h_accel"])
    ax[2].axhline(ride["true_h"], color="grey", linestyle="--",
                  label="True height")
    ax[2].set_ylabel("Height (m)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend()

    ax[0].set_title("Motion of the lift - " + ride["label"])
    fig.tight_layout()
    plt.savefig("graph3_motion_summary.png", dpi=200)


def graph4_height_two_methods(ride):
    """Height against time from both methods, for one ride."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ride["t"], ride["h_accel"], label="From accelerometer")
    ax.plot(ride["t"], ride["h_pressure"], label="From pressure")
    ax.axhline(ride["true_h"], color="grey", linestyle="--",
               label="True height")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height change (m)")
    ax.set_title("Two ways of measuring height - " + ride["label"])
    ax.legend()
    plt.savefig("graph4_height_two_methods.png", dpi=200)


def graph5_measured_vs_true(rides):
    """Final measured height against true height, for every ride."""
    true_p, measured_p = [], []     # pressure method
    true_a, measured_a = [], []     # accelerometer method

    for ride in rides:
        true = abs(ride["true_h"])
        true_p.append(true)
        measured_p.append(abs(ride["h_pressure"][-1]))   # [-1] = last value
        if ride["accel_ok"]:
            true_a.append(true)
            measured_a.append(abs(ride["h_accel"][-1]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 21], [0, 21], "--", color="grey", label="Perfect agreement")
    ax.scatter(true_p, measured_p, marker="s", color="tab:orange",
               label="Pressure")
    ax.scatter(true_a, measured_a, marker="o", color="tab:blue",
               label="Accelerometer")
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.set_xlabel("True height (m)")
    ax.set_ylabel("Measured height (m)")
    ax.set_title("Measured vs true height")
    ax.legend()
    plt.savefig("graph5_measured_vs_true.png", dpi=200)


def graph6_air_density(rides):
    """Pressure change against height change, with a straight-line fit.
    Since pressure change = -rho * g * height, the slope = -rho * g,
    so we can work out the density of air from the slope."""
    heights = []
    pressures = []
    for ride in rides:
        heights.append(ride["true_h"])
        pressures.append(ride["dp"][-1])     # total pressure change
    heights = np.array(heights)
    pressures = np.array(pressures)

    fit = linregress(heights, pressures)
    rho = -fit.slope / G
    rho_error = fit.stderr / G

    x = np.array([-21, 21])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(heights, pressures, color="black", zorder=3, label="Data")
    ax.plot(x, fit.slope * x + fit.intercept, color="tab:blue",
            label=f"Fit: ρ = {rho:.2f} ± {rho_error:.2f} kg/m³")
    ax.plot(x, -RHO * G * x, "--", color="grey",
            label=f"Expected: ρ = {RHO} kg/m³")
    ax.set_xlabel("Height change (m)   [up = +]")
    ax.set_ylabel("Pressure change (Pa)")
    ax.set_title("Pressure change vs height")
    ax.legend()
    plt.savefig("graph6_air_density.png", dpi=200)

    print(f"Air density from fit: {rho:.3f} ± {rho_error:.3f} kg/m^3")


def print_results(rides):
    """Print each ride's measured heights and % errors."""
    print("\nRide | true height | accelerometer | pressure")
    for ride in rides:
        true = abs(ride["true_h"])
        h_p = abs(ride["h_pressure"][-1])
        error_p = (h_p - true) / true * 100
        if ride["accel_ok"]:
            h_a = abs(ride["h_accel"][-1])
            error_a = (h_a - true) / true * 100
            accel_text = f"{h_a:.2f} m ({error_a:.1f}%)"
        else:
            accel_text = "bad data"
        print(f"{ride['label']} | {true:.1f} m | {accel_text} | "
              f"{h_p:.2f} m ({error_p:.1f}%)")


# ------------------------------------------------------------ main code
rides = []
for r in RIDES:
    rides.append(load_ride(r[0], r[1], r[2], r[3]))

graph1_vertical_acceleration(rides)
graph2_velocity(rides)
graph3_motion_summary(rides[0])
graph4_height_two_methods(rides[0])
graph5_measured_vs_true(rides)
graph6_air_density(rides)
print_results(rides)

plt.show()

