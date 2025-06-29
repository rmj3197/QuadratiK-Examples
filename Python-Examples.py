"""
This file contains Python equivalent of all listings in R-Examples.R.
"""

############################################################################
#                               LISTING 1                                  #
############################################################################

from QuadratiK.kernel_test import KernelTest
import pandas as pd
from QuadratiK.datasets import load_wine_data

X = load_wine_data()
y = X["Class"]
x = X.drop(columns=["Class"])
k_sample_test = KernelTest(h=2.4, method="subsampling", random_state=42).test(x, y)
print(k_sample_test.summary())

############################################################################
#                              END OF LISTING 1                            #
############################################################################

############################################################################
#                               LISTING 2                                  #
############################################################################

from QuadratiK.kernel_test import KernelTest
from QuadratiK.datasets import load_wisconsin_breast_cancer_data

X = load_wisconsin_breast_cancer_data()

X1 = X[X["Class"] == 0].drop(columns=["Class"]).values
X2 = X[X["Class"] == 1].drop(columns=["Class"]).values
two_sample_test1 = KernelTest(h=2.4, method="subsampling", random_state=42).test(X1, X2)
print(two_sample_test1.summary())

############################################################################
#                              END OF LISTING 2                            #
############################################################################

############################################################################
#                               LISTING 3                                  #
############################################################################

from QuadratiK.kernel_test import KernelTest
from QuadratiK.kernel_test import select_h

# Load and preprocess data
df = pd.read_csv("Datasets/open_exoplanet_catalogue.txt")
df = df[
    ["hoststar_mass", "hoststar_radius", "hoststar_metallicity", "binaryflag"]
].dropna()

# Prepare data for two-sample test
X_2 = df[df["binaryflag"] == 0][
    ["hoststar_mass", "hoststar_radius", "hoststar_metallicity"]
].values
Y_2 = df[df["binaryflag"] == 2][
    ["hoststar_mass", "hoststar_radius", "hoststar_metallicity"]
].values
two_sample_test2 = KernelTest(h=0.4, method="subsampling", random_state=42).test(
    X_2, Y_2
)
print(two_sample_test2.summary())

############################################################################
#                              END OF LISTING 3                            #
############################################################################

############################################################################
#                               LISTING 4                                  #
############################################################################

# This code was executed on HPC cluster. Please ensure you have the necessary
# configuration to run it on your local machine or cluster. If you are running this
# on a local machine, you may need to adjust the number of jobs and memory settings.
# The data file 'HIGGS.csv.gz' should be downloaded UCI Machine Learning Repository.
# The dataset is large, hence it is not included in the repository.
"""
import pandas as pd
import time
import tracemalloc

# Load data
df = pd.read_csv('HIGGS.csv.gz', compression='gzip', header=None)
X = df[df[0] == 0][:20000].values[:,1:22]
Y = df[df[0] == 1][:20000].values[:,1:22]

# Import your kernel test
from QuadratiK.kernel_test import KernelTest

# Start timing and memory tracing
start_time = time.time()
tracemalloc.start()

# Run the two sample test
two_sample_test = KernelTest(h=1.5, num_iter=150, n_jobs=20, random_state=42).test(X, Y)

# End timing and memory tracking
current, peak = tracemalloc.get_traced_memory()
end_time = time.time()

# Print results
print(two_sample_test.summary())
print(f"\nExecution time: {end_time - start_time:.2f} seconds")
print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage: {peak / 10**6:.2f} MB")

# Stop the trace
tracemalloc.stop()
"""

############################################################################
#                              END OF LISTING 4                            #
############################################################################

############################################################################
#                               LISTING 5                                  #
############################################################################

from QuadratiK.kernel_test import KernelTest
import pandas as pd
from QuadratiK.datasets import load_wine_data

X = load_wine_data()
y = X["Class"].values
x = X.drop(columns=["Class"]).values

h_selected, all_powers, plot = select_h(
    x=x,
    y=y,
    alternative="location",
    power_plot=True,
    method="subsampling",
    b=0.9,
    delta=[1, 2, 3],
    random_state=100,
)
print(f"Selected h is: {h_selected}")

############################################################################
#                              END OF LISTING 5                            #
############################################################################

############################################################################
#                               LISTING 6                                  #
############################################################################

from QuadratiK.kernel_test import KernelTest
from QuadratiK.datasets import load_wisconsin_breast_cancer_data

X = load_wisconsin_breast_cancer_data()

X1 = X[X["Class"] == 0].drop(columns=["Class"]).values
X2 = X[X["Class"] == 1].drop(columns=["Class"]).values

h_selected, all_powers, plot = select_h(
    x=X1,
    y=X2,
    alternative="location",
    power_plot=True,
    method="subsampling",
    b=0.9,
    delta=[1, 2, 3],
    random_state=100,
)
print(f"Selected h is: {h_selected}")

############################################################################
#                              END OF LISTING 6                            #
############################################################################

############################################################################
#                               LISTING 7                                  #
############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from QuadratiK.poisson_kernel_test import PoissonKernelTest

# Load earthquake data
usgs_df = pd.read_csv("Datasets/usgs_earthquake_data.csv")

# Convert lat/lon to Cartesian coordinates on unit sphere
lat_rad = np.radians(usgs_df["latitude"])
lon_rad = np.radians(usgs_df["longitude"])
usgs_df["x"] = np.cos(lat_rad) * np.cos(lon_rad)
usgs_df["y"] = np.cos(lat_rad) * np.sin(lon_rad)
usgs_df["z"] = np.sin(lat_rad)

# Create a transparent 3D unit sphere mesh
phi, theta = np.mgrid[0 : np.pi : 100j, 0 : 2 * np.pi : 100j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Plot the 3D Earthquake data and sphere
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(azim=40, elev=50)

# Earthquake points
ax.scatter(
    usgs_df["x"],
    usgs_df["y"],
    usgs_df["z"],
    color="red",
    s=15,
    marker="*",
    zorder=5,
    depthshade=True,
)

# Unit sphere
ax.plot_surface(
    x,
    y,
    z,
    color="gray",
    alpha=0.1,
    linewidth=0,
    edgecolor="none",
    zorder=0,
    shade=False,
)

# Axis setup
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
ax.set(
    xticks=ticks, yticks=ticks, zticks=ticks, xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1]
)
ax.set_aspect("equal")
ax.tick_params(axis="both", which="major", labelsize=8)

# Save and show
plt.tight_layout()
# plt.savefig("Earthquakes.pdf", bbox_inches='tight')
plt.show()

# Perform uniformity test using Poisson kernel
unif_test = PoissonKernelTest(rho=0.3, random_state=42).test(
    usgs_df[["x", "y", "z"]].values
)

# Print test summary
print(unif_test.summary())

############################################################################
#                              END OF LISTING 7                            #
############################################################################

############################################################################
#                               LISTING 8                                  #
############################################################################

import os
import json
import datetime
import urllib.request

from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------- Constants -----------------------
TLE_DATA_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle"
TLE_FILE_NAME = "satellite_data_oneweb.tle"
GEOJSON_FOLDER = "./geojson_data"
GEOJSON_FILE = f"{GEOJSON_FOLDER}/satellite_data_oneweb.geojson"
NUM_DIVISIONS = 1


# ----------------------- TLE Fetch -----------------------
def fetch_tle_data():
    try:
        data = urllib.request.urlopen(TLE_DATA_URL)
        with open(TLE_FILE_NAME, "w") as file:
            file.write(data.read().decode("utf-8"))
        print("TLE data fetched successfully.")
    except IOError as e:
        print("Error fetching TLE data:", str(e))


# ----------------------- Satellite Initialization -----------------------
def initialize_satellites(file_name):
    satellites = {}
    with open(file_name, "r") as file:
        while True:
            name = file.readline()
            line1 = file.readline()
            line2 = file.readline()
            if not (name and line1 and line2):
                break
            satellite = EarthSatellite(line1, line2, name.strip())
            satellites[satellite.model.satnum] = satellite
    return satellites


# ----------------------- GeoJSON Conversion -----------------------
def get_current_position(satellite, ts):
    subpoint = satellite.at(ts.now()).subpoint()
    return [subpoint.longitude.degrees, subpoint.latitude.degrees]


def satellite_to_geojson_obj(satellite, ts):
    coordinates = get_current_position(satellite, ts)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": coordinates},
        "properties": {
            "name": satellite.name,
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "id": satellite.model.satnum,
        },
    }


def satellites_to_geojson(satellites, out_file, ts):
    features = [satellite_to_geojson_obj(sat, ts) for sat in satellites]
    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_file, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"GeoJSON written to {out_file}")


# ----------------------- Division Logic -----------------------
def divide_satellites(satellites, n):
    items = list(satellites.items())
    chunk_size = len(items) // n
    remainder = len(items) % n
    divisions = {}
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        divisions[i + 1] = dict(items[start:end])
        start = end
    return divisions


# ----------------------- Visualization -----------------------
def visualize_geojson(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    gdf = gpd.GeoDataFrame.from_features(data["features"])
    gdf["longitude"] = gdf.geometry.x
    gdf["latitude"] = gdf.geometry.y

    # Convert to Cartesian coordinates
    lat_rad = np.radians(gdf["latitude"])
    lon_rad = np.radians(gdf["longitude"])
    gdf["x"] = np.cos(lat_rad) * np.cos(lon_rad)
    gdf["y"] = np.cos(lat_rad) * np.sin(lon_rad)
    gdf["z"] = np.sin(lat_rad)

    # Sphere for Earth
    phi, theta = np.mgrid[0 : np.pi : 100j, 0 : 2 * np.pi : 100j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        gdf["x"],
        gdf["y"],
        gdf["z"],
        color="r",
        s=25,
        marker=".",
        label="Satellites",
        zorder=10,
    )
    ax.plot_surface(
        x,
        y,
        z,
        color="gray",
        alpha=0.1,
        linewidth=0,
        edgecolor="none",
        shade=False,
        zorder=1,
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.view_init(elev=40, azim=50)

    plt.savefig("Satellites.pdf", bbox_inches="tight", dpi=300)
    plt.show()


# ----------------------- Execution -----------------------

# Create folder for output
os.makedirs(GEOJSON_FOLDER, exist_ok=True)

# Load and process satellite data
fetch_tle_data()
ts = load.timescale()
satellites = initialize_satellites(TLE_FILE_NAME)
division_map = divide_satellites(satellites, NUM_DIVISIONS)

# Generate GeoJSON file
for i in range(1, NUM_DIVISIONS + 1):
    satellites_to_geojson(list(division_map[i].values()), GEOJSON_FILE, ts)

# Visualize satellite distribution
visualize_geojson(GEOJSON_FILE)

# Perfrorm uniformity test using Poisson kernel
import numpy as np
from QuadratiK.poisson_kernel_test import PoissonKernelTest

data = np.genfromtxt(
    "Datasets/satellite_coordinates.csv", delimiter=",", skip_header=True
)

unif_test = PoissonKernelTest(rho=0.4, random_state=42).test(data)

# Print test summary
print(unif_test.summary())

############################################################################
#                              END OF LISTING 8                            #
############################################################################

############################################################################
#                           LISTING 9,10,11,12                             #
############################################################################

from QuadratiK.datasets import load_wireless_data
from QuadratiK.spherical_clustering import PKBC
import pandas as pd

# Load the wireless dataset
wireless_data = load_wireless_data()
print(wireless_data.head())

# Load data and labels
X, y = load_wireless_data(return_X_y=True)

# Apply PKBC clustering
pkbc = PKBC(num_clust=range(2, 11), random_state=42).fit(X)

# Validate clustering results
validation_metrics, elbow_plots = pkbc.validation(y_true=y)
print(validation_metrics.round(2))
elbow_plots.show()

# Show cluster statistics for 4 clusters
print(pkbc.stats_clusters(num_clust=4))

############################################################################
#                          END OF LISTING 9,10,11,12                       #
############################################################################
