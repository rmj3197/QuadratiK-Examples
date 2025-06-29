############################################################################
#                               LISTING 1                                  #
############################################################################

library(QuadratiK)

# select numeric variables without the column with the labels
data(wine)
x <- wine[, -14]
# Define the vector with the group labels
y <- factor(wine[, 14])

# Perform the k-sample test
system.time(kbqd_test_wine <- kb.test(x = x, y = y, h = 2.4))
kbqd_test_wine

############################################################################
#                              END OF LISTING 1                            #
############################################################################

############################################################################
#                               LISTING 2                                  #
############################################################################

library(QuadratiK)

data(breast_cancer)
dat <- breast_cancer
x <- dat[which(dat$y == "B"), -ncol(dat)]
y <- dat[which(dat$y == "M"), -ncol(dat)]

# Perform the test
system.time(kbqd_test <- kb.test(x = x, y = y, h = 0.4))
kbqd_test

############################################################################
#                              END OF LISTING 2                            #
############################################################################

############################################################################
#                               LISTING 3                                  #
############################################################################

library(QuadratiK)
df <- read.csv("Datasets/open_exoplanet_catalogue.txt", header = TRUE)
# Select columns and remove rows with missing values
df <- df[, c(
    "hoststar_mass", "hoststar_radius",
    "hoststar_metallicity", "binaryflag"
)]
df <- df[complete.cases(df), ]

# Prepare data for two-sample test
X <- as.matrix(df[df$binaryflag == 0, -which(names(df) == "binaryflag")])
Y <- as.matrix(df[df$binaryflag == 2, -which(names(df) == "binaryflag")])

system.time(kbqd_test <- kb.test(x = X, y = Y, h = 0.4))

kbqd_test

############################################################################
#                              END OF LISTING 3                            #
############################################################################

############################################################################
#                               LISTING 4                                  #
############################################################################

# This code was executed on HPC cluster. Please ensure you have the necessary
# configuration to run it on your local machine or cluster.
# The data file 'HIGGS.csv.gz' should be downloaded UCI Machine Learning Repository.
# The dataset is large, hence it is not included in the repository.

library(QuadratiK)

# Load and preprocess data
df <- read.csv("HIGGS.csv.gz", header = FALSE)

# Select relevant features
df <- df[, 1:22]

# Separate samples based on class label 
X <- df[df$V1 == 0, 2:22]
Y <- df[df$V1 == 1, 2:22]

# Take first 10000 samples from each class
X <- as.matrix(X[1:20000, ])
Y <- as.matrix(Y[1:20000, ])

# Run two-sample test
system.time(kbqd_test <- kb.test(x = X, y = Y, h = 1.5))

# Print the result
kbqd_test

############################################################################
#                              END OF LISTING 4                            #
############################################################################

############################################################################
#                               LISTING 5                                  #
############################################################################

data(wine)
# select the data and the labels
x <- wine[, -14]
y <- factor(wine[, 14])
# Perform the algorithm for the selection of h
set.seed(123)
time_wine <- system.time(h_sel_wine <- select_h(
    x = x, y = y,
    alternative = "location", method = "subsampling",
    b = 0.9, delta = c(1, 2, 3)
))
time_wine
h_sel_wine$h_sel

############################################################################
#                              END OF LISTING 5                            #
############################################################################

############################################################################
#                               LISTING 6                                  #
############################################################################

dat <- breast_cancer
x <- dat[which(dat$y == "B"), -ncol(dat)]
y <- dat[which(dat$y == "M"), -ncol(dat)]

# Perform algorithm for selection of h
time_wisc <- system.time(
    h_sel <- select_h(
        x = x, y = y, alternative = "skewness",
        method = "subsampling", b = 0.5
    )
)
time_wisc
h_sel$h_sel

############################################################################
#                              END OF LISTING 6                            #
############################################################################

############################################################################
#                               LISTING 7                                  #
############################################################################

library(QuadratiK)

usgs_df <- read.csv("Datasets/usgs_earthquake_data.csv")[, c("latitude", "longitude")]
lat_rad <- usgs_df$latitude * pi / 180
lon_rad <- usgs_df$longitude * pi / 180

# Compute unit sphere coordinatess
usgs_df$x <- cos(lat_rad) * cos(lon_rad)
usgs_df$y <- cos(lat_rad) * sin(lon_rad)
usgs_df$z <- sin(lat_rad)

X <- as.matrix(usgs_df[, c("x", "y", "z")])

system.time(unif_test <- pk.test(x = X, rho = 0.3, B = 300, Quantile = 0.95))
unif_test

############################################################################
#                              END OF LISTING 7                            #
############################################################################

############################################################################
#                               LISTING 8                                  #
############################################################################

library(QuadratiK)
satelite_data <- read.csv("Datasets/satellite_coordinates.csv")
# Perform Poisson Kernel-based uniformity test on the sphere
system.time(unif_test <- pk.test(
    x = satelite_data, rho = 0.4,
    B = 300, Quantile = 0.95
))
unif_test

############################################################################
#                              END OF LISTING 8                            #
############################################################################
