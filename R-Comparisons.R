##############################################################################
#       COMPARISON OF MMD AND ENERGY STATISTICS ON EXOPLANET DATASET        #
##############################################################################
library(energy)
library(kernlab)
library(QuadratiK)

# Load and prepare Exoplanet dataset
df <- read.csv("Datasets/open_exoplanet_catalogue.txt", header = TRUE)
df <- df[, c("hoststar_mass", "hoststar_radius", "hoststar_metallicity", "binaryflag")]
df <- df[complete.cases(df), ]

x <- as.matrix(df[df$binaryflag == 0, -which(names(df) == "binaryflag")])  
y <- as.matrix(df[df$binaryflag == 2, -which(names(df) == "binaryflag")])  

cat("\n================== EXOPLANET DATASET ==================\n")

# MMD Test on Exoplanet Data
cat("\n--- Maximum Mean Discrepancy (MMD) Test ---\n")
mmd_time <- system.time({
  mmdtest <- kmmd(x = x, y = y, kernel = "rbfdot", kpar = "automatic", alpha = 0.05, asymptotic = TRUE, B = 150)
})
print(mmd_time)
print(mmdtest)

cat("\n##############################################################################\n")
cat("#                      END OF EXOPLANET MMD TEST                             #\n")
cat("##############################################################################\n")

# Energy Test on Exoplanet Data
cat("\n--- Energy Distance Test ---\n")
energy_time <- system.time({
  energy_test <- eqdist.etest(x = rbind(x, y), sizes = c(dim(x)[1], dim(y)[1]), R = 150)
})
print(energy_time)
print(energy_test)

cat("\n##############################################################################\n")
cat("#                    END OF EXOPLANET ENERGY TEST                            #\n")
cat("##############################################################################\n")


##############################################################################
#                        UNIFORMITY TESTS ON EARTHQUAKE                      #
##############################################################################

library(sphunif)
library(QuadratiK)

cat("\n================== USGS EARTHQUAKE DATA ==================\n")

# Read and preprocess USGS Earthquake data
usgs_df <- read.csv("Datasets/usgs_earthquake_data.csv")[, c("latitude", "longitude")]

# Convert latitude and longitude to radians
lat_rad <- usgs_df$latitude * pi / 180
lon_rad <- usgs_df$longitude * pi / 180

# Compute 3D coordinates on the unit sphere
usgs_df$x <- cos(lat_rad) * cos(lon_rad)
usgs_df$y <- cos(lat_rad) * sin(lon_rad)
usgs_df$z <- sin(lat_rad)

# Verify unit norm
usgs_df$norm <- sqrt(usgs_df$x^2 + usgs_df$y^2 + usgs_df$z^2)
if (all(abs(usgs_df$norm - 1) < 1e-6)) {
  cat("All vectors are on the unit sphere.\n")
} else {
  warning("Some vectors are not unit length!")
}

# Create matrix of 3D coordinates
X <- as.matrix(usgs_df[, c("x", "y", "z")])

# Perform uniformity tests (Ajne and Rayleigh)
cat("\n--- Uniformity Tests (Ajne and Rayleigh) on USGS Data ---\n")
usgs_unif_test <- unif_test(data = X, type = c("Ajne", "Rayleigh"))
print(usgs_unif_test)

cat("\n##############################################################################\n")
cat("#                  END OF UNIFORMITY TEST ON USGS EARTHQUAKE DATA            #\n")
cat("##############################################################################\n\n")


##############################################################################
#             UNIFORMITY TESTS ON ONEWEB SATELITE DATA                       #
##############################################################################

cat("\n================== ONEWEB SATELLITE COORDINATES ==================\n")

# Read satellite coordinate data
satellite_data <- as.matrix(read.csv("Datasets/satellite_coordinates.csv"))

# Optional: check norm of satellite vectors (should also be unit norm)
sat_norms <- sqrt(rowSums(satellite_data^2))
if (all(abs(sat_norms - 1) < 1e-6)) {
  cat("All satellite vectors are on the unit sphere.\n")
} else {
  warning("Some satellite vectors are not unit length!")
}

# Perform uniformity tests (Ajne and Rayleigh)
cat("\n--- Uniformity Tests (Ajne and Rayleigh) on Satellite Data ---\n")
sat_unif_test <- unif_test(data = satellite_data, type = c("Ajne", "Rayleigh"))
print(sat_unif_test)

cat("\n##############################################################################\n")
cat("#               END OF UNIFORMITY TEST ON ONEWEB SATELLITE DATA              #\n")
cat("##############################################################################\n")



