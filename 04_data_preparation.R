library(abind)
library(imputeTS)

nspec  <- 176
MV     <- F
MVFlag <- if(MV) "MVT" else "MVF"

# load raw Sentinel-2 reflectance and Sentinel-1 backscatter and interferometric coherence data extracted from CREODIAS cloud; also includes extracted deadwood cover values per pixel based on UAV-based CNN predictions
RDataFile <- paste0("01_data/timeseries_orig_S1_S2_", nspec, "_104.RData")
load(RDataFile)

S2    <- reflectance[,,-13]/65535
kNDVI <- reflectance[,,13]
NDWI  <- (reflectance[,,9]-reflectance[,,11]) / (reflectance[,,9]+reflectance[,,11])
Y     <- target$dead
S     <- data.frame(target)

# remove unplausible coregistrations
dates      <- read.csv2("01_data/UAV_acquisition_dates.csv")
identifier <- paste(dates$plot, dates$year, sep = "_")


par(mfrow = c(2,2), mar = c(3,3,2,1))
boxplot(Y~substr(S$plot, 1,3), width = table(substr(S$plot, 1,3)), xlab = "site", ylab = "deadwood cover")

## remove too short timeseries
Xvalid <- rowSums(!is.na(S2[,,1])*1)
nObs   <- 25
hist(Xvalid, breaks = 30); abline(v = nObs)
idx <- which(Xvalid >= nObs)

S2          <- S2[idx,,]
kNDVI       <- kNDVI[idx,]
NDWI        <- NDWI[idx,]
backscatter <- backscatter[idx,,]
coherence   <- coherence[idx,,]
Y           <- Y[idx]
S           <- S[idx,]


## stratification
h <- hist(Y, breaks = 100, plot = F)
set.seed(7)
idx <- c()
for(v in (length(h$breaks[1:11])-1):1) {
  s   <- which(Y >= h$breaks[v] & Y <= h$breaks[v+1])
  s   <- sample(which(Y >= h$breaks[v] & Y <= h$breaks[v+1]), size = 25, replace = F)
  idx <- c(s, idx)
}

h <- hist(Y, breaks = 10, plot = F)
# h$counts
# min(h$counts)
for(v in (length(h$breaks)-1):2) {
  s   <- which(Y >= h$breaks[v] & Y <= h$breaks[v+1])
  s   <- sample(which(Y >= h$breaks[v] & Y <= h$breaks[v+1]), size = 248, replace = F)
  idx <- c(s, idx)
}

# boxplot(Y[idx])
S2          <- S2[idx,,]
kNDVI       <- kNDVI[idx,]
NDWI        <- NDWI[idx,]
backscatter <- backscatter[idx,,]
coherence   <- coherence[idx,,]
Y           <- Y[idx]
S           <- S[idx,]

boxplot(Y~substr(S$plot,1,3), width = table(substr(S$plot, 1,3)), xlab = "site", ylab = "deadwood cover")
boxplot(Y)

par(mfrow = c(1,1))
hist(Y, breaks = 100)

## combine data
X <- abind(S2, kNDVI, NDWI, backscatter, coherence, along = 3)


## interpolation
ii <- sample(1:nrow(S), 1)
par(mfrow = c(1,1), mar = c(5,4,4,2)+0.1)
plot(X[ii,,13], type = "p", ylim = c(0,1), cex = 1.5, lwd = 2,
     ylab = "kNDVI", xlab = "7-day-interval", cex.axis = 1.5, cex.lab = 1.4)
for(i in 1:dim(X)[3]) {
  X[,,i] <- t(apply(X[,,i], 1, na_interpolation, option = "linear"))
}
points(X[ii,,13], type = "p", ylim = c(0,1), pch = 16, cex = .9, col = "darkgrey")


## save data
save(X, Y, S, file = paste0("01_data/data_orig_", nspec, "_104.RData"))



# additional data ---------------------------------------------------------

load("01_data/timeseries_S1_S2_add_104.RData")
add_S2          <- reflectance[,,-13]/65535
add_kNDVI       <- reflectance[,,13]
add_NDWI        <- (reflectance[,,9]-reflectance[,,11]) / (reflectance[,,9]+reflectance[,,11])
add_backscatter <- backscatter
add_coherence   <- coherence
add_Y           <- target$dead
add_S           <- data.frame(target)

## remove too short timeseries
Xvalid <- rowSums(!is.na(add_S2[,,1])*1)
nObs <- 25
hist(Xvalid, breaks = 30); abline(v = nObs)
idx <- which(Xvalid >= nObs)

add_S2          <- add_S2[idx,,]
add_kNDVI       <- add_kNDVI[idx,]
add_NDWI        <- add_NDWI[idx,]
add_backscatter <- add_backscatter[idx,,]
add_coherence   <- add_coherence[idx,,]
add_Y           <- add_Y[idx]
add_S           <- add_S[idx,]

## combine data
add_X <- abind(add_S2, add_kNDVI, add_NDWI, add_backscatter, add_coherence, along = 3)


## interpolation
ii <- sample(1:nrow(add_S), 1)
par(mfrow = c(1,1), mar = c(5,4,4,2)+0.1)
plot(add_X[ii,,13], type = "p", ylim = c(0,1), cex = 1.5, lwd = 2,
     ylab = "kNDVI", xlab = "7-day-interval", cex.axis = 1.5, cex.lab = 1.4)
for(i in 1:dim(add_X)[3]) {
  add_X[,,i] <- t(apply(add_X[,,i], 1, na_interpolation, option = "linear"))
}
points(add_X[ii,,13], type = "p", ylim = c(0,1), pch = 16, cex = .9, col = "darkgrey")


## save data
save(add_X, add_Y, add_S, file = paste0("01_data/data_add_104.RData"))
