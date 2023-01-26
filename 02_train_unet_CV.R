

# libraries + path -----------------------------------------------------------

pkgs <- c("keras", "tidyverse", "tibble", "tensorflow", "caret", "magick")
sapply(pkgs, require, character.only = TRUE)

set.seed(28)

# when runing multi gpu model
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)
tf$config$experimental$set_memory_growth(device = gpu2, enable = TRUE)
strategy <- tf$distribute$MirroredStrategy()
strategy$num_replicas_in_sync

# mixed precision training speeds up model training
# https://blogs.rstudio.com/ai/posts/2020-01-13-mixed-precision-training/
mixedPrecision <- tf$keras$mixed_precision$experimental
policy <- mixedPrecision$Policy('mixed_float16')
mixedPrecision$set_policy(policy)
policy$compute_dtype  # datatype of tensors
policy$variable_dtype # datatype of weights


source("00_helper_functions.R")
source("00_tfdataset_pipeline.R")
source("00_Unet.R")


# Parameters --------------------------------------------------------------

tilesize   <- 256L
useDSM     <- F
noEpochs   <- 60
repeatData <- 2
if(useDSM) noBands <- 4 else noBands <- 3 # if ortho is combined with DSM = 4 (RGB + DSM) else DSM = 3 (RGB)


# sites <- list.files("01_data/UAV/", pattern = "ortho_", recursive = T)
sites <- list.files("/media/sysgen/Volume/Felix/UAVforSAT/upscale/var/", pattern = "ortho", recursive = T)
nSites <- length(sites)


set.seed(2)
folds <- list()
for (s in c("HAI", "KAB", "FIN", "DDH", "NBF", "CFB")) {
  site_idx <- which(substr(sites, 6,8) == s)
  fold     <- createFolds(site_idx, k=5)
  
  for(f in 1:5) {
    folds[[paste0("Fold",f)]] <- c(folds[[paste0("Fold",f)]], site_idx[fold[paste0("Fold",f)][[1]]])
  }
}


siteTag <- paste0(nSites, "s")
tileTag <- paste0("t", 1024, "var")
dimTag  <- paste0("d", tilesize)
epoTag  <- paste0(noEpochs, "epo")

# outDir = "34sCFB_512_2cm_tfdataset_DSM_100epo/"
outDir <- paste(siteTag, tileTag, dimTag, epoTag, sep = "_")
dir.create(paste0("02_pipeline/runs/CNN/", outDir), recursive = TRUE)


for(k in 1:5) {
  
  # clear GPU
  tf$keras.backend$clear_session()
  py_gc <- import('gc')
  py_gc$collect()
  
  tf$compat$v1$set_random_seed(as.integer(28))
  
  # Load Data ---------------------------------------------------------------
  
  ## list all data
  pathPattern <- paste(tileTag, "b3", sep = "_")
  pathImg     <- list.files("02_pipeline/img/", full.names = T, pattern = pathPattern, recursive = T)
  pathMsk     <- list.files("02_pipeline/msk/", full.names = T, pattern = pathPattern, recursive = T)
  
  testSites <- sites[folds[[paste0("Fold",k)]]]
  
  # Data split --------------------------------------------------------------
  
  s  <- strsplit(pathImg, "/")
  ss <- paste(lapply(s, "[", length(s[[1]])-3), lapply(s, "[", length(s[[1]])-2), sep = "/")
  
  idxTes  <- which(ss %in% substr(testSites,1,11))
  testImg <- pathImg[idxTes]
  testMsk <- pathMsk[idxTes]
  pathImg <- pathImg[-idxTes]
  pathMsk <- pathMsk[-idxTes]
  
  idxVal <- sample(length(pathImg), size = floor(length(pathImg)/5), replace = F)
  idxTra <- (1:length(pathImg))[-idxVal]
  
  trainImg <- pathImg[idxTra]
  trainMsk <- pathMsk[idxTra]
  validImg <- pathImg[idxVal]
  validMsk <- pathMsk[idxVal]
  
  ## create tibbles
  testData  <- tibble(img = testImg, msk = testMsk)
  trainData <- tibble(img = trainImg, msk = trainMsk)
  validData <- tibble(img = validImg, msk = validMsk)
  dir.create(paste0("02_pipeline/runs/CNN/", outDir, "/Fold", k, "/"), recursive = T, showWarnings = F)
  save(testData, trainData, validData, file = paste0("02_pipeline/runs/CNN/", outDir, "/Fold", k, "/data.RData"))
  
  ## calculate dataset size
  datasetSize <- length(trainData$img) * repeatData
  
  
  # Create tfdatasets --------------------------------------------------------------
  
  ## define batch size
  batchSize <- 32
  
  trainingDataset   <- createDataset(trainData, train = T, batch = batchSize, epochs = noEpochs,
                                     datasetSize = datasetSize, useDSM = useDSM, tileSize = tilesize)
  validationDataset <- createDataset(validData, train = F, batch = batchSize, epochs = noEpochs,
                                     useDSM = useDSM, tileSize = tilesize)
  
  # dataIter <- reticulate::as_iterator(trainingDataset)
  # example <- reticulate::iter_next(dataIter)
  # par(mfrow = c(1,2), oma = c(0,0,0,0), mar = c(.5,.5,.5,.5))
  # plot(as.raster(as.array(example[[1]][1,,,1:3]), max = 1))
  # plot(as.raster(as.array(example[[2]][1,,,1]), max = 1))
  
  # multiple gpu (custom loss/metric not supported)
  with(strategy$scope(), {
    model <- getUnet()
  })
  model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate = 1e-4),
    loss = "binary_crossentropy",
    metrics = list("Precision", "Recall")
  )
  
  # # single gpu
  # model <- getUnet()
  # model %>% compile(
  #   optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  #   loss = bce_dice_loss,
  #   metrics = list("Precision", "Recall", dice_coef)
  # )
  
  # Train U-net -------------------------------------------------------------
  
  checkpointDir <- paste0("02_pipeline/runs/CNN/", outDir, "/Fold", k, "/checkpoints/")
  # unlink(checkpointDir, recursive = TRUE)
  dir.create(checkpointDir, recursive = TRUE)
  filepath <- file.path(checkpointDir, "weights.{epoch:02d}-{val_loss:.5f}.hdf5")
  
  ckpt_callback <- callback_model_checkpoint(filepath = filepath,
                                             monitor = "val_loss",
                                             save_weights_only = F,
                                             save_best_only = TRUE,
                                             verbose = 1,
                                             mode = "auto",
                                             save_freq = "epoch")
  csv_callback  <- callback_csv_logger(filename = paste0("02_pipeline/runs/CNN/", outDir, "/Fold", k, "/epoch_results.csv"))
  
  history <- model %>% fit(x = trainingDataset,
                           epochs = noEpochs,
                           steps_per_epoch = datasetSize/(batchSize),
                           callbacks = list(ckpt_callback,
                                            csv_callback,
                                            callback_terminate_on_naan()),
                           validation_data = validationDataset)
  
  # clear GPU
  tf$keras.backend$clear_session()
  py_gc <- import('gc')
  py_gc$collect()
  
  files <- list.files(checkpointDir, recursive = T, full.names = T)
  unlink(files[1:length(files)-1], recursive = T)
  
}
  

# prediction loop ---------------------------------------------------------


for(k in 1:5) {
  
  load(paste0("02_pipeline/runs/CNN/", outDir, "/Fold", k, "/data.RData"))
  testDataset <- createDataset(testData, train = FALSE, batch = 1, shuffle = FALSE, tileSize = tilesize)
  
  checkpointDir <- paste0("02_pipeline/runs/CNN/", outDir, "/Fold", k, "/checkpoints/")
  with(strategy$scope(), {
    model <- loadModel(checkpointDir, compile = TRUE)
  })
  preds <- predict(model, testDataset)
  rm(model)
  
  # clear GPU
  tf$keras.backend$clear_session()
  py_gc <- import('gc')
  py_gc$collect()
  
  
  library(doParallel)
  library(foreach)
  cl <- makeCluster(19)
  registerDoParallel(cl)
  XY <- foreach(s = 1:nrow(testData), .packages = c("raster", "rgdal", "keras", "magick"), .inorder = T) %dopar% {

    pred <- round(preds[s,,,])/255
    pred <- image_read(as.raster(pred))
    
    mskDir <- testData$msk[s]
    year   <- strsplit(mskDir, "/")[[1]][9]
    plot   <- strsplit(mskDir, "/")[[1]][10]
    mem    <- strsplit(strsplit(mskDir, "_")[[1]][3], "/")[[1]][7]
    cell   <- substr(mem, 4, nchar(mem))
    
    prdDir <- paste0("02_pipeline/runs/CNN/", outDir, "/prd/", year, "/", plot, "/")
    dir.create(prdDir, showWarnings = F, recursive = T)
    image_write(pred, format = "png",
                path = paste0(prdDir, "prd", cell, "_", plot, "_fold", k, ".png"))
    
    # s = 1
    # s = sample(1:length(testData$msk), 1)
    # # i = i2[s]
    # img = as.array(tf$image$decode_png(tf$io$read_file(testData$img[s]), channels = 3))
    # msk = as.array(tf$image$resize(tf$image$decode_png(tf$io$read_file(testData$msk[s]), channels = 1),
    #                                size = c(tilesize, tilesize), method = "nearest"))
    # par(mfrow = c(1,3), oma = c(0,0,0,0), mar = c(.5,.5,.5,.5))
    # plot(as.raster(img/255))
    # plot(as.raster(msk[,,1]))
    # # plot(as.raster(round(preds[s,,,])))
    # plot(as.raster(preds[s,,,]))
    # s = s+1
  }
  stopCluster(cl)
  gc()
  
}



# evaluation --------------------------------------------------------------


sites <- list.files("/media/sysgen/Volume/Felix/UAVforSAT/upscale/var/", pattern = "ortho", recursive = T)
tilesize <- 256L

acc_per_site <- data.frame(year = substr(sites,1,4),
                           site = substr(sites, 6,11),
                           dice = rep(NA, length(sites)),
                           precision = rep(NA, length(sites)),
                           recall = rep(NA, length(sites)),
                           F1 = rep(NA, length(sites)))
pb <- txtProgressBar(max = length(sites), style = 3)
for(i in 1:length(sites)) {
 
  year <- substr(sites[i], 1,4)
  plot <- substr(sites[i], 6,11)
  
  pathImg <- list.files(paste0("02_pipeline/img/", year, "/", plot, "/1024var/"), pattern = ".png", full.names = T)
  pathMsk <- list.files(paste0("02_pipeline/msk/", year, "/", plot, "/1024var/"), full.names = T)
  pathPrd <- list.files(paste0("02_pipeline/runs/176s_t1024var_d256_b3_61epo/prd/", year, "/", plot, "/"), full.names = T)
  
  # cellMsk <- substr(pathMsk, 41,43)
  mem <- strsplit(pathMsk, "_")
  mem <- unlist(lapply(mem, FUN = "[[", 2))
  mem <- strsplit(mem, "/")
  mem <- lapply(mem, FUN = "[[", 7)
  cellMsk <- unlist(lapply(mem, FUN = function(x){substr(x,4,nchar(x))}))
  # cellPrd <- substr(pathPrd, 66,68)
  mem <- strsplit(pathPrd, "_")
  mem <- unlist(lapply(mem, FUN = "[[", 6))
  mem <- strsplit(mem, "/")
  mem <- lapply(mem, FUN = "[[", 6)
  cellPrd <- unlist(lapply(mem, FUN = function(x){substr(x,4,nchar(x))}))
  idx     <- match(cellPrd, cellMsk)
  
  msks <- lapply(pathMsk[idx], FUN = function(x) {
    as.array(tf$image$resize(tf$image$decode_png(tf$io$read_file(x), channels = 1),
                             size = c(tilesize, tilesize), method = "nearest"))
    }) 
  msks <- abind::abind(msks)
  # dim(msks)
  prds <- lapply(pathPrd, FUN = function(x) {
    as.array(tf$image$decode_png(tf$io$read_file(x), channels = 1))
  })
  prds <- abind::abind(prds)
  # dim(prds)
  
  prdVec <- as.vector(prds)+1
  mskVec <- as.vector(msks)+1
  # u      <- sort(union(obsVec, prdVec))
  conmat <- caret::confusionMatrix(factor(prdVec, c(0,1)+1), factor(mskVec, c(0,1)+1), positive = "2")
  
  # par(mfrow = c(1,3), mar = rep(.1, 4))
  # s = 1
  # img = as.array(tf$image$decode_png(tf$io$read_file(pathImg[s]), channels = 3))
  # plot(as.raster(img/255))
  # plot(as.raster(msks[,,s]))
  # plot(as.raster(prds[,,s]))
  # s=s+1
  
  acc_per_site$dice[i]      <- as.numeric(dice_coef(y_true = msks, y_pred = prds))
  acc_per_site$precision[i] <- conmat$byClass["Pos Pred Value"] # precision
  acc_per_site$recall[i]    <- conmat$byClass["Sensitivity"] # recall
  acc_per_site$F1[i]        <- (2*conmat$byClass["Pos Pred Value"]*conmat$byClass["Sensitivity"]) / (conmat$byClass["Pos Pred Value"]+conmat$byClass["Sensitivity"])
  
  setTxtProgressBar(pb, i)
}


write_csv(acc_per_site, paste0("02_pipeline/runs/", outDir, "/acc_per_site.csv"))
outDir = "176s_t1024var_d256_b3_61epo"
