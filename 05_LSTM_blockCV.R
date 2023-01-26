
library(keras)
require(raster)
library(tensorflow)
library(tfdatasets)
library(tibble)
library(purrr)
require(abind)
library(imputeTS)
library(signal)
library(caret)
source("00_helper_functions.R")


nY     <- 2
nsites <- 176
nFold  <- 10
nTimes <- 5
add    <- T

yearFlag <- paste0("nY", nY)


# load subsampled timeseries data
RDataFile <- paste0("01_data/data_orig_", nsites, "_104.RData")
load(RDataFile)


## test split
identifier <- paste(S$plot, S$year, sep = "_")
sites      <- unique(identifier)

set.seed(19)
folds  <- createMultiFolds(y = sites, k = nFold, times = nTimes)
splits <- list()
for (i in 1:(nFold*nTimes)) {
  splits$x_train[[i]] <- X[which(identifier %in% sites[folds[[i]]]),,]
  splits$y_train[[i]] <- Y[which(identifier %in% sites[folds[[i]]])]
  splits$s_train[[i]] <- S[which(identifier %in% sites[folds[[i]]]),]
  
  splits$x_test[[i]]  <- X[-which(identifier %in% sites[folds[[i]]]),,]
  splits$y_test[[i]]  <- Y[-which(identifier %in% sites[folds[[i]]])]
  splits$s_test[[i]]  <- S[-which(identifier %in% sites[folds[[i]]]),]
  
  splits$idx[[i]]     <- c(1:length(Y))[-which(identifier %in% sites[folds[[i]]])]
}


param <- expand.grid(res = c(10, 20, 60),
                     S1 = c(T, F),
                     S2 = c(T, F),
                     VI = c(T, F))
param <- param[-which(param$S1 == F & param$S2 == F & param$VI == F),]
param <- param[-which(param$S1 == T & param$S2 == F & param$VI == T),]
param <- param[-which(param$S1 == F & param$S2 == F & param$VI == T),]
param <- param[-which(param$S1 == T & param$S2 == F & param$VI == F),]

param$RMSE <- param$MAE <- param$R2 <- param$slope <- param$intercept <- NA


for(g in 1:nrow(param)) {
  
  res <- param$res[g]
  S1  <- param$S1[g]
  S2  <- param$S2[g]
  VI  <- param$VI[g]
  
  resFlag  <- paste0(res, "m")
  S1Flag   <- if(S1) "S1T" else "S1F"
  S2Flag   <- if(S2) "S2T" else "S2F"
  VIFlag   <- if(VI) "VIT" else "VIF"
  
  outDir   <- paste0("02_pipeline/runs/LSTM/", resFlag, "_", yearFlag, "_", S1Flag, "_", S2Flag, "_", VIFlag, "/")
  if(!dir.exists(outDir)) dir.create(outDir, recursive = T)

  
  ## band selection
  bIdx <- 1:18
  if(res == 10) bIdx[c(1,5:7,9:12)] <- NA else if (res == 20) bIdx[c(1,10)] <- NA
  if(!S1)       bIdx[15:18] <- NA
  if(!S2)       bIdx[1:12] <- NA
  if(!VI)       bIdx[13:14] <- NA
  bIdx <- as.numeric(na.omit(bIdx))
  
  
  ## background data forest floor
  if(add) {
    
    load("01_data/data_add_104.RData")
    
    add_X <- add_X[,,bIdx]
    if(length(bIdx) == 1) add_X <- array_reshape(add_X, dim = c(dim(add_X), 1))
    
  }
  
  
  results         <- data.frame(loss = rep(NA, nFold*nTimes), mae = rep(NA, nFold*nTimes))
  preds           <- as.data.frame(matrix(data = NA, nrow = dim(X)[1], ncol = nTimes))
  colnames(preds) <- paste0("Rep", 1:5)
 
  NO_EPOCHS <- 200L
  BATCH     <- 128L
  LENGTH_TS <- 104L
  NUM_B     <- length(bIdx)
  for (i in 1:(nFold*nTimes)) {
    
    model <- keras_model_sequential() %>%
      bidirectional(layer_lstm(units = 100L,
                               input_shape=c(LENGTH_TS, NUM_B),
                               return_sequences = TRUE,
                               # dropout = 0.2,
                               # recurrent_regularizer = regularizer_l1_l2(),
                               # recurrent_dropout = 0.5,
                               recurrent_activation = "sigmoid")) %>%
      bidirectional(layer_lstm(units = 100L,
                               # dropout = 0.2,
                               # recurrent_regularizer = regularizer_l1_l2(),
                               # recurrent_dropout = 0.5,
                               recurrent_activation = "sigmoid")) %>%
      layer_dense(units = 1, activation = 'sigmoid')
    
    
    model %>% compile(
      loss = 'mse',
      optimizer = optimizer_adam(learning_rate = 0.0005),
      metrics = c("mae")
    )
    
    checkpointDir <- paste0(outDir, names(folds)[i], "/checkpoints/")
    unlink(checkpointDir, recursive = T)
    dir.create(checkpointDir, recursive = TRUE)
    filepath <- file.path(checkpointDir, "weights.{epoch:03d}-{val_loss:.6f}.hdf5")
    
    ckpt_callback <- callback_model_checkpoint(filepath = filepath,
                                               monitor = "val_loss",
                                               save_weights_only = F,
                                               save_best_only = TRUE,
                                               mode = "auto",
                                               save_freq = "epoch")
    
    x_train <- splits$x_train[[i]][,,bIdx]
    if(length(bIdx) == 1) x_train <- array_reshape(x_train, dim = c(dim(x_train), 1))
    y_train <- round(splits$y_train[[i]],2)
    s_train <- splits$s_train[[i]]
    
    if(add) {
      samp <- sample(1:length(add_Y), round(length(y_train)/10))
      Xadd <- add_X[samp,,]
      if(length(bIdx) == 1) Xadd <- array_reshape(Xadd, dim = c(dim(Xadd), 1))
      Yadd <- round(add_Y[samp],2)
      Sadd <- add_S[samp,]
      
      x_train <- abind(x_train, Xadd, along = 1)
      y_train <- c(y_train, Yadd)
      s_train <- rbind(s_train, Sadd)
    }
    
    
    dataset <- tensor_slices_dataset(tibble(x = x_train, y = y_train)) %>%
      dataset_shuffle(buffer_size = dim(x_train)[1], reshuffle_each_iteration = F)
    
    SPLIT <- as.integer(0.75*dim(x_train)[1])
    train_dataset <- dataset$take(SPLIT)
    valid_dataset <- dataset$skip(SPLIT)
    
    train_dataset <- train_dataset %>%
      dataset_shuffle(buffer_size = dim(x_train)[1]) %>%
      dataset_map( ~.x %>% list_modify(
        x = tf$cast(.$x, dtype = "float32"))) %>%
      dataset_batch(BATCH, drop_remainder = TRUE) %>%
      dataset_map(unname) %>%
      dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    
    valid_dataset <- valid_dataset %>%
      dataset_shuffle(buffer_size = dim(x_train)[1]) %>%
      dataset_map( ~.x %>% list_modify(
        x = tf$cast(.$x, dtype = "float32")
      )) %>%
      dataset_batch(BATCH, drop_remainder = TRUE) %>%
      dataset_map(unname) %>%
      dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    
    # iterator_get_next(make_iterator_one_shot(valid_dataset))
    
    history <- model %>% fit(
      x = train_dataset,
      epochs = NO_EPOCHS,
      verbose = 0, view_metrics = T,
      callbacks = ckpt_callback,
      validation_data = valid_dataset
    )
    
    save(history, file = paste0(outDir, names(folds)[i], "/history.RData"))
    
    pdf(paste0(outDir, names(folds)[i], "/history.pdf"))
    print(plot(history))
    dev.off()
    
    ## evaluation on "independent" test dataset
    x_test <- splits$x_test[[i]][,yIdx,bIdx]
    y_test <- splits$y_test[[i]]
    x_test <- array_reshape(x_test, c(length(y_test), LENGTH_TS, NUM_B))
    
    
    test_dataset <- tensor_slices_dataset(tibble(x = x_test, y = y_test)) %>%
      dataset_map( ~.x %>% list_modify(
        x = tf$cast(.$x, dtype = "float32")
      )) %>%
      dataset_batch(1, drop_remainder = TRUE) %>%
      dataset_map(unname) %>%
      dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    
    model <- loadModel(checkpointDir, compile = TRUE)
    eval  <- model %>% evaluate(test_dataset)
    
    results$loss[[i]] <- eval[1]
    results$mae[[i]]  <- eval[2]
    
    ## predict on testdata
    repIdx <- substr(names(folds)[i], 8,11)
    preds[splits$idx[[i]], repIdx] <- as.vector(predict(model, test_dataset))
    
    files <- list.files(checkpointDir, recursive = T, full.names = T)
    unlink(files[1:length(files)-1], recursive = T)
  }
  
  rownames(results) <- names(folds)
  colMeans(results,  na.rm = T)
  write.csv(results, file = paste0(outDir, "CV_results.csv"))
  
  pdf(paste0(outDir, "rep_fit.pdf"))
  for(i in 1:nTimes) {
    plot(Y, preds[,i], xlim = c(0,1), ylim = c(0,1), asp = 1)
    abline(0,1)
  }
  dev.off()
  
  ## predict on testdata
  cols <- as.factor(substr(S$plot, 1,3))
  
  agg_preds <- rowMeans(preds)
  rmse <- sqrt(mean( (agg_preds-Y)^2 ))
  mae  <- mean(abs(agg_preds-Y))
  R2   <- cor(agg_preds, Y)^2
  lmod <- lm(agg_preds~Y)
  
  param$RMSE[g]      <- rmse
  param$MAE[g]       <- mae
  param$R2[g]        <- R2
  param$slope[g]     <- lmod$coefficients[2]
  param$intercept[g] <- lmod$coefficients[1]
  
  load(RDataFile)
  S$prediction       <- agg_preds
  S                  <- cbind(S, preds)
  write.csv(S, file = paste0(outDir, "prediction.csv"))
  
  smooth <- data.frame(mean = rep(NA, 101), sd = rep(NA, 101))
  for(e in 1:101) {
    a <- seq(0,1,.01)
    idx <- which(Y > (a[e]-.05) &  Y <= (a[e]+.05))
    smooth$mean[e] <- mean(agg_preds[idx])
    smooth$sd[e]   <- sd(agg_preds[idx])
  }
  
  pdf(paste0(outDir, "fit.pdf"))
  plot(Y, agg_preds, main = paste0("RMSE = ", round(rmse, 3), "; MAE = ", round(mae, 3), "; R² = ", round(R2, 2)), asp = 1,
       cex.axis = 1.3, cex.lab = 1.3, xlim = c(0,1), ylim = c(0,1), pch = 19, col = cols,
       xlab = "observation", ylab = "prediction", type = "n")
  grid(col = "darkgrey")
  points(Y, agg_preds, pch = 19, col = cols)
  abline(0,1, lty = 2)
  abline(lmod)
  text(0, 1, adj = 0, paste0("y=", round(lmod$coefficients[2],3), "x+", round(lmod$coefficients[1],3)), )
  legend("bottomright", col = 1:7, legend = levels(cols), pch = 19, cex = .8, bg = NA)
  
  plot(Y, agg_preds, xlim = c(0,1), ylim = c(0,1), col = alpha("black", .5), cex = .5, type = "n",
       cex.axis = 1.3, cex.lab = 1.3, asp = 1,
       xlab = "observation", ylab = "prediction",
       main = paste0("RMSE = ", round(rmse, 3), "; MAE = ", round(mae, 3), "; R² = ", round(R2, 2)))
  grid(col = "darkgrey")
  polygon(x = c(seq(0,1,.01),seq(1,0,-.01)), y = c(smooth$mean-smooth$sd, rev(smooth$mean+smooth$sd)),
          col = alpha("navy", .2), border = NA)
  points(Y, agg_preds, xlim = c(0,1), ylim = c(0,1), col = alpha("black", .5), cex = .5, xlab = "observation", ylab = "prediction")
  lines(seq(0,1,.01), smooth$mean,  xlim = c(0,1), ylim = c(0,1), col = "navy", lwd = 4,
        pch = 19, xlab = "observation", ylab = "prediction")
  abline(0,1, lty = 2)
  abline(lmod)
  text(0, 1, adj = 0, paste0("y=", round(lmod$coefficients[2],3), "x+", round(lmod$coefficients[1],3)))
  dev.off()
  

  
  # final model -------------------------------------------------------------
  
  model <- keras_model_sequential() %>%
    bidirectional(layer_lstm(units = 100L,
                             input_shape=c(LENGTH_TS, NUM_B),
                             return_sequences = TRUE,
                             recurrent_activation = "sigmoid")) %>%
    bidirectional(layer_lstm(units = 100L,
                             recurrent_activation = "sigmoid")) %>%
    layer_dense(units = 1, activation = 'sigmoid')
  
  model %>% compile(
    loss = 'mse',
    optimizer = optimizer_adam(learning_rate = 0.0005),
    metrics = c("mae")
  )
  
  
  finDir        <- paste0("02_pipeline/runs/LSTM/", resFlag, "_", yearFlag, "_", S1Flag, "_", S2Flag, "_", VIFlag, "/final/")
  checkpointDir <- paste0(finDir, "/checkpoints/")
  unlink(checkpointDir, recursive = T)
  dir.create(checkpointDir, recursive = TRUE)
  filepath <- file.path(checkpointDir, "weights.{epoch:03d}-{val_loss:.6f}.hdf5")
  
  ckpt_callback <- callback_model_checkpoint(filepath = filepath,
                                             monitor = "val_loss",
                                             save_weights_only = F,
                                             save_best_only = TRUE,
                                             mode = "auto",
                                             save_freq = "epoch")
  
  x_train <- X[,yIdx,bIdx]
  y_train <- Y
  s_train <- S
  x_train <- array_reshape(x_train, c(length(y_train), LENGTH_TS, NUM_B))
  
  if(add) {
    samp <- sample(1:length(add_Y), round(length(y_train)/10))
    Xadd <- add_X[samp,,]
    if(length(bIdx) == 1) Xadd <- array_reshape(Xadd, dim = c(dim(Xadd), 1))
    Yadd <- round(add_Y[samp],2)
    Sadd <- add_S[samp,]
    
    x_train <- abind(x_train, Xadd, along = 1)
    y_train <- c(y_train, Yadd)
    s_train <- rbind(s_train[,-c(8:13)], Sadd)
  }
  
  shuffleIDX <- sample(1:length(y_train), size = length(y_train), replace = F)
  x_train <- x_train[shuffleIDX,,]
  if(length(bIdx) == 1) x_train <- array_reshape(x_train, dim = c(dim(x_train), 1))
  y_train <- y_train[shuffleIDX]
  s_train <- s_train[shuffleIDX,]
  
  history <- model %>% fit(
    x_train, y_train,
    epochs = 300,
    batch_size = BATCH,
    callbacks = ckpt_callback,
    verbose = 0, view_metrics = T,
    validation_split = 0.25, # internal validation split for validation after each epoch
  )
  
  files <- list.files(finDir, recursive = T, full.names = T)
  unlink(files[1:length(files)-1], recursive = T)
  
  model <- loadModel(checkpointDir, compile = TRUE)
  preds <- as.vector(predict(model, x_train))
  lmod  <- lm(preds~y_train)
  
  pdf(paste0(outDir, "final_model_fit.pdf"))
  plot(y_train, preds, xlim = c(0,1), ylim = c(0,1), col = alpha("black", .5), cex = .75,
       cex.axis = 1.3, cex.lab = 1.3, asp = 1,
       xlab = "observation", ylab = "prediction",)
  grid()
  abline(0,1, lty = 2, lwd = 2)
  abline(lmod, lwd = 2)
  text(0, 1, adj = 0, paste0("y=", round(lmod$coefficients[2],3), "x+", round(lmod$coefficients[1],3)))
  dev.off()
  
  write.csv(param, paste0("02_pipeline/runs/LSTM/comparison_nY", nY, "_lstm", nsites,".csv"))
  
}

# write.csv(param, paste0("02_pipeline/runs/LSTM/comparison_nY", nY, "_lstm", nsites,".csv"))
param <- read.csv(paste0("02_pipeline/runs/LSTM/comparison_nY", nY, "_lstm", nsites,".csv"), row.names = 1)


## application of the model to landscape level runs on CREODIAS cloud 