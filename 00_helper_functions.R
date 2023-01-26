require(keras)

# load keras model --------------------------------------------------------

loadModel = function(path, epoch = NULL, compile = FALSE, custom_objects = NULL) {
  require(keras)
  saved_epochs = as.numeric(gsub(".*weights.(.+)-.*", "\\1", paste0(path, "/", list.files(path))))
  if(is.null(epoch)) { # if no epoch specified load best model...
    loss    = gsub(".*-(.+).hdf5.*", "\\1", paste0(path, "/", list.files(path)))
    loadmod = which(loss == min(loss))[1]
  } else { # else model of specified epoch
    loadmod = which(saved_epochs == epoch)
  }
  # load model
  print(paste0("Loaded model of epoch ", saved_epochs[loadmod], "."))
  load_model_hdf5(paste0(path, "/", list.files(path)[loadmod]), compile = compile, custom_objects = custom_objects)
}


# Subsampling dataset -----------------------------------------------------

dataSplit <- function(probTest = 0.1, probTrain = 0.75, seed = 28, tilesize = tilesize) {
  require(raster)
  
  set.seed(seed)
  m = matrix(NA, 9,9)
  
  nTest  = floor(81*probTest)
  nTrain = ceiling( (81-nTest)*probTrain ) 
  
  test  = sample(1:81, nTest)
  train = sample(c(1:81)[-test], nTrain)
  valid = c(1:81)[-c(test,train)]
  m[train] = 1; m[valid] = 2; m[test]  = 3
  
  rasM = raster(m)
  shpM = rasterToPolygons(rasM, dissolve = F)
  
  if(tilesize == 1024) x = 9 else if(tilesize == 512) x = 18 else x = 36
  mTarget = matrix(1:x^2, x, x)
  rasTarget = raster(mTarget)
  
  idxList = raster::extract(rasTarget, shpM)
  idxTest = unlist(idxList[test])
  idxTrain = unlist(idxList[train])
  idxValid = unlist(idxList[valid])
  
  return(list(test = idxTest, train = idxTrain, valid = idxValid))
}


speciesOccurence = function(data, tilesize = tilesize) {
  out = matrix(NA, nrow = 15, ncol = 3)
  for(j in 1:length(data)) {
    msks = array(NA, dim = c(length(data[[j]]$msk), tilesize, tilesize))
    
    pb = txtProgressBar(min = 0, max = dim(msks)[1], style = 3)
    for(i in 1:dim(msks)[1]) {
      msks[i,,] = as.array(stack(data[[j]]$msk[i]))
      setTxtProgressBar(pb, i)
    }
    t = table(msks)
    out[match(as.numeric(names(t)), 1:14) , j] = t
  }
  return(out)
}


# custom loss function ----------------------------------------------------

weightedCategoricalCrossentropy <- function(yTrue, yPred, weights = invWeights) {
  
  # code based on the following resources:
  # https://keras.rstudio.com/articles/examples/unet.html
  # https://stackoverflow.com/questions/51316307/custom-loss-function-in-r-keras
  
  kWeights <- k_constant(weights, dtype = tf$float32, shape = c(1,1,1,12))
  
  yWeights <- kWeights * yPred
  yWeights <- k_sum(yWeights, axis = 4L)
  
  loss     <- tf$keras$losses$categorical_crossentropy(yTrue, yPred)
  wLoss    <- yWeights * loss
  
  return(tf$reduce_mean(wLoss))
}
# weightedCategoricalCrossentropy(yTrue, yPred, invWeights)

wcce_loss <- function(yTrue, yPred) {
  result <- weightedCategoricalCrossentropy(yTrue, yPred, weights = invWeights)
  return(result)
}
# wcce_loss(yTrue, yPred)

weightedBinaryCrossentropy <- function(y_true, y_pred, weights = invWeights) {
  
  # code based on the following resources:
  # https://keras.rstudio.com/articles/examples/unet.html
  # https://stackoverflow.com/questions/51316307/custom-loss-function-in-r-keras
  
  kWeights <- k_constant(weights, dtype = tf$float32, shape = c(1,1,1,length(weights)))
  
  yWeights <- kWeights * y_pred
  yWeights <- k_sum(yWeights, axis = 4L)
  
  loss     <- tf$keras$losses$binary_crossentropy(y_true, y_pred)
  wLoss    <- yWeights * loss
  
  return(tf$reduce_mean(wLoss))
}

wbce_loss <- function(y_true, y_pred) {
  result <- weightedBinaryCrossentropy(y_true, y_pred, weights = invWeights)
  return(result)
}

# dice <- custom_metric("dice", function(y_true, y_pred, smooth = 1.0) {
#   y_true_f <- k_flatten(y_true)
#   y_pred_f <- k_flatten(y_pred)
#   intersection <- k_sum(y_true_f * y_pred_f)
#   (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
# })
# 
# dice_coef <- function(y_true, y_pred, smooth = 1.0) {
#   y_true_f <- k_flatten(y_true)
#   y_pred_f <- k_flatten(y_pred)
#   intersection <- k_sum(y_true_f * y_pred_f)
#   result <- (2 * intersection + smooth) /
#     (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
#   return(result)
# }

dice_coef <- custom_metric("dice", function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / 
    (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(result)
})

bce_dice_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
}

jaccard_index <- function(y_true, y_pred){
  threshold = 0.5
  y_true = k_cast(k_greater(y_true, threshold), 'float32')
  y_pred = k_cast(k_greater(y_pred, threshold), 'float32')
  intersection = k_sum(y_true * y_pred)
  union = k_sum(y_true) + k_sum(y_pred)
  # avoid division by zero by adding 1
  jaccard = (intersection + k_epsilon()) / (union - intersection + k_epsilon())  # this contains as many elements as there are classes
  return (jaccard)
}

jaccard <- function(y_true, y_pred){
  # y_true = k_cast(y_true, 'float16')
  # y_pred = k_cast(y_pred, 'float16')
  intersection = k_sum(k_abs(y_true * y_pred))
  union = k_sum(y_true) + k_sum(y_pred)
  # avoid division by zero by adding 1
  jaccard = (intersection + k_epsilon()) / (union - intersection + k_epsilon())  # this contains as many elements as there are classes
  return (jaccard)
}

jaccard_loss <- function(y_true, y_pred) {
  result <- 1 - jaccard(y_true, y_pred)
  return(result)
}

jaccard_wce_loss <- function(y_true, y_pred, invWeights = invWeights) {
  result <- weightedCategoricalCrossentropy(y_true, y_pred, invWeights) +
    (1 - jaccard(y_true, y_pred))
  return(result)
}


tversky <- function(y_true, y_pred, alpha = 0.8) {
  # https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Focal-Tversky-Loss
  # https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
  # https://amaarora.github.io/2020/06/29/FocalLoss.html
  y_true_pos <- k_flatten(y_true)
  y_pred_pos <- k_flatten(y_pred)
  
  TP <- k_sum(y_true_pos * y_pred_pos)
  FN <- k_sum(y_true_pos * (1 - y_pred_pos))
  FP <- k_sum((1 - y_true_pos) * y_pred_pos)
  
  return((TP + k_epsilon()) / (TP + alpha*FP + (1-alpha)*FN + k_epsilon())) 
}

tversky_loss <- function(y_true, y_pred) {
  result <- 1 - tversky(y_true, y_pred)
  return(result)
}


focalTversky_loss <- function(y_true, y_pred, gamma = 2., alpha = 0.7) {
  y_true_pos <- k_flatten(y_true)
  y_pred_pos <- k_flatten(y_pred)
  
  TP <- k_sum(y_true_pos * y_pred_pos)
  FN <- k_sum(y_true_pos * (1 - y_pred_pos))
  FP <- k_sum((1 - y_true_pos) * y_pred_pos)
  
  tversky <- (TP + k_epsilon()) / (TP + alpha*FP + (1-alpha)*FN + k_epsilon())
  return(k_pow((1 - tversky), gamma))
}


# decode one-hot encodings ------------------------------------------------

decode_one_hot <- function(x, progress = TRUE) { # TODO: Rewrite using foreach
  require(raster)
  results = array(data = NA, dim = dim(x)[-4])
  if(progress)  pb = txtProgressBar(min = 0, max = dim(results)[1], style = 3)
  for (i in 1:dim(results)[1]) {
    results[i,,] = as.vector(t(which.max(brick(x[i,,,]))))
    if(progress) setTxtProgressBar(pb, i)
  }
  results
}

decodeOneHot <- function(x, progress = TRUE) {
  results = array(data = NA, dim = dim(x)[-4])
  if(progress)  pb = txtProgressBar(min = 0, max = dim(results)[1], style = 3)
  for (i in 1:dim(results)[1]) {
    results[i,,] = as.array(tf$argmax(x[i,,,], axis = 2L))
    if(progress) setTxtProgressBar(pb, i)
  }
  results
}
