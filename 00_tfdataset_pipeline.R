
createDataset <- function(data,
                          train, # logical. TRUE for augmentation of training data
                          batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                          epochs,
                          useDSM = FALSE, 
                          shuffle  = TRUE, # logical. default TRUE, set FALSE for test data
                          tileSize = as.integer(tilesize),
                          datasetSize) { # numeric. number of samples per epoch the model will be trained on
  require(tfdatasets)
  require(purrr)
  if(useDSM) chnl <- 4L else chnl <- 3L
  
  if(shuffle){
    dataset = data %>%
      tensor_slices_dataset() %>%
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE, seed = 1L)
  } else {
    dataset = data %>%
      tensor_slices_dataset() 
  } 
  
  dataset = dataset %>%
    dataset_map(~.x %>% list_modify( # read files and decode png
      img = tf$image$decode_png(tf$io$read_file(.x$img), channels = chnl) %>%
        tf$image$resize(size = c(tileSize, tileSize), method = "nearest") %>%
        tf$image$convert_image_dtype(dtype = tf$float32) %>%
        tf$reshape(shape = c(tileSize, tileSize, chnl)),
      msk = tf$image$decode_png(tf$io$read_file(.x$msk), channels = 1L) %>%
        tf$image$resize(size = c(tileSize, tileSize), method = "nearest") %>%
        # tf$image$convert_image_dtype(dtype = tf$float32) %>%
        tf$reshape(shape = c(tileSize, tileSize, 1L))
    ), num_parallel_calls = parallel::detectCores())
  
  if(train) {
    dataset = dataset %>%
      dataset_map(~.x %>% list_modify( # randomly flip up/down and left/right
        img = layer_random_flip(.x$img, seed = 1L),
        msk = layer_random_flip(.x$msk, seed = 1L)
      ), num_parallel_calls = parallel::detectCores()) %>%
      dataset_map(~.x %>% list_modify( # randomly assign brightness, contrast and saturation to images
        img = tf$image$random_brightness(.x$img, max_delta = 0.1, seed = 1L) %>% 
          tf$image$random_contrast(lower = 0.8, upper = 1.2, seed = 2L) %>%
          tf$image$random_saturation(lower = 0.8, upper = 1.2, seed = 3L) %>% # requires 3 chnl -> with useDSM chnl = 4
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.
      ), num_parallel_calls = parallel::detectCores()) %>%
      dataset_repeat(count = ceiling(epochs * (datasetSize/length(trainData$img))) )
  }
  
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    # dataset_prefetch_to_device(device = "/gpu:0", buffer_size = tf$data$experimental$AUTOTUNE)
}
