
# library + path ----------------------------------------------------------

pkgs <- c("raster", "rgdal", "rgeos", "keras", "tensorflow", "tibble", "foreach", "doParallel", "abind", "magick", "tfdatasets", "purrr", "stars", "velox")
sapply(pkgs, require, character.only = TRUE)

source("00_helper_functions.R")
source("00_tfdataset_pipeline.R")


# set prerequisites -------------------------------------------------------

outDir <- "176s_t1024var_d256_60epo"

imgSize  <- as.numeric( substr(strsplit(outDir, "_")[[1]][2], 2, nchar(strsplit(outDir, "_")[[1]][2])-3) )/100
tileSize <- as.integer( substr(strsplit(outDir, "_")[[1]][3], 2, nchar(strsplit(outDir, "_")[[1]][3]))   )
movingWindow <- F

# Load data ---------------------------------------------------------------

# orthoFolder <- "/media/sysgen/Volume/Felix/UAVforSAT/upscale/rcl/"
orthoFolder <- "PATH/TO/ORTHOMOSAIC/TIF"
outputFolder <- paste0("03_output/predictions/CNN/", outDir)

# all scenes
scenes      <- list.files(orthoFolder, pattern = ".tif", recursive = T)
years       <- substr(scenes, 1, 4)
sites       <- substr(scenes, 6, 11)
outputFiles <- paste0("prd_", years, "_", sites, ".tif")


pb <- txtProgressBar(max = length(scenes), style = 3)
for(k in 1:length(scenes)) {#1:length(scenes)
  
  scene <- scenes[k]
  year  <- substr(scene, 1,4)
  site  <- substr(scene, 6,11)
  
  orthoFile <- paste0(orthoFolder, year, "/", site, "_ortho.tif")
  ortho     <- raster::stack(orthoFile)
  ortho     <- ortho[[-4]]
  
  foldFiles  <- list.files(paste0("02_pipeline/runs/CNN/", outDir, "/prd/"), recursive = T)
  siteSub    <- substr(foldFiles, 1, 11)
  mem        <- strsplit(foldFiles, "_")
  mem        <- lapply(mem, "[[", 3)
  foldSub    <- unlist(lapply(mem, function(x){substr(x,5,5)}))
  uniqueFold <- unique(paste(siteSub, foldSub, sep = "/"))
  kTab <- data.frame(year = substr(uniqueFold, 1, 4), site = substr(uniqueFold, 6, 11), fold = substr(uniqueFold, 13, 13)) 
  
  fold <- kTab$fold[which(kTab$year == year & kTab$site == site)]
  if(length(fold) == 0) fold <- 1
  checkpointDir <- paste0("02_pipeline/runs/CNN/", outDir, "/Fold", fold, "/checkpoints/")
  model         <- loadModel(checkpointDir, compile = TRUE)
  
  
  ## apply histogram stretch
  # if(!file.exists(paste0(substr(orthoFolder,1,45), "rcl/", year, "/", site, "_ortho.tif"))) {
  #   values(ortho)[values(ortho[[1]]) == 0 & values(ortho[[2]]) == 0 & values(ortho[[3]]) == 0] <- NA
  #   if(substr(site,1,3) %in% c("HAI", "SAX", "DDH")) {
  #     values(ortho)[values(ortho[[1]]) == 255 & values(ortho[[2]]) == 255 & values(ortho[[3]]) == 255] <- NA
  #   }
  #   q     <- quantile(ortho, probs = c(.001, .999), na.rm = T)
  #   ortho <- (ortho-min(q[,1])) * 255 / (max(q[,2]) - min(q[,1]))
  #   # f1 <- function(x, q) (x-min(q[,1])) * 255 / (max(q[,2]) - min(q[,1]))
  #   beginCluster()
  #   # ortho <- clusterR(ortho, fun = f1, args = list(q = q), export = 'q')
  #   ortho <- clusterR(ortho, fun = reclassify, args = list(rcl = c(-Inf,0,0, 255,Inf,255)), datatype = "INT1U")
  #   endCluster()
  #   values(ortho)[values(ortho[[1]]) == 0 & values(ortho[[2]]) == 0 & values(ortho[[3]]) == 0] <- NA
  #   ortho <- stack(ortho)
  #   writeRaster(ortho, filename = paste0(substr(orthoFolder,1,45), "rcl/", year, "/", site, "_ortho.tif"),
  #               datatype = "INT1U", options = c("COMPRESS=LZW"), overwrite = T)
  #   rm(ortho, q)
  #   removeTmpFiles(h=0); gc() 
  # }
  # ortho <- stack(paste0(substr(orthoFolder,1,45), "rcl/", year, "/", site, "_ortho.tif"))
  plotRGB(ortho, colNA = "pink")
  
  
  ## define kernel size
  kernelSizeX <- floor(imgSize / xres(ortho))
  kernelSizeY <- floor(imgSize / yres(ortho))
  
  nTilesX <- floor(ncol(ortho) / kernelSizeX)
  nTilesY <- floor(nrow(ortho) / kernelSizeY)
  
  xOffset <- round( ( ncol(ortho) - (nTilesX * kernelSizeX) ) / 2 )
  yOffset <- round( ( nrow(ortho) - (nTilesY * kernelSizeY) ) / 2 )
  
  ## create sample position
  idxX    <- seq(1, nTilesX * kernelSizeX, kernelSizeX) + xOffset
  idxY    <- seq(1, nTilesY * kernelSizeY, kernelSizeY) + yOffset
  idxGrd  <- expand.grid(idxX, idxY)
  
  xOffsetGeo <- xOffset*xres(ortho)
  yOffsetGeo <- yOffset*yres(ortho)
  grd <- raster(vals = 1:(nTilesX*nTilesY), nrows = nTilesY, ncol = nTilesX, crs = crs(ortho),
                xmn = xmin(ortho)+xOffsetGeo, xmx = xmax(ortho)-xOffsetGeo,
                ymn = ymin(ortho)+yOffsetGeo, ymx = ymax(ortho)-yOffsetGeo)
  grdVec <- rasterToPolygons(grd)
  lines(grdVec, col = "red")
  
  
  if(!movingWindow) {
    
    if(!dir.exists(paste0(outputFolder, "tif/"))) dir.create(paste0(outputFolder, "tif/"), recursive = T)
    
    ## load NA coverage per tile
    grdperc <- raster(paste0("01_data/coverage_NAvalues/percNA_", year, "_", site, ".tif"))
    grdBin  <- Which(grdperc > 0.90)
    grdBin[grdBin == 0] <- NA
    grdBinVec <- rasterToPolygons(grdBin)
    lines(grdBinVec, col = "blue")
    
    
    cl <- makeCluster(19)
    registerDoParallel(cl)
    
    crops <- foreach(i = 1:nrow(idxGrd), .packages = c("raster","magick"), .inorder = TRUE) %dopar% {
      xmin <- idxGrd[i,2]
      xmax <- idxGrd[i,2]+kernelSizeX-1
      ymin <- idxGrd[i,1]
      ymax <- idxGrd[i,1]+kernelSizeY-1
      cropExtent <- extent(ortho, xmin, xmax, ymin, ymax)
      orthoCrop <- output <- crop(ortho, cropExtent)
      
      extent(orthoCrop) <- extent(0, kernelSizeX, 0, kernelSizeY)
      orthoCrop <- as.array(orthoCrop)
      orthoCrop <- image_read(orthoCrop / 255)
      image_write(orthoCrop, format = "png", path = paste0("02_pipeline/tmp/img", sprintf("%05d", i), ".png"))
      
      return(output)
      
      rm(orthoCrop, cropExtent, output)
      gc()
    }
    
    stopCluster(cl)
    
    validTiles <- unname(unlist(velox(grd)$extract(sp = grdBinVec)))

    prediction <- raster(vals = NA, nrows = nTilesY*tileSize, ncol = nTilesX*tileSize, crs = crs(ortho),
                         xmn = xmin(ortho)+xOffsetGeo, xmx = xmax(ortho)-xOffsetGeo,
                         ymn = ymin(ortho)+yOffsetGeo, ymx = ymax(ortho)-yOffsetGeo)
    
    for(i in validTiles) {
      
      xmin <- idxGrd[i,2]
      xmax <- idxGrd[i,2]+kernelSizeX-1
      ymin <- idxGrd[i,1]
      ymax <- idxGrd[i,1]+kernelSizeY-1
      cropExtent <- extent(ortho, xmin, xmax, ymin, ymax)
      
      NACrop <- raster::extract(grdBin, cropExtent)
      if(!is.na(NACrop)) {
        img <- tf$image$decode_png(tf$io$read_file(paste0("02_pipeline/tmp/img", sprintf("%05d", i), ".png")), channels = 3L) %>%
          tf$image$convert_image_dtype(dtype = tf$float32) %>%
          tf$image$resize(size = c(tileSize, tileSize), method = "nearest") %>%
          tf$reshape(shape = c(1L, tileSize, tileSize, 3L))
        
        pred <- round(predict(model, img))[1,,,]
        
        cellIdx             <- cellsFromExtent(prediction, cropExtent)
        prediction[cellIdx] <- t(pred)
      }
    }
    
    plot(prediction, colNA = "pink")
    lines(grdBinVec, col = "red")
    
    writeRaster(prediction, paste0(outputFolder, "tif/", outputFiles[k]), overwrite = T)
    
    rm(ortho, prediction, crops, model)
    removeTmpFiles(h=0)
    gc()
    
    unlink(list.files("02_pipeline/tmp/", full.names = T))
    
  } else {
    
    if(!dir.exists(paste0(outputFolder, "MV/tif/"))) dir.create(paste0(outputFolder, "MV/tif/"), recursive = T)
    
    ## load NA coverage per tile
    uav1Vel <- velox(ortho[[1]])
    numpix  <- function(x) length(na.omit(x)) / length(x)
    nas     <- uav1Vel$extract(grdVec, fun = numpix)
    grdperc <- grd; values(grdperc) <- nas
    # plot(grdperc)

    grdBin  <- Which(grdperc >= 0.80)
    grdBin[grdBin == 0] <- NA
    grdBinVec <- rasterToPolygons(grdBin)
    lines(grdBinVec, col = "blue")
    grdBound <- boundaries(grdBin)
    grdBound <- Which(grdBound == 0)
    grdBound[grdBound == 0] <- NA
    grdBoundVec <- rasterToPolygons(grdBound)
    lines(grdBoundVec, col = "yellow")
    
    validTiles <- unname(unlist(velox(grd)$extract(sp = grdBoundVec)))
    
    # window shifts
    shift <- expand.grid(x = c(-tileSize/2, 0, tileSize/2), y = c(-tileSize/2, 0, tileSize/2))
    
    prediction <- list()
    for(p in 1:nrow(shift)) {
      
      xShift <- xres(ortho)*shift$x[p]
      yShift <- yres(ortho)*shift$y[p]
      
      cl <- makeCluster(19)
      registerDoParallel(cl)
      crops <- foreach(i = validTiles, .packages = c("raster","magick"), .inorder = TRUE) %dopar% {
        xmin <- idxGrd[i,2]+xShift
        xmax <- idxGrd[i,2]+kernelSizeX-1+xShift
        ymin <- idxGrd[i,1]+yShift
        ymax <- idxGrd[i,1]+kernelSizeY-1+yShift
        cropExtent <- extent(ortho, xmin, xmax, ymin, ymax)
        orthoCrop <- output <- crop(ortho, cropExtent)
        
        extent(orthoCrop) <- extent(0, kernelSizeX, 0, kernelSizeY)
        orthoCrop <- as.array(orthoCrop)
        orthoCrop <- image_read(orthoCrop / 255)
        image_write(orthoCrop, format = "png", path = paste0("02_pipeline/tmp/img", sprintf("%05d", i), ".png"))
        
        return(output)
        
        rm(orthoCrop, cropExtent, output)
        gc()
      }
      stopCluster(cl)
      
      prediction[[p]] <- raster(vals = NA, nrows = (nTilesY+1)*tileSize, ncol = (nTilesX+1)*tileSize, crs = crs(ortho),
                                xmn = xmin(ortho)+xOffsetGeo+min(shift$x)*0.04,
                                xmx = xmax(ortho)-xOffsetGeo+max(shift$x)*0.04,
                                ymn = ymin(ortho)+yOffsetGeo+min(shift$y)*0.04,
                                ymx = ymax(ortho)-yOffsetGeo+max(shift$y)*0.04)
      
      for(i in validTiles) {
        
        xmin <- idxGrd[i,2]+xShift
        xmax <- idxGrd[i,2]+kernelSizeX-1+xShift
        ymin <- idxGrd[i,1]+yShift
        ymax <- idxGrd[i,1]+kernelSizeY-1+yShift
        cropExtent <- extent(ortho, xmin, xmax, ymin, ymax)
        
        img <- tf$image$decode_png(tf$io$read_file(paste0("02_pipeline/tmp/img", sprintf("%05d", i), ".png")),
                                   channels = 3L) %>%
          tf$image$convert_image_dtype(dtype = tf$float32) %>%
          tf$image$resize(size = c(tileSize, tileSize), method = "nearest") %>%
          tf$reshape(shape = c(1L, tileSize, tileSize, 3L))
        
        pred <- round(predict(model, img))[1,,,]
        
        cellIdx                  <- cellsFromExtent(prediction[[p]], cropExtent)
        prediction[[p]][cellIdx] <- t(pred)
      }
      
      unlink(list.files("02_pipeline/tmp/", full.names = T))
    }
    
    AOI <- readOGR("01_data/shape/upscale_AOIs.shp")
    AOI <- AOI[AOI$plot_no == site,]
    AOI <- spTransform(AOI, crs(prediction[[1]]))
    AOI <- gBuffer(AOI, width = 10)
    prediction <- lapply(prediction, crop, y=AOI)
    
    # stack predictions
    predStack <- stack(prediction)
    
    # majority vote
    beginCluster()
    prediction <- clusterR(predStack, calc, args = list(modal, na.rm = T, ties = "random"))
    endCluster()
    
    plot(prediction, colNA = "pink")
    lines(grdBoundVec, col = "red")
    
    outputFile <- paste0("prd_", year, "_", site, "_MV.tif")
    writeRaster(prediction, paste0(outputFolder, "MV/tif/", outputFile), overwrite = T)
    
    rm(ortho, prediction, crops, predStack)
    removeTmpFiles(h=0)
    gc()
    
  }
  
  setTxtProgressBar(pb, k)
  
}


pred_tifs <- list.files(paste0(outputFolder, "tif/"), pattern = ".tif")

cl <- makeCluster(19)
registerDoParallel(cl)

foreach(i = 1:length(pred_tifs), .packages = c("stars", "raster")) %dopar% {

  # load tif
  r <- raster(paste0(outputFolder, "tif/", pred_tifs[i]))
  s <- st_as_stars(r, ignore_file = T)
  s[[1]][is.na(s[[1]])] <- 9
  
  # resample
  dest <- st_as_stars(st_bbox(r), dx = 0.04)
  w    <- st_warp(s, dest)
  # plot(w)
  
  # polygonize
  v <- st_as_sf(s, as_points = F, merge = T, na.rm = F, use_integer = T)
  # plot(v)
  
  # write file
  if(!dir.exists(paste0(outputFolder, "shape/"))) dir.create(paste0(outputFolder, "shape/"))
  outfile <- paste0(outputFolder, "shape/", names(v)[1], ".shp")
  names(v)[1] <- "prd"
  st_write(v, outfile, quiet = T)
  
  rm(r, s, dest, w, v, outfile)
  removeTmpFiles(h=0)
  gc()
  
}

stopCluster(cl)
