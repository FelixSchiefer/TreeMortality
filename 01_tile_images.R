

# libraries + path --------------------------------------------------------

pkgs <- c("raster", "rgdal", "rgeos", "foreach", "doParallel", "magick")
sapply(pkgs, require, character.only = TRUE)

source("00_helper_functions.R")


# segment images function -------------------------------------------------

# dataDir <- "/media/sysgen/Volume/Felix/UAVforSAT/upscale/rcl/"
dataDir <- "PATH/TO/ORTHOMOSAIC/TIF"
files <- list.files(dataDir, pattern = "ortho", recursive = T)

tilesize = 10.24; plot = T; overwrite = T

segmentImages <- function(files, # character
                          # outDir, # path to outputfolder
                          useDSM = FALSE,
                          tilesize, # numeric, tilesize in meters
                          plot = TRUE,
                          overwrite = FALSE) { # NAvalue, histogram stretch
  
  sites <- substr(files, 6, 11)
  years <- substr(files, 1, 4)
  
  ## loop over sites
  for(j in 1:length(files)) {
    
    site <- sites[j]
    year <- years[j]
    
    #### load data ####
    message(paste0("loading data ", j, "/", length(sites)))
    
    ## create outputfolder
    tileTag <- paste0("t", tilesize*100, "var")

    imgDir <- paste0("02_pipeline/", "img/", year, "/", site, "/", tilesize*100, "var/")
    mskDir <- paste0("02_pipeline/", "msk/", year, "/", site, "/", tilesize*100, "var/")
    if(!dir.exists(imgDir)){
      dir.create(imgDir, recursive = TRUE)
      dir.create(mskDir, recursive = TRUE)
    }
    
    ## remove old files if overwrite == TRUE
    if(overwrite) {
      unlink(list.files(imgDir, full.names = TRUE))
      unlink(list.files(mskDir, full.names = TRUE))
    }
    if(length(list.files(imgDir)) > 0 & overwrite == FALSE) {
      stop(paste0("Can't overwrite files in ", imgDir, " -> set 'overwrite = TRUE'"))
    }
    
    ## load ortho
    orthoFile <- paste0(dataDir, files[j])
    ortho     <- stack(orthoFile)
    ortho     <- ortho[[-4]] # remove alpha channel
    
    ## load area of interest
    AOI <- readOGR(dsn = "01_data/shape/upscale_AOIs.shp", verbose = FALSE)
    AOI <- AOI[AOI$plot_no == site, ]
    AOI <- spTransform(AOI, crs(ortho))
    AOI <- gBuffer(AOI, byid = TRUE, width = 0)
    
    
    ## crop ortho to AOI
    ortho <- crop(ortho, AOI)
    if(substr(site,1,3) %in% c("FIN", "HAI", "KAB")) ortho <- mask(ortho, AOI)
    
    
    ## load reference data
    shape <- NULL
    try({shape <- readOGR(dsn = paste0("01_data/shape/delineation/", year, "/poly_", site, "_deadwood.shp"), verbose = FALSE)
    shape <- gBuffer(shape, byid = TRUE, width = 0)
    shape <- spTransform(shape, crs(ortho))
    shape$species_ID <- as.numeric(shape$species_ID)-7}, silent = T)
    
    
    ## plot site
    if(plot) {
      plotRGB(ortho, colNA = "pink")
      lines(shape, lwd = 1.5, col = "orange")
      lines(AOI, col = "red", lwd = 2)
    }
    
    
    bg               <- AOI
    bg$species_ID    <- 0
    bg@data          <- data.frame(species_ID = bg@data$species_ID, row.names = rownames(bg@data))
    if(!is.null(shape)) {
      bg             <- erase(bg, shape)
      mask           <- union(bg, shape)
      mask$species_ID.1[!is.na(mask$species_ID.2)] <- mask$species_ID.2[!is.na(mask$species_ID.2)]
      mask@data      <- data.frame(species_ID = mask@data[,1])
    } else {
      mask           <- bg
    }
    # plot(mask, col = mask$species_ID+7)
    
    
    #### segment images + masks ####
    message(paste0("segmenting images ", j, "/", length(sites)))
    
    ## define kernel size
    kernelSizeX <- floor(tilesize / xres(ortho))
    kernelSizeY <- floor(tilesize / yres(ortho))
    
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
    lines(grdVec, col = "blue")
    

    cl <- makeCluster(19)
    registerDoParallel(cl)
    
    XY <- foreach(i = 1:nrow(idxGrd), .packages = c("raster", "rgdal", "keras", "magick"), .inorder = T) %dopar% {
      
      xmin <- idxGrd[i,2]
      xmax <- idxGrd[i,2]+kernelSizeX-1
      ymin <- idxGrd[i,1]
      ymax <- idxGrd[i,1]+kernelSizeY-1
      cropExtent <- extent(ortho, xmin, xmax, ymin, ymax)
      
      
      ## crop images and calc percentage cover of endmember
      orthoCrop <- crop(ortho, cropExtent)
      # plotRGB(orthoCrop)

      polyCrop <- crop(mask, cropExtent)
      if(length(polyCrop) > 0) { # rasterize shapefile if polygons exist 
        polyCropR  <- rasterize(polyCrop, orthoCrop[[1]], field = polyCrop$species_ID)
        
        NAidx      <- which(is.na(values(polyCropR)))
        flagPolyNA <- length(NAidx) < 2500 # TRUE if NAValues exist AND no less then 2500 (50*50 pixel = 1m2) in crop
        flagOrtho  <- length(which(is.na(values(orthoCrop[[1]])) == TRUE))/length(orthoCrop[[1]]) < 0.05 # TRUE if less then 5% NA in crop
      } else {
        flagPolyNA <- flagOrtho <- FALSE
      }
      
      
      if(flagOrtho && flagPolyNA) {
        # fill NA values
        if(length(NAidx) > 0) {
          rows <- rowFromCell(polyCropR, NAidx)
          cols <- colFromCell(polyCropR, NAidx)
          
          left <- cols-floor(40/2); left[left < 1] = 1
          top  <- rows-floor(40/2); top[top < 1] = 1
          for(k in 1:length(NAidx)) {
            vals                <- getValuesBlock(polyCropR, row = top[k], nrow = 50, col = left[k], ncol = 50)
            polyCropR[NAidx[k]] <- as.numeric(names(table(vals)[1]))
          } 
        }
        
        extent(orthoCrop) <- extent(0, kernelSizeX, 0, kernelSizeY)
        extent(polyCropR) <- extent(0, kernelSizeX, 0, kernelSizeY)
        
        orthoCrop <- as.array(orthoCrop)
        polyCropR <- as.array(polyCropR)
        orthoCrop <- image_read(orthoCrop / 255)
        polyCropR <- image_read(polyCropR / 255)
        
        filename  <- paste(site, tileTag, paste0("b", nlayers(ortho)), sep = "_")
        image_write(orthoCrop, format = "png",
                    path = paste0(imgDir, "img", sprintf("%03d", i), "_", filename, ".png"))
        image_write(polyCropR, format = "png",
                    path = paste0(mskDir, "msk", sprintf("%03d", i), "_", filename, ".png"))
        
        cropExtent[1:4]
      } else {
        FALSE
      }
    }
    stopCluster(cl)
    
    # export xy positions to a text file
    XYpos           <- cbind(1:length(XY), do.call(rbind, XY))
    colnames(XYpos) <- c("ID", "xmin", "xmax", "ymin", "ymax")
    XYpos <- XYpos[lapply(XY, sum) != 0, ]
    write.csv(XYpos, file = paste0(imgDir, "/metadataXYpos.csv"), row.names = F)
    
    removeTmpFiles(h=0)
    gc()
  }
}

segmentImages(files, tilesize = 10.24, plot = T, overwrite = T)


