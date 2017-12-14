###############################
set.seed(1024)
rm(list = ls(all = T))
setwd('Y:/Desktop/StonyBrookData/') # Set the working directory
# install packages if not installed
#install.packages(c("clusterCrit", "mclust", "NbClust", "apcluster")) 
library(clusterCrit)
library(mclust)
library(NbClust)
library(apcluster)

# list of all the directories containing Raw data/csv files
filepaths <- dir(path = './RawData/') 
# Set lower and upper limit on data points
min_num_datapts <- 10 # has to be greater than or equal to 2
max_num_datapts <- 30000
save_images <- FALSE
# set the study name, "SKCM", "LUAD", etc.
study <- "PAAD"
fnum <- which(filepaths ==tolower(study))

today <- function(){day <- format(Sys.Date(), "%d %b %Y")
return(gsub(" ", "_", day))}

for (filename in filepaths[fnum]){
  filepath <- paste('./RawData/',filename, '/', filename, '/' ,sep = "" )
  wpath <- paste('./Results_priority/', tolower(study),'_', 
                 today(),  '_indices_and_images/', sep = "")
  
  if (!dir.exists(wpath)){dir.create(wpath)}
  
  path = list.files(filepath)
  # starting and ending number of files to be processed
  starts <- 1
  ends <- length(path)
  
  criteriaNames <- getCriteriaNames(TRUE)
  intCindex=matrix(NA,length(path), length(criteriaNames) + 1)
  # get the rownames as filenames and colnames as the criteria names
  colnames(intCindex) <- c("Slides", criteriaNames)
  # rownames(intCindex) <- path
  intCindex_ap <- intCindex
  
  datalen <- matrix(NA,length(path), 3)
  # rownames(datalen) <- path
  colnames(datalen) <- c('Slides','number of data points',  'number of clusters')
  path = list.files(path= filepath, full.names = TRUE)
  
  for (indx in starts:ends){
    cat("file number is ", indx, '\n')
    gc()
    dat= read.csv(path[indx], header=T)
    idx = which(dat[,3] == 1)##binary presence of til
    tmp1 = dat[idx,1:2]
    rm(dat)
    
    ##############################
    csvname <- basename(path[indx])
    name <- substr(csvname, 1,nchar(csvname)-4)
    wfname <- paste(wpath, name, '_heatmap.jpg', sep = "" )
    wfname_dt <- paste(wpath, name, '_cluster_info.rda', sep = "" )
    wfname_ap <- paste(wpath, name, '_clusters_ap.jpg', sep = "" )
    datalen[indx, 1:2] <- c(name, dim(tmp1)[1])
    intCindex_ap[indx, 1] <- csvname
    ###############################
    
    if (dim(tmp1)[1] > min_num_datapts & dim(tmp1)[1] < max_num_datapts){
      ptm <- proc.time()
      ##sim <- negDistMat(tmp1, r=2)
      apclus = apcluster(negDistMat(r=2), tmp1, q=0)##can change the similarity matrix rbf etc..
      # get the clusters from apclus
      cl_ap <- as.integer(apclus@idx)
      # collect info abour the number of clusters
      datalen[indx, 3] <-length(apclus@exemplars)
      save(tmp1, cl_ap, file = wfname_dt)
      
      if (length(apclus@exemplars) > 1){
        if (save_images) {
          # writing heatmap and cluster plot to jpg
          jpeg(wfname,width = 1000, height = 1000)
          heatmap(apclus)
          dev.off()
          jpeg(wfname_ap,width = 1000, height = 1000)
          plot(apclus, tmp1)
          dev.off()
        }
        ##show(apclus)
        rm(apclus)
        tmp2 = apply(tmp1,2,as.numeric)
        rm(tmp1)
        # Compute all the internal indices
        intCindex_ap[indx,]=c(csvname, unlist(intCriteria(as.matrix(tmp2), cl_ap,"all")))
        ##http://www.sthda.com/english/wiki/determining-the-optimal-number-of-clusters-3-must-known-methods-unsupervised-machine-learning#compute-all-the-30-indices
        gc()
      }
      
      else{cat('this file has only one cluster')}
      
      cat('time taken to process this file: ', (proc.time() - ptm), '\n')
    }
    
    if (indx %% 10 ==0 | indx == ends) {
      save.image(file = paste(wpath,filename, indx, '_currentVersion.RData', sep = ""))
    }
    
  }
  #http://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters/36729465
  resultpath <- paste(wpath, filename, '_indices_', starts, '_to_', ends,'_ap.csv', sep = "" )
  clusterInfo <- paste(wpath, filename, '_clusterInfo_', starts, '_to_', ends,'.csv', sep = "" )
  ###############################
  write.table(intCindex_ap, resultpath, sep = ",", row.names = FALSE)
  write.table(datalen, clusterInfo, sep = ",", row.names = FALSE)
}