# initialize
set.seed(1024)
rm(list = ls(all = T))

## Parse input arguments
args = commandArgs(trailingOnly=TRUE)

if (length(args)!=4) {
   stop("Required arguments: <workdir> <input file> <output folder> <study>.\n", call.=FALSE)
}

workdir <- args[1]
inpfile <- args[2]
wpath <- args[3]
study <- tolower(args[4])

setwd(workdir) # Set the working directory
# install packages if not installed
#install.packages(c("clusterCrit", "mclust", "NbClust", "apcluster")) 
library(clusterCrit)
library(mclust)
library(NbClust)
library(apcluster)

# Set lower and upper limit on data points
# ORIGINAL: min_num_datapts <- 10 # has to be greater than or equal to 2
min_num_datapts <- 3 # has to be greater than or equal to 2
# ORIGINAL: max_num_datapts <- 30000
max_num_datapts <- 150000 
save_images <- FALSE

 # this function calculates the details of the clusters
calculate_cluster_details <- function(x1, apres){
   N = length(apres@exemplars)
   cluster_details = list()
   for (j in 1:N) {
     A = as.matrix(x1[apres@clusters[[j]],])
     exemplar <- matrix(unlist(x1[apres@exemplars[[j]],]), nrow = 1)
     X =  A - rep(exemplar, rep.int(nrow(A), ncol(A)))

     # cluster index calculation
     cl_extent <- max(apply(X, 1, function(x){sqrt(sum(x^2))}))

     mu =  colMeans(A, na.rm = FALSE, dims = 1)
     cent_X =  A - rep(mu, rep.int(nrow(A), ncol(A)))
     M = t(cent_X)%*%cent_X
     SS = sum(diag(M))

     name <- paste('Cluster ', j, sep='')
     cluster_details[[name]] = list(Num_Points =nrow(A), WC_dispersion = SS,
                                    Scatter_Matrix = M, cluster_extent = cl_extent)
   }
   return(cluster_details)
}
  
criteriaNames <- getCriteriaNames(TRUE)
intCindex=matrix(NA,1, length(criteriaNames) + 1)
# get the rownames as filenames and colnames as the criteria names
colnames(intCindex) <- c("Slides", criteriaNames)
intCindex_ap <- intCindex
  
datalen <- matrix(NA,1, 3)
colnames(datalen) <- c('Slides','number of data points',  'number of clusters')

# processing one file
indx <- 1 
  
gc()
dat= read.csv(inpfile, header=T)
idx = which(dat[,3] == 1) ##binary presence of til
tmp1 = dat[idx,1:2]
rm(dat)
    
##############################
csvname <- basename(inpfile)
name <- substr(csvname, 1,nchar(csvname)-4)
wfname <- paste(wpath, name, '_heatmap.jpg', sep = "" )
wfname_dt <- paste(wpath, name, '_cluster_info.rda', sep = "" )
wfname_ap <- paste(wpath, name, '_clusters_ap.jpg', sep = "" )
datalen[indx, 1:2] <- c(name, dim(tmp1)[1])
intCindex_ap[indx, 1] <- csvname
###############################

cat('Processing: ',inpfile,' name:',name,' wfname_dt: ',wfname_dt,'\n')
    
if (dim(tmp1)[1] > min_num_datapts & dim(tmp1)[1] < max_num_datapts) {
   ptm <- proc.time()
   apclus = apcluster(negDistMat(r=2), tmp1, q=0) ##can change the similarity matrix rbf etc..
   # get the clusters from apclus
   cl_ap <- as.integer(apclus@idx)
   # collect info abour the number of clusters
   datalen[indx, 3] <-length(apclus@exemplars)
   #save(tmp1, cl_ap, file = wfname_dt)
      
   if (length(apclus@exemplars) > 1) {
      if (save_images) {
         # # writing heatmap and cluster plot to jpg
         #jpeg(wfname,width = 1000, height = 1000)
         #heatmap(apclus)
         #dev.off()
         jpeg(wfname_ap,width = 1000, height = 1000)
         plot(apclus, tmp1)
         dev.off()
      }
      cl_info <- calculate_cluster_details(tmp1, apclus)

      save(tmp1, cl_info, file = wfname_dt)
      ##show(apclus)
      rm(apclus)
      tmp2 = apply(tmp1,2,as.numeric)
      rm(tmp1)
      # Compute all the internal indices
      intCindex_ap[indx,]=c(csvname, unlist(intCriteria(as.matrix(tmp2), cl_ap, "all")))
      gc()
   }
   else{cat('this file has only one cluster')}
      
   cat('time taken to process this file: ', (proc.time() - ptm), '\n')
}
    
save.image(file = paste(wpath,name, indx, '_currentVersion.RData', sep = ""))
    
resultpath <- paste(wpath, name, '_indices_ap.csv', sep = "" )
clusterInfo <- paste(wpath, name, '_clusterInfo.csv', sep = "" )
write.table(intCindex_ap, resultpath, sep = ",", row.names = FALSE)
write.table(datalen, clusterInfo, sep = ",", row.names = FALSE)
