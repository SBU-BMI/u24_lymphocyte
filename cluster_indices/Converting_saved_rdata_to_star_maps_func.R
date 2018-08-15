#
# This code could be used to generate star maps if we have generated
# the cluster indices with clust_index_single_with_details.R.
# Code developed by Pankaj Singh
#
library(ggplot2)
library(grid)
# set the theme for the plots in ggplot2
theme_set(theme_bw())


get_star_map <- function(rda_file_name, RData_file_name, png_file_name){
    # load them to the workspace
    load(rda_file_name)
    load(RData_file_name)
    # create a data frame from the tmp2 variable in the workspace
    tmp <- as.data.frame(tmp2)
    # change colnames
    colnames(tmp)<- c('X1', 'Y1')
    # add the cluster membership to each point
    tmp$cluster_id <- cl_ap
    # create two new columns
    tmp$X2 <- NA
    tmp$Y2 <- NA
    # get the exemplars from the cluster memberships
    exemplars <- unique(cl_ap)
    
    for (exemp in exemplars) {
      # get rows for each exemplar
      rows <- which(tmp$cluster_id == exemp)
      # change the NAs to the coordinates of the exemplars
      tmp$X2[rows] <- tmp$X1[exemp]
      tmp$Y2[rows] <- tmp$Y1[exemp]
    }
    # convert the cluster_ids to factors
    tmp$cluster_id <- as.factor(tmp$cluster_id)
    # get the ggplot object containing the star map
    p <- ggplot(tmp, aes(x=X1, y=Y1, color=cluster_id)) + geom_point(size = 1)+
      geom_segment(aes(x = X1, y = Y1, xend = X2, yend = Y2), data = tmp) + 
      theme(legend.position="none") + 
      theme(axis.text = element_blank(),
            axis.ticks = element_blank(), axis.title = element_blank())
    # save the rotated (by 90 degrees) plot as png files
    
    png(filename = png_file_name,
        width = 1000, height = 1000)
    
    grid.newpage() 
    pushViewport(viewport(angle=-90)) 
    grid.draw(ggplot_gtable(ggplot_build(p)))
    dev.off()
    
}
#### EXAMPLE #####
rda_file_name = "TCGA-33-AASL-01Z-00-DX1_cluster_info.rda"
RData_file_name = "TCGA-33-AASL-01Z-00-DX11_currentVersion.RData"
png_file_name = "TCGA-33-AASL-01Z-00-DX1_Star_map.png"
get_star_map(rda_file_name, RData_file_name, png_file_name)




