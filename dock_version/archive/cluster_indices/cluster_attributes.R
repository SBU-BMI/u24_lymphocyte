files <- list.files(path="./out_indices_new_22Sep17",
    pattern="*cluster_info.rda", full.names=T, recursive=T)

tabmat <- matrix(NA, nrow=length(files), ncol=9)
for (i in 1:length(files)) {
    fname = files[i]
    load(file = fname)

    NP_list = vector()
    WCD_list = vector()
    CE_list = vector()
    list_i <- 1
    for (name in names(cl_info)) {
        NP_list[list_i] <- cl_info[[name]][["Num_Points"]]
        WCD_list[list_i] <- cl_info[[name]][["WC_dispersion"]]
        CE_list[list_i] <- cl_info[[name]][["cluster_extent"]]
        list_i <- list_i + 1
    }

    fields <- strsplit(fname, "/|_", fixed=FALSE, perl=FALSE, useBytes=FALSE)
    tabmat[i,] = c(fields[[1]][6], fields[[1]][7], list_i-1, mean(NP_list), sd(NP_list), 
        mean(WCD_list), sd(WCD_list), mean(CE_list), sd(CE_list))
}

colnames(tabmat) <- c("cancer_type", "subject_id",
    "N_cluster", "NP_mean", "NP_sd", "WCD_mean", "WCD_sd", "CE_mean", "CE_sd")
write.table(as.table(tabmat), "cluster_attrib.csv", sep=",", row.names=FALSE, quote=F)

