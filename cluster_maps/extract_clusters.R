args = commandArgs(trailingOnly=TRUE)

file_name = args[1]
fold_name = args[2]

case_name = strsplit(file_name, '1_current')[[1]][1]
out_file  = paste('cluster_csv/',fold_name,'/',case_name, '.clusters.csv', sep = '')
inp_file  = paste('rdata/', fold_name, '/', file_name, sep = '')

print(inp_file)
load(inp_file)
if (!exists('cl_ap')) {
   err_msg = paste("ERROR: Clusters do not exist for: ",case_name,sep='')
   print(err_msg)
} else if (exists('tmp2')) {
 	tmp <- as.data.frame(tmp2)
	colnames(tmp)<- c('y', 'x')
	tmp$cluster_id <- cl_ap
	print(out_file)
    write.csv(tmp,file=out_file,row.names=FALSE)
} else if (exists("tmp1")) {
    print("EXISTS")
 	tmp <- as.data.frame(tmp1)
	colnames(tmp)<- c('y', 'x')
	tmp$cluster_id <- cl_ap
	print(out_file)
    write.csv(tmp,file=out_file,row.names=FALSE)
}
