import csv
import sys
import json
import os
import glob

def get_file_list(folder):
    metafiles = []
    print (folder)
    fnames = folder + "/*.csv"
    for name in glob.glob(fnames):
       task = {}
       task["folder"] = folder
       task["name"]   = os.path.basename(name)
       print (task)
       metafiles.append(task)
    return metafiles

def process_file(fname,ffolder,img_json):
    caseid = fname.split('.')[0]
    c_type = ffolder.split('/')[1]
    name = ffolder + "/" + fname
    print (c_type,name,caseid)
    
    out_folder = "output/"+str(c_type)
    if os.path.isdir(out_folder)==False:
       os.mkdir(out_folder)

    one_json = ''
    for i in range(len(img_json)):
        if (img_json[i]["case_id"]==caseid):
           one_json = img_json[i]
           break;
    one_json["file-location"] = one_json["filename"]
    one_json["mpp-x"] = float(one_json["mpp_y"])
    one_json["mpp-y"] = float(one_json["mpp_y"])

    # open file containing all patches to compute xlen and ylen
    allname   = './inputs/' + c_type + '/' + caseid + '.csv'
    csvall    = open(allname)
    allreader = csv.reader(csvall) 
    # find x and y length of each patch
    xmax = 0
    ymax = 0
    for row in allreader:
       if (int(row[0])>ymax):
          ymax = int(row[0])
       if (int(row[1])>xmax):
          xmax = int(row[1])
    xlen = int(float(one_json["width"])/float(xmax)+0.5)
    ylen = int(float(one_json["height"])/float(ymax)+0.5)

    # open file and find the number of clusters
    csvfile   = open(name)
    csvreader = csv.reader(csvfile)
    headers   = csvreader.next()
    c_set = set()
    c_lst = {}
    f_lst = {}
    i     = 0
    header = "AreaInPixels,ClusterId,Polygon\n" 
    for row in csvreader:
        if row[2] not in c_set:
           c_lst[str(row[2])] = i+1

           metafile = "output/" + str(c_type) + "/out_" + str(i+1) + "_" + caseid + "-algmeta.json"
           f = open(metafile,"w") 
           mdata = {}
           mdata["input_type"] = "wsi"
           mdata["mpp"] = float(one_json["mpp_x"])
           mdata["case_id"] = caseid
           mdata["subject_id"] = one_json["subject_id"] 
           mdata["out_file_prefix"] = "out_" + str(i+1) + "_" + caseid 
           mdata["image_width"] = int(one_json["width"])
           mdata["image_height"] = int(one_json["height"])
           mdata["analysis_id"] = "til-cluster-" + str(i+1)
           mdata["analysis_desc"] = "til-cluster-" + str(i+1)
           json.dump(mdata,f)
           f.close()

           outfile  = "output/" + str(c_type) + "/out_" + str(i+1) + "_" + caseid + "-features.csv" 
           f_lst[str(row[2])] = open(outfile,"w")
           f_lst[str(row[2])].write(header)
           c_set.add(row[2])
           i = i + 1
 
    # reopen the file to reset csvreader
    csvfile   = open(name)
    csvreader = csv.reader(csvfile)
    headers   = csvreader.next()
    area = 2000*2000
    for row in csvreader:
        y = int(row[0])-1
        x = int(row[1])-1
        p = int(row[2])
        x0 = x*xlen
        y0 = y*ylen
        x1 = x0+xlen
        y1 = y0
        x2 = x0+xlen
        y2 = y0+ylen
        x3 = x0
        y3 = y0+ylen
        poly = "[" 
        poly = poly + str(x0) + ":"
        poly = poly + str(y0) + ":"
        poly = poly + str(x1) + ":"
        poly = poly + str(y1) + ":"
        poly = poly + str(x2) + ":"
        poly = poly + str(y2) + ":"
        poly = poly + str(x3) + ":"
        poly = poly + str(y3) + "]"
        outstr = str(area)+","+str(p)+","+poly+"\n"
        f_lst[str(row[2])].write(outstr)

if __name__ == "__main__":
   csv.field_size_limit(sys.maxsize)
   
   mfiles = get_file_list(sys.argv[1])
   f = open("img_list.json","r")
   img_json = json.load(f)
   for mfile in mfiles:
       process_file(mfile["name"],mfile["folder"],img_json)

