import os
import csv

ext = 'clusterInfo.csv';
#ext = 'indices_ap.csv';

#root = './out_indices_8Sep17_12Sep17-18-00';
root = './out_indices_new_22Sep17';
#root = './out_indices_8Sep17_sample_1K';
info_root = 'inputs_full_new_22Sep17';
outdir = './detail_collect';
detail_file = 'cluster_attrib.csv';
TILClass_file = 'TIL-Pattern-Labels.csv';

eli_slides = {};
eli_slides['TCGA-RZ-AB0B-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V3-A9ZX-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V3-A9ZY-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V4-A9E5-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V4-A9E7-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V4-A9E8-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V4-A9E9-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V4-A9EA-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V4-A9EC-01Z-00-DX1'] = 'skcm';
eli_slides['TCGA-V4-A9ED-01Z-00-DX1'] = 'skcm';

# Read TIL classification
dict_TILClass = {};
with open(TILClass_file, 'r') as f:
    reader = csv.reader(f);
    ridx = 0;
    for row in reader:
        if ridx == 0:
            TILClass_len = len(row) - 1;
            header_TILClass = ',"' + '","'.join(row[1:]) + '"';
        else:
            dict_TILClass[row[0]] = ',"' + '","'.join(row[1:]) + '"';
        ridx += 1;

# Read additional detail
dict_detail = {};
with open(detail_file, 'r') as f:
    reader = csv.reader(f);
    ridx = 0;
    for row in reader:
        if ridx == 0:
            detail_len = len(row) - 2;
            header_detail = ',"' + '","'.join(row[2:]) + '"';
        else:
            dict_detail[row[1]] = ',"' + '","'.join(row[2:]) + '"';
        ridx += 1;



subdirs = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))];
print subdirs;

sup_idx = 0;
sup_out_file = outdir + '/all_' + ext;
sup_out_fid = open(sup_out_file, 'w');
n_no_til = 0;
for subd in subdirs:
    subpath = root + '/' +  subd;

    outfile = outdir + '/' + subd + '_' + ext;
    fwrite = open(outfile, 'w');
    idx = 0;

    for file in os.listdir(subpath):
        if file.endswith(ext):
            filepath = subpath + '/' + file;
            with open(filepath) as f:
                lines = f.readlines();
            lines = [x.strip() for x in lines]

            if (sup_idx == 0):
                sup_header = lines[0] + ',"til_percentage"' + header_detail + header_TILClass;
                sup_out_fid.write(sup_header + '\n');
                sup_idx = sup_idx + 1;

            if (idx == 0):
                file_header = lines[0] + ',"til_percentage"' + header_detail + header_TILClass;
                fwrite.write(file_header + '\n');
                idx = idx + 1;

            ln = lines[1];
            ln = ln.replace('.csv', '');

            second_quote_idx = ln.find('"', 1);
            slide_id = ln[1:second_quote_idx];
            info_file = info_root + '/' + subd + '/' + slide_id + '.info';
            info_field = {};
            info_field['til_number'] = 0;
            info_field['tissue_number'] = 0;
            with open(info_file, 'rb') as f_info:
                reader = csv.reader(f_info);
                for row in reader:
                    info_field[row[0]] = float(row[1]);

            til_percentage = "NA";
            if (info_field['tissue_number'] != 0):
                til_percentage = str(100 * info_field['til_number'] / info_field['tissue_number']);
            else:
                print "no til slide: ", subd, ", ", slide_id;
                n_no_til = n_no_til + 1;
            #ln = ln + ',"' + til_percentage + '"';
            detail_written = '';
            for idetail in range(detail_len):
                detail_written = detail_written + ',"NA"';
            if slide_id in dict_detail:
                detail_written = dict_detail[slide_id];

            TILClass_written = '';
            for iTILClass in range(TILClass_len):
                TILClass_written = TILClass_written + ',"NA"';
            if slide_id in dict_TILClass:
                TILClass_written = dict_TILClass[slide_id];

            #ln = ln + ',"' + til_percentage + '"' + detail_written;
            ln = ln + ',"' + til_percentage + '"' + detail_written + TILClass_written;


            fwrite.write(ln + '\n');
            sup_out_fid.write(ln + '\n');

    fwrite.close();

sup_out_fid.close();

print "no til: ", n_no_til;
