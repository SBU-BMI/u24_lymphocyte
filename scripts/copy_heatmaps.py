import os
import sys

cancertype = sys.argv[1]
out_dir = sys.argv[2]
option = int(sys.argv[3])

def copy_heatmap():
    if option == 1:     # to copy heatmaps
        grayscale = '../data/grayscale_heatmaps'
        thresholded = '../data/thresholded_heatmaps'

        #rates-<cancer_type>-all-auto
        dest = '../data/rates-{}-all-auto'.format(cancertype)
        if not os.path.isdir(dest):
            os.system('mkdir ' + dest)
        else:
            os.system('rm -rf ' + dest + '/*')

        files = os.listdir(grayscale)
        print(files)
        for file in files:
            if '.png' in file:
                file_dest1 = 'rate.' + file[:-4] + '.automatic.png'
                file_dest2 = file_dest1[:-4] + '_thres.png'
                os.system('cp ' + os.path.join(grayscale, file) + ' ' + os.path.join(dest, file_dest1))
                os.system('cp ' + os.path.join(thresholded, file) + ' ' + os.path.join(dest, file_dest2))
        os.system('cp ' + os.path.join(thresholded, '*.csv') + ' ' + dest)
    elif option == 2:
        # create input_full.csv
        filename = os.path.join(out_dir, 'input_full.csv');
        files = os.listdir(os.path.join(os.path.join(out_dir,"inputs"), cancertype))
        print('filename: ', filename)

        os.system('rm -rf ' + filename)
        output = os.path.join(out_dir, 'output')
        if os.path.isdir(output):
            os.system('rm -rf ' + output + '/*')
        else:
            os.system(output)
        os.system('mkdir ' + os.path.join(output, cancertype))
        inputs = os.path.join(out_dir, 'inputs')
        for file in files:
            if '.csv' in file:
                with open(filename, 'a') as f:
                    f.write('inputs/' + cancertype + '/' + file + ',' + './output/' + cancertype + '/,' + cancertype + '\n')



if __name__ == "__main__":
    print('Usage: python copy_heatmap.py [cancertype] [out_dir] [func_option] \nfunc_option: 1 for copy the heatmap, 2 for create file input_full.csv')
    if len(sys.argv) < 3:
        print('missing arguments')
    else:
        copy_heatmap()
