import glob


folder = '/data03/shared/lehhou/lym_project/svs_tiles/TCGA-38-4629-01Z-00-DX1.d00cc280-5370-4b9f-9655-6c35deb94647.svs/';

outfile = folder + 'list.txt';
filevar = open(outfile, 'w');

filelist = [];
for filename in glob.glob(folder + '/*.png'):
    name = filename[len(folder):-4];
    filevar.write("%s\n" % name);

filevar.close();
