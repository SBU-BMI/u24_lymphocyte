function low_res_all()

files = dir('./patch-level-merged/prediction-*');
for f = files'
    fn = f.name;
    low_res(['./patch-level-merged/', fn]);
end

