import json
from bson import json_util
import os.path
import sys

list_file = sys.argv[1];
meta_dir = './';

with open(list_file) as f_in:
    content = f_in.readlines();

for line in content:
    if (line[-1] == '\n'):
        line = line[:-1];

    path, file = os.path.split(line);
    case_id = file[:23];
    subject_id = file[:12];

    dict_humanmark = {};
    dict_humanmark['title'] = 'humanmark';

    dict_provenance = {};
    dict_provenance['analysis_execution_id'] = 'humanmark';
    dict_provenance['type'] = 'human';
    dict_humanmark['provenance'] = dict_provenance;

    dict_image = {};
    dict_image['case_id'] = case_id;
    dict_image['subject_id'] = subject_id;
    dict_humanmark['image'] = dict_image;

    meta_file = os.path.join(meta_dir, 'meta_' + case_id + '.json');
    with open(meta_file, 'w') as f:
        json.dump(dict_humanmark, f, default=json_util.default);

