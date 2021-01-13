import os
import json

with open(os.path.join(os.getcwd(),'manifest_files','target.json'), 'w') as outfile:
    meta = {'This':'0', 'Is':1, 'Test':2, 'Data':"is this rewriting the file?"}
    json.dump(meta, outfile)
    outfile.write("\n")
