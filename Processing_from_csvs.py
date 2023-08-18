import numpy as np
import pandas as pd
import os
from Manual_processing import Manual_processing

folderpath = 'path here'

savefolder = os.path.join(folderpath, "NPYs")
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

files = []
filenames = []
for filename in sorted(os.listdir(folderpath)):
    if os.path.isfile(os.path.join(folderpath, filename)) and filename.endswith('.csv'):
        files.append(os.path.join(folderpath, filename))
        filenames.append(filename[:-4]) #remove .csv extension from filename
numfiles = len(files)

for i in range(numfiles):
    df = pd.read_csv(os.path.join(folderpath, files[i]))
    coords = np.stack((df['X'], df['Y'],df['Z'])) #csv columns must be labelled 'X' 'Y' and 'Z'
    savepath = os.path.join(savefolder, filenames[i])
    np.save(f"{savepath}.npy", coords.T)
  
branching_parameters = {
    'remove_overlap' : 0, #minimum distance between nodes to reduce clutter (adjacent nodes under threshold will be removed)
    'num_nearestneighbours' : 3, #number of k-nearest neighbours to consider for branching
    'min_unconnected_nodes' : 5, #minimum number of nodes for an unconnected cluster to be considered important enough to attach to the main branch
    'max_unconnected_distance' : 50 #maximum distance from main branch for an unconnected cluster to be considered important enough to be added to the main branch
}

'''
Section 3: Manual processing
- first GUI lets you choose root, start, end nodes, and remove points
- branching is run using data from first GUI
- second GUI lets you correct processed branching
- third GUI lets you make measurements
'''
Manual_processing(folderpath = savefolder, branching_parameters = branching_parameters)