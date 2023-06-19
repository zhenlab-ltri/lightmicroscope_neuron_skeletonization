# lightmicroscope_neuron_skeletonization
 Automated processing of light microscope image stacks (.tif) using ImageJ, 3D skeletonization of processed stack, and efficient GUIs for neuron branching analysis. Saves results as excel file.

### GUI Features:
#### pre-branching
- removing noise clutter
- connecting broken branches
- setting root,start,and end nodes of neuron
- setting threshold for branch length and number of nodes
- setting scale of x,y,z pixels
#### post-branching
- removing branches
- naming branches
- measuring euclidean distance between points
- measuring angle between points
- measuring branch lengths by fitting

## Instructions for use
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Edit Main.py to point to correct file directory with .tif image stacks
```bash
root_folderpath = 'example\folder\path'
```
3. Edit thresholding, nuclei removal, and slice removal settings in Main.py as desired

4. Run Main.py
```bash
python Main.py
```
