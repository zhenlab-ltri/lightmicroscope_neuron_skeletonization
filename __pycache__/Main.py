import logging
import os
from ImageJ_processing import ImageJ_processing
from Skeletonization_processing import Skeletonization_processing
from Manual_processing import Manual_processing

root_folderpath = '/home/zhenlab/shuyu/PyQt_related/LMS_Pipeline/IMAGES/RIH3/04152022'
num_slices_to_remove = 5 #number of slices to remove from start of image stack

skipfirststeps = True

threshold1 = {
    'blur_sigma': [2, 2, 2],  # x, y, z
    'sb_radius': 15, #sliding paraboloid radius
    'rawthresh': 2, #thresholding method 1 is an abs global threshold
    'minsize': 250, #all objects less than this pixel area will be removed after thresholding
}

threshold2 = {
    'blur_sigma': [2, 2, 2],  # x, y, z
    'sb_radius': 15, #sliding paraboloid radius
    'threshtype': 'Triangle', #thresholding method for 2
    'minsize': 250, #all objects less than this pixel area will be removed after thresholding
}

threshold3 = {
    'blur_sigma': [2, 2, 2],  # x, y, z
    'sb_radius': 15, #sliding paraboloid radius
    'threshtype': 'Default', #thresholding method for 2
    'minsize': 250, #all objects less than this pixel area will be removed after thresholding
}

remove_nuclei = False
remove_nuclei_parameters = { #if remove_nuclei is set to False, it doesn't matter what these parameters are set to
    'minArea' : 5000,
    'maxArea' : 20000,
    'minCircularity' : 0.25,
    'maxCircularity' : 1,
    'minConvexity' : 0.2,
    'maxConvexity' : 1,
    'minInertia' : 0.1,
    'maxInertia' : 1
}

#configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if skipfirststeps == False:
    '''
    Section 1: Run image processing using ImageJ 
    - preprocessing, i.e. intensity normalization, conversion to 8-bit data
    - 3d gaussian blur with sigma defined above
    - subtract background function using sliding paraboloid with radius defined above
    - thresholding using method and value defined above
    - removal of all small objects lower than minimum size defined above
    - save processed and thresholded image in new /Processed folder as .tif
    '''
    processed_folderpath = ImageJ_processing(
        folderpath=root_folderpath,
        threshold1=threshold1,
        threshold2=threshold2,
        threshold3=threshold3,
        num_slices_to_remove=num_slices_to_remove,
        batchsize=5
    )
    logging.info("Done Section 1: ImageJ processing on all images")
    # shortcut (if trying to skip this section)

    '''
    Section 2: Skeletonize processed images
    - run optional cv2 Blob Detection with parameters to remove nuclei (which in some cases may be more circular than other 'branch' blobs)
    - skeletonize thresholded images using Ske`letonize3D from scikit-image
    - save resulting skeleton as a point cloud in .npy file
    '''
    coords_folderpath = Skeletonization_processing(
        folderpath=processed_folderpath,
        remove_nuclei=remove_nuclei,
        remove_nuclei_parameters=remove_nuclei_parameters)
    logging.info("Done Section 2: Skeletonization of all processed images")
    # shortcut (if trying to skip this section)
else:
    processed_folderpath = os.path.join(root_folderpath, "Processed")
    coords_folderpath = os.path.join(processed_folderpath, "Coords")

'''
Section 3: Manual processing
- first GUI lets you choose root, start, end nodes, and remove points
- second GUI lets you correct processed branching
- third GUI lets you make measurements
'''
Manual_processing(folderpath = coords_folderpath)