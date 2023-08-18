import os
from skimage import io
import numpy as np
from skimage.morphology import skeletonize_3d
import logging
import cv2
from tqdm import tqdm

def Skeletonization_processing(folderpath: str,
                              remove_nuclei: bool = False,
                              remove_nuclei_parameters: dict = {}
                              ) -> str:
    """
    Perform skeletonization processing on a folder of 3D image stacks.

    Parameters
    ----------
    folderpath : str
        The path to the folder containing the 3D image stacks to be processed.

    remove_nuclei : bool, optional
        Flag indicating whether nuclei removal should be performed on the image stacks.
        If True, nuclei removal will be applied using the provided parameters.
        If False, nuclei removal will be skipped. Default is False.

    remove_nuclei_parameters : dict, optional
        Parameters for nuclei removal. Only relevant if `remove_nuclei` is True.
        The dictionary should contain the following keys:
            - 'minArea' : int, minimum area of nuclei
            - 'maxArea' : int, maximum area of nuclei
            - 'minCircularity' : float, minimum circularity of nuclei
            - 'maxCircularity' : float, maximum circularity of nuclei
            - 'minConvexity' : float, minimum convexity of nuclei
            - 'maxConvexity' : float, maximum convexity of nuclei
            - 'minInertia' : float, minimum inertia of nuclei
            - 'maxInertia' : float, maximum inertia of nuclei
        Default is an empty dictionary.

    Returns
    -------
    str
        The path to the folder where the skeletonized coordinate files are saved.

    Notes
    -----
    This function performs the following steps:
    1. Read all the .tif files in the specified folder.
    2. If `remove_nuclei` is True, perform nuclei removal on the image stacks using the provided parameters.
    3. Create a directory to store the skeletonized coordinate files.
    4. Iterate through each .tif file and perform skeletonization.
    5. Save the skeletonized coordinates as .npy files in the output directory.
    """
    
    #if remove_nuclei is true, log its parameters
    if remove_nuclei == True:
        logging.info(f'''Nuclei removal attemp will be run with parameters:
                    minArea: {remove_nuclei_parameters['minArea']}
                    maxArea: {remove_nuclei_parameters['maxArea']}
                    minCircularity: {remove_nuclei_parameters['minCircularity']}
                    maxCircularity: {remove_nuclei_parameters['maxCircularity']}
                    minConvexity: {remove_nuclei_parameters['minConvexity']}
                    maxConvexity: {remove_nuclei_parameters['maxConvexity']}
                    minInertia: {remove_nuclei_parameters['minInertia']}
                    maxInertia: {remove_nuclei_parameters['maxInertia']}
                    ''')

    
    #retrieve all .tif files in processed image folder
    files = []
    filenames = []
    for filename in sorted(os.listdir(folderpath)):
        if os.path.isfile(os.path.join(folderpath, filename)):
            files.append(os.path.join(folderpath,filename))
            filenames.append(filename[:-4]) #remove the .tif extension when adding to filenames
    numfiles = len(files)
    if numfiles > 0:
        logging.info(f"There are {numfiles} processed .tif files to be skeletonized")
    else:
        raise RuntimeError(f'No suitable .tif files found in {folderpath}, make sure ImageJ processing occured correctly')

    #create directory for skeletonized coordinate files
    savefolder = os.path.join(folderpath, "Coords")
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
        
    def nuclei_removal(img_arr: np.ndarray) -> np.ndarray:
        """
        This function is designed to detect and remove nuclei from a given image array. 
        It makes use of OpenCV's SimpleBlobDetector for nuclei detection.

        Parameters:
        -----------
        img_arr : np.ndarray
            The input 3D array representing the stack of 2D images 
            on which nuclei detection and removal is to be performed.

        Returns:
        --------
        new_img_arr : np.ndarray
            The output 3D array with nuclei removed.

        Notes:
        ------
        This function performs the following steps:
        1. Set up blob detection parameters.
        2. Create a SimpleBlobDetector with the defined parameters.
        3. Loop over each slice of the input 3D image array.
        4. Invert the slice to prepare for blob detection.
        5. Detect blobs (nuclei) in the inverted image using the detector.
        6. For each detected blob, perform a flood-fill operation to erase the blob from the image.
        7. Invert the image back to its original color space and store it in the output array.
        8. Return the output array with all detected nuclei removed.
        """

        from collections import deque
        
        params = cv2.SimpleBlobDetector_Params()
        
        #setup filtering parameters
        params.filterByArea = True
        params.minArea, params.maxArea = remove_nuclei_parameters['minArea'], remove_nuclei_parameters['maxArea']
        params.filterByCircularity = True
        params.minCircularity, params.maxCircularity = remove_nuclei_parameters['minCircularity'], remove_nuclei_parameters['maxCircularity']
        params.filterByConvexity = True
        params.minConvexity, params.maxConvexity = remove_nuclei_parameters['minConvexity'], remove_nuclei_parameters['maxConvexity']
        params.filterByInertia = True
        params.minInertiaRatio, params.maxInertiaRatio = remove_nuclei_parameters['minInertia'], remove_nuclei_parameters['maxInertia']
        
        #create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        def erase_blob(blob_coords: list, 
                       slice_arr: np.ndarray):
            """
            Erases a detected blob from a 2D image slice by performing a flood-fill operation.

            This function iterates over the image from the given blob coordinates, 
            replacing the color of the blob with the inverse of its original color until the entire blob is erased.

            Parameters
            ----------
            blob_coords : list
                The x and y coordinates where the blob is located. It is expected to be a list of two integers.

            slice_arr : np.ndarray
                The 2D image slice from which the blob is to be erased. 

            Returns
            -------
            slice_arr : np.ndarray
                The 2D image slice with the blob erased.

            Note
            ----
            The function uses a breadth-first search (BFS) algorithm for the flood-fill operation 
            which is implemented using a deque from Python's collections module. This ensures that the 
            function efficiently erases blobs of any size without exceeding Python's maximum recursion depth.
            """
            x,y = blob_coords
            target_color = slice_arr[y,x]
            replacement_color = 255 - target_color #i.e. the background color
            queue = deque([(x, y)])
            
            #floodfill
            while queue:
                x, y = queue.popleft()
                if slice_arr[y, x] == target_color:
                    slice_arr[y, x] = replacement_color
                    if x > 0: 
                        queue.append((x - 1, y))  # left neighbor
                    if x < slice_arr.shape[1] - 1: 
                        queue.append((x + 1, y))  # right neighbor
                    if y > 0: 
                        queue.append((x, y - 1))  # upper neighbor
                    if y < slice_arr.shape[0] - 1: 
                        queue.append((x, y + 1))  # lower neighbor
            return slice_arr 
        
        new_img_arr = np.zeros_like(img_arr)
        
        for i, slice in enumerate(img_arr):
            inverted_slice = cv2.bitwise_not(slice)
            
            #detect blobs
            keypoints = detector.detect(inverted_slice)
            
            #erase blobs
            num_blobs = 0
            for keypoint in keypoints:
                num_blobs +=1
                x,y = map(int, keypoint.pt)
                inverted_slice = erase_blob([x,y], inverted_slice)
            
            new_img_arr[i] = cv2.bitwise_not(inverted_slice)

        return new_img_arr
            
    def Skeletonize(img_arr: np.ndarray,
                    filename: str):
        """
        Skeletonize the 3D numpy image array and save the result as a .npy file
        
        Parameters
        ----------
        img_arr : np.ndarray
            The 3D image array to be skeletonized.
            
        filename : str
            The name of the output file where the coordinates of the skeletonized image will be saved.
        
        Returns:
        --------
        None
        
        Notes
        -----
        `skeletonize_3d` is an image thinning or skeletonization algorithm which works by peeling off 
        the outer layers of the object iteratively until only the "skeleton" of the object is left. 
        It uses a lookup table that defines the conditions under which the outer layer of an object 
        can be peeled off without changing the topology of the object. 

        In 3D, the algorithm works similarly but considers a 3x3x3 neighborhood for each voxel. It checks 
        each voxel in the image to see if it meets the conditions defined in the lookup table. If the 
        conditions are met, the voxel is set to 0, and the next layer of voxels is considered.

        The algorithm is applied iteratively until no more changes can be made (i.e., until the image 
        has been fully skeletonized). It works best on binary images.

        The output skeleton is a binary image that has the same dimensions as the input image.
        """

        thresh = np.max(img_arr)*0.99 #since the array is already threshold, we simply chose a value that is not the max
        thresholded_arr = np.zeros_like(img_arr)
        
        for z in range(img_arr.shape[0]): #the number of slices
            thresholded_arr[z] = np.where(img_arr[z] > thresh, 1, 0)
        thresholded_arr = thresholded_arr.astype(np.uint8)
        
        skeleton = skeletonize_3d(thresholded_arr)
        x_pts, y_pts, z_pts = [], [], []

        for k in range(len(skeleton)):
            try:
                y_ind, x_ind = np.where(skeleton[k] == 1)
                if len(x_ind) != 0:
                    x_pts.extend(x_ind)
                    y_pts.extend(y_ind)
                    z_pts.extend(np.ones(len(y_ind))*k)
            except ValueError:
                print(np.where(skeleton[k] == 1))

        coords = np.stack((x_pts[:], y_pts[:], z_pts[:]))
        
        savepath = os.path.join(savefolder, filename)
        np.save(f"{savepath}.npy", coords.T)
        
    #iterate through all .tif files and complete nuclei removal and skeletonization
    with tqdm(total=numfiles, desc='Processing Images', unit='img') as pbar:
        for i in range(numfiles):
            filepath = files[i]
            filename = filenames[i]
            
            pbar.set_description(f'Image {i+1}/{numfiles}')
            
            if str(filename) + '.npy' not in os.listdir(savefolder):
                print(filename)
                arr = io.imread(filepath)

                if remove_nuclei == True:
                    arr = nuclei_removal(arr)
                    
                Skeletonize(arr, filename)
            else:
                print("already skeletonized")
                print(filename)
                
            pbar.update(1)
        
    return savefolder