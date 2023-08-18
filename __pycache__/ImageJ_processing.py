import imagej
import numpy as np
import os
from skimage import io
from skimage import exposure
import xarray
from tqdm import tqdm
import logging
import math
    
def ImageJ_processing(folderpath: str,
                        threshold1: dict, 
                        threshold2: dict, 
                        threshold3: dict,
                        num_slices_to_remove: int,
                        batchsize: int = 5):
    '''
    Preprocesses images using ImageJ and applies various thresholding methods and filters.
    
    Parameters:
    -----------
    folderpath : str
        Path to the folder containing the images.
    rawthreshold : int
        Raw threshold value for thresholding method 1 -- ex. 2
    threshtype2 : str
        Threshold type for method 2 -- ex. 'Triangle'
    threshtype3 : str
        Threshold type for method 3 -- ex. 'Default'
    minsize : int
        Minimum size of particles to keep -- ex. 250
    num_slices_to_remove : int
        Number of slices to remove from the start of each image array. 
        These are usually faulty images that we don't want to include
    batchsize : int
        ImageJ takes a lot of memory. How many files to run before re-initalizing ImageJ
        to save memory (this is a stupid fix)

    Returns
    -------
    str
        The path to the folder where the processed image files are saved.
    '''
    
    # initialize ImageJ2
    ij = imagej.init(mode = 'headless') #mode can be changed to 'interactive' to view processing in real time
    print(f" legacy service: {ij.legacy}")
    print(f"  legacy active? {ij.legacy and ij.legacy.isActive()}")
    print(f"ImageJ2 version: {ij.getVersion()}")

    #import all .tif files from folder
    files = []
    filenames = []
    for filename in sorted(os.listdir(folderpath)):
        if os.path.isfile(os.path.join(folderpath, filename)) and filename.endswith('.tif'):
            files.append(os.path.join(folderpath, filename))
            filenames.append(filename[:-4]) #remove .tif extension from filename
    numfiles = len(files)
    if numfiles > 0:
        logging.info(f"There are {numfiles} .tif files to be processed")
    else:
        raise RuntimeError(f'No suitable .tif files found in {folderpath}, make sure you put your images there')

    #create directory for processed images
    savefolder = os.path.join(folderpath, "Processed")
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    def dump_info(image):
        """A handy function to print details of an image object if errors arise. Not run in code but included for debugging purposes.

        Parameters:
        -----------
        image : object
            The image object to retrieve details from.

        Returns:
        --------
        None
        """
        name = image.name if hasattr(image, 'name') else None # xarray
        if name is None and hasattr(image, 'getName'): name = image.getName() # Dataset
        if name is None and hasattr(image, 'getTitle'): name = image.getTitle() # ImagePlus
        print(f" name: {name or 'N/A'}")
        print(f" type: {type(image)}")
        print(f"dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
        print(f"shape: {image.shape}")
        print(f" dims: {image.dims if hasattr(image, 'dims') else 'N/A'}")
        
    
    def process_image(i: int, 
                      filepath: str, 
                      filename: str,
                      ij):
        '''
        Process a single image using ImageJ and apply different thresholding methods.

        Parameters:
        -----------
        i : int
            Index of the image being processed.
        filepath : str
            Path to the image file.
        filename : str
            Name of the image file.

        Returns:
        --------
        None
        
        Notes:
        ------
        Threshold1, Threshold2, Threshold3 dictionaries are not explicitly passed but are defined in the parent function
        '''
        
        
        #opening .tif image as a np array
        arr = io.imread(filepath) #[num_slices_to_remove+1:,:,:]
        if len(arr) > 150:
            #print(f"{int(len(arr)/6)} slices removed")
            #arr = arr[int(len(arr)/7):,:,:]
            arr = arr[40:, :, :]
        else:
            arr = arr[num_slices_to_remove+1:,:,:]
        logging.info(f'''Starting image:
        Filename:   {filename}
        Pixel sum:  {np.sum(arr)}
        Num slices: {arr.shape[0]}
        ''')
        
        #normalizing intensity from 0-255 and converting to uint8 datatype as xarray
        arr = exposure.rescale_intensity(arr, out_range=(0, 255))
        arr = (arr + 0.5).astype(np.uint8) #the +0.5 is so that it rounds to the nearest integer instead of always down
        x_array = xarray.DataArray(arr, name='imp', dims=('pln','row','col'))
        del arr
        
        #defining the path for the processed images to be saved at
        savepath = os.path.join(savefolder,filename)
        
        #running thresholding method 1 (raw threshold)
        imp = ij.py.to_imageplus(x_array) #create imagePlus object from arr
        ij.ui().show(imp)
        ij.IJ.run(imp, "8-bit", "")
        ij.IJ.run(imp, "Gaussian Blur 3D...", f"x={threshold1['blur_sigma'][0]} y={threshold1['blur_sigma'][1]} z={threshold1['blur_sigma'][2]}")
        ij.IJ.run(imp, "Subtract Background...", f"rolling={threshold1['sb_radius']} sliding disable stack")
        imp.setTitle("PROCESSED.tif")
        ij.IJ.setRawThreshold(imp, threshold1['rawthresh'], 255)
        ij.IJ.run(imp, "Analyze Particles...", f"size={threshold1['minsize']}-Infinity show=Masks overlay stack")
        result = ij.WindowManager.getImage("Mask of PROCESSED.tif")
        ij.IJ.saveAs(result, "Tiff", f"{savepath}_thresh1.tif")
        imp.close()
        logging.info(f'Done thresholding method 1')
        
        #running threshold method 2
        imp = ij.py.to_imageplus(x_array) 
        ij.ui().show(imp)
        ij.IJ.run(imp, "8-bit", "")
        ij.IJ.run(imp, "Gaussian Blur 3D...", f"x={threshold2['blur_sigma'][0]} y={threshold2['blur_sigma'][1]} z={threshold2['blur_sigma'][2]}")
        ij.IJ.run(imp, "Subtract Background...", f"rolling={threshold2['sb_radius']} sliding disable stack")
        imp.setTitle("PROCESSED.tif")
        ij.IJ.setAutoThreshold(imp, f"{threshold2['threshtype']} dark no-reset stack")
        ij.IJ.run(imp, "Analyze Particles...", f"size={threshold2['minsize']}-Infinity show=Masks overlay stack")
        result = ij.WindowManager.getImage("Mask of PROCESSED.tif")
        ij.IJ.saveAs(result, "Tiff", f"{savepath}_thresh2.tif")
        imp.close()
        logging.info(f'Done thresholding method 2')
        
        #running threshold method 3
        imp = ij.py.to_imageplus(x_array) 
        ij.ui().show(imp)
        ij.IJ.run(imp, "8-bit", "")
        ij.IJ.run(imp, "Gaussian Blur 3D...", f"x={threshold3['blur_sigma'][0]} y={threshold3['blur_sigma'][1]} z={threshold3['blur_sigma'][2]}")
        ij.IJ.run(imp, "Subtract Background...", f"rolling={threshold3['sb_radius']} sliding disable stack")
        imp.setTitle("PROCESSED.tif")
        ij.IJ.setAutoThreshold(imp, f"{threshold3['threshtype']} dark no-reset stack")
        ij.IJ.run(imp, "Analyze Particles...", f"size={threshold3['minsize']}-Infinity show=Masks overlay stack")
        result = ij.WindowManager.getImage("Mask of PROCESSED.tif")
        ij.IJ.saveAs(result, "Tiff", f"{savepath}_thresh3.tif")
        imp.close()
        logging.info(f'Done thresholding method 3')
        
        del x_array
        # ij.IJ.run("Dispose All Windows", "/all image")

    #all files are processed iteratively
    with tqdm(total=numfiles, desc='Processing Images', unit='img') as pbar:

        for i in range(numfiles):
            filepath = files[i]
            filename = filenames[i]
            pbar.set_description(f'Image {i+1}/{numfiles}')
            
            if str(filename)+'_thresh3.tif' not in os.listdir(savefolder):
                print(filename)
                try:
                    process_image(i, filepath, filename, ij)
                except IndexError:
                    pass
            else:
                print(f"{filename} processed already")
            
            # Close all opened tabs to lighten memory (even if you can't see them)
            ij.WindowManager.setTempCurrentImage(None)
            ij.IJ.run("Close All")
            
            pbar.update(1)
            
    ij.context().dispose() #Close ImageJ instance
    return savefolder


    # num_batches = math.ceil(numfiles / batchsize)
#         for batch in range(num_batches):
#             start = batch * batchsize + 47
#             end = min((batch + 1) * batchsize, numfiles) + 47
#             print(f'Starting batch {batch + 1} of {num_batches}, processing files {start + 1} to {end}')

#             for i in range(start, end):
#                 filepath = files[i]
#                 filename = filenames[i]
#                 pbar.set_description(f'Image {i+1}/{numfiles}')
                
#                 #COULD ADD QUICK CHECK FOR IF IT HAS THREE DIMS HERE (to continue)
#                 try:
#                     arr = io.imread(filepath)[num_slices_to_remove+1:,:,:]
#                     if arr.shape[0] < 100:
#                         del arr
#                         process_image(i, filepath, filename, ij)
#                     else:
#                         logging.info(f"File {filename} is probably an invalid file -- too many slices")
#                 except IndexError:
#                     logging.info(f"File {filename} is an invalid file")
                
#                 # Close all opened tabs to lighten memory (even if you can't see them)
#                 ij.WindowManager.setTempCurrentImage(None)
#                 ij.IJ.run("Close All")
                
#                 pbar.update(1)
            
#             #Dispose of ImageJ instance to clear up memory
#             ij.getContext().dispose()
#             # ij.context().dispose()
# # ij = imagej.init('...', new_instance=True)