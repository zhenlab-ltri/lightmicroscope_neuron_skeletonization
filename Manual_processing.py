import numpy as np
import logging
import os
from Run_branching import Run_branching
from GUIs.GUI1_choosepoints import GUI1
from GUIs.GUI2_checkbranches import GUI2
from GUIs.GUI3_measurements import GUI3

def initialization(filepath : str,
               filename : str, 
               branching_parameters : dict,
               existing_coords : np.ndarray = None) -> dict:
    """
    Runs GUI1 where users can clean up the data

    Parameters:
    -----------
        filepath : str
            String filepath of the saved .npy file with all points for that file
        filename : str
            Filename of file
        existing_coords : np.ndarray, optional
            Existing processed points for that file, if exists. Defaults to None.
        branching_parameters : dict
            Dictionary containing thresholds for determining branching. See Main for description
            
    Returns:
    --------
        dict
            'continue' : bool
                Boolean describing whether the 'done' button was pressed or 
                whether another button ('reset', 'next', 'back') was pressed
            'coords' : np.ndarray
                Array of all processed coordinates
            'root_coords' : np.ndarray
                Array describing the coordinates of the selected root node
            'start_coords' : np.ndarray
                Array describing the coordinates of the selected start node
            'end_coords' : np.ndarray
                Array describing the coordinates of the selected end node
            'length_threshold' : float
                Float describing the minimum length of a branch
            'node_threshold' : float
                Float describing the minimum number of nodes in a branch
            'scaling' : np.ndarray
                Array describing the scale of the plot in x,y,z directions
    """
    if existing_coords is not None: #if existing processed points exist
        return GUI1('No filepath', filename, branching_parameters, existing_coords)
    else:
        return GUI1(filepath, filename, branching_parameters, existing_coords = None)
    
def branching(data1, branching_parameters) -> dict:
    """
    Executes branching cycle with the provided data and parameters.

    Parameters:
    -----------
        data1 : dict
            See initialization() above for description.
        branching_parameters : dict
            Dictionary containing thresholds for determining branching. See Main for description

    Returns:
    --------
        dict
            'continue' : bool
                Boolean describing whether 'done' or 'go back' was pressed
            'branches' : pd.DataFrame
                Dataframe with information about every found branch (length, parent, degree, nodes, etc...)
            'nodes' : pd.DataFrame
                Dataframe with information about every node in the structure (x,y,z,parent, etc...)
    """
    try:
        return Run_branching(data1, branching_parameters)
    except Exception as e:
        print(f"Branching failed due to {e}")
        return {'continue' : False}
    
def naming_removing(data2):
    """
    Runs GUI2 and lets user name/remove branches or optionally 'go back' to GUI1

    Parameters:
    -----------
        data2 : dict
            See branching() above for description
    
    Returns:
    --------
        dict
            'branch_names' : dict
                Dictionary with assigned branch names from GUI2
            'branch_remove_list' : list
                List of branch indexes to drop
    """
    return GUI2(data2['branches'], data2['nodes'])

def analysis_cycle(data1 : dict, 
                   data2 : dict, 
                   data3 : dict, 
                   filename : str, 
                   savefolder : str):
    """
    Analyzes the branching cycle data and saves the result. Runs GUI3.

    Parameters:
    -----------
        data1 : dict
            See initialization() above for description.
        data2 : dict
            See branching() above for description
        data3 : dict
            See naming_removing() above for description
        filename : str
            Name of the file.
        savefolder : str
            Path to the folder where results will be saved.
    """
    branch_names_dict = {}
    for item in data3['branch_names']:
        key_value = item.split(':')
        key = int(key_value[0].strip())
        value = key_value[1].strip()
        if value:  # Only add entries with non-empty values
            branch_names_dict[key] = value
            
    data2['branches']['branch_name'] = [branch_names_dict.get(i,"") for i in range(len(data2['branches']))]
    data2['branches']['project'] = [filename]*len(data2['branches'])
    branch_remove_list = [int(i) for i in data3['branch_remove_list']]
    data2['branches'].drop(labels = branch_remove_list, axis = 0, inplace = True)
    data2['branches'].reset_index(drop=True, inplace=True)
    logging.info(f"{len(branch_remove_list)} branches dropped from df")
    data2['branches'].to_excel(f"{os.path.join(savefolder, filename)}_branches.xlsx")
    data2['nodes'].to_excel(f"{os.path.join(savefolder, filename)}_nodes.xlsx")
    logging.info(f"Done saving excel files for {filename}") 
    _ = GUI3(data2['branches'], data2['nodes'], data1['scaling'], data1['start_coords'])

def UserInputCycle(filepath, filename, savefolder, branching_parameters, existing_coords = None):
    """
    Manages the input cycle from the user for manual processing.

    Parameters:
    -----------
        filepath : str
            Path to the file.
        filename : str
            Name of the file.
        savefolder : str
            Path to the folder where results will be saved.
        branching_parameters : dict
            Dictionary containing thresholds for determining branching. See Main for description
        existing_coords : np.ndarray, optional
            Existing processed points for the file, if exists. Defaults to None.

    Returns:
    --------
        int
            Value determining the direction of the cycle (rerun same file, go to next or previous file).
    """
    
    data1 = initialization(filepath, filename, branching_parameters, existing_coords)
        
    if data1['continue'] == False: #Reset, next, or back pressed
        logging.info(f"Input Cycle for file stopped. Reason: {data1['direction']}")
        if data1['direction'] == 'next':
            return 1
        elif data1['direction'] == 'previous':
            return -1
        elif data1['direction'] == 'reset':
            return 0
        
    elif data1['continue'] == True: #Done pressed
        if len(data1['start_coords']) > 0 and len(data1['root_coords']) > 0 and len(data1['end_coords']) > 0: #if points have been selected for GUI1
            
            data2 = branching(data1, branching_parameters)
            
            if data2['continue'] == False:
                logging.info("Branching failed. Try again. Likely there was a problematic gap in the point  cloud structure between the p-end and p-start nodes, or root, start, end nodes were not chosen correctly")
                return UserInputCycle("No filepath", f"Retrying {filename}", savefolder, branching_parameters, existing_coords = data1['coords'])
                
            elif data2['continue'] == True:
            
                data3 = naming_removing(data2)
                
                if data3['continue'] == False:
                    logging.info("Go back was pressed. Please retry point selection.")
                    return UserInputCycle("No filepath", f"Retrying {filename}", savefolder, branching_parameters, existing_coords = data1['coords'])
                
                elif data3['continue'] == True:
                    analysis_cycle(data1, data2, data3, filename, savefolder)
                    return 1 # This will end the current call of UserInputCycle and move onto the next image

def Manual_processing(folderpath: str, branching_parameters: dict):
    '''
    Processes manual input for a list of .npy files in a directory.

    This function scans the specified directory for .npy files, launches a series of GUIs
    to process each file, and saves the results to a new 'RESULTS' directory. The function
    also handles cases where the user chooses to continue, go back or skip the file.
    
    If a file has already been processed and results saved, it is skipped in the processing.

    Parameters
    ----------
    folderpath : str
        The path to the directory containing .npy files to be processed
    branching_parameters : dict
        Dictionary containing thresholds for determining branching. See branching_cycle for more info.
    '''
    
    #retrieve all .npy files in processed coords folder
    files = []
    filenames = []
    for filename in sorted(os.listdir(folderpath)):
        if os.path.isfile(os.path.join(folderpath, filename)) and filename.endswith('.npy'):
            files.append(os.path.join(folderpath,filename))
            filenames.append(filename[:-4]) #remove the .npy extension when adding to filenames
    numfiles = len(files)
    if numfiles > 0:
        logging.info(f"There are {numfiles} processed .npy files to be manually processed")
    else:
        raise RuntimeError(f'No suitable .npy files found in {folderpath}, make sure Skeletonization processing occured correctly')
    
    #create directory for skeletonized coordinate files
    savefolder = os.path.join(folderpath, "RESULTS")
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    run_files_queue = [0] #list of all files that still need to be run
    
    #iterate through files using a queue
    while len(run_files_queue) > 0:
        ind = run_files_queue[0]
        filepath = files[ind]
        filename = filenames[ind]
        c = UserInputCycle(filepath, filename, savefolder, branching_parameters) #will return whether to rerun the same file (0), go to next file (+1), or go to previous file (-1)
        new_ind = ind + c
        if new_ind < 0:
            logging.info("You are on file 1. Cannot go to previous file. Rerunning file 1.")
            new_ind = 0
        run_files_queue.append(new_ind)
        run_files_queue.pop(0)
        