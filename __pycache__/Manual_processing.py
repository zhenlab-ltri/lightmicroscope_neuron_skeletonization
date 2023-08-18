import numpy as np
import logging
import os
import pickle
from Run_branching import Run_branching
from GUIs.GUI1_choosepoints import GUI1
from GUIs.GUI2_checkbranches import GUI2
from GUIs.GUI3_measurements import GUI3


def Manual_processing(folderpath: str):
    '''
    Processes manual input for a list of .npy files in a directory.

    This function scans the specified directory for .npy files, launches a series of GUIs
    to process each file, and saves the results to a new 'RESULTS' directory. The function
    also handles cases where the user chooses to continue, go back or skip the file.
    
    If a file has already been processed and results saved, it is skipped in the processing.

    Parameters
    ----------
    folderpath : str
        The path to the directory containing .npy files to be processed.

    Returns
    -------
    None
    
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
        
    def UserInputCycle(filepath, filename):
        data1 = GUI1(filepath, filename)
        if data1['continue'] == True:
            root_coords = data1['root_coords']
            start_coords = data1['start_coords']
            end_coords = data1['end_coords']
            scaling = data1['scaling']

            if len(start_coords) > 0 and len(root_coords) > 0 and len(end_coords) > 0:
                try:
                    branches, nodes_df = Run_branching(data1, filepath)
                except KeyError or ValueError or UnboundLocalError:
                    logging.info("Branching failed. Try again.")
                    return 0
                data2 = GUI2(branches, nodes_df)
            
                if data2['continue']:
                    branch_names = data2['branch_names']
                    # branch_names_dict = {int(item.split(':')[0].strip()): item.split(':')[1].strip() for item in branch_names}
                    branch_names_dict = {}
                    for item in branch_names:
                        key_value = item.split(':')
                        key = int(key_value[0].strip())
                        value = key_value[1].strip()
                        if value:  # Only add entries with non-empty values
                            branch_names_dict[key] = value
                    branches['branch_name'] = [branch_names_dict.get(i,"") for i in range(len(branches))]
                    branches['project'] = [filename]*len(branches)
                    branch_remove_list = [int(i) for i in data2['branch_remove_list']]
                    branches.drop(labels = branch_remove_list, axis = 0, inplace = True)
                    branches.reset_index(drop=True, inplace=True)
                    logging.info(f"{len(branch_remove_list)} branches dropped from df")
                    branches.to_excel(f"{os.path.join(savefolder, filename)}_branches.xlsx")
                    nodes_df.to_excel(f"{os.path.join(savefolder, filename)}_nodes.xlsx")
                    logging.info(f"Done saving excel files for {filename}") 
                    _ = GUI3(branches, nodes_df, scaling, start_coords)
                    return 1 # This will end the current call of UserInputCycle and move onto the next image
                else:
                    logging.info("Go back was pressed. Please retry point selection.")
                    return 0
            else:
                logging.info("GUI1 was not completed correctly. Please try again to proceed to branching.")
                return 0
        else:
            if data1['direction'] == 'next':
                return 1
            elif data1['direction'] == 'previous':
                return -1
            else: #if reset button pressed
                return 0

    run_files_queue = [60] #list of all files that still need to be run
    
    while len(run_files_queue) > 0:
        ind = run_files_queue[0]
        filepath = files[ind]
        filename = filenames[ind]
        c = UserInputCycle(filepath, filename) #will return whether to rerun the same file (0), go to next file (+1), or go to previous file (-1)
        new_ind = ind + c
        if new_ind < 0:
            logging.info("You are on file 1. Cannot go to previous file. Rerunning file 1.")
            new_ind = 0
        run_files_queue.append(new_ind)
        run_files_queue.pop(0)
        

    # for i in range(numfiles):
    #     print(i)
    #     filepath = files[i]
    #     filename = filenames[i]
        
    #     if filename+".xlsx" not in os.listdir(savefolder): #if it hasn't already been processed
    #         UserInputCycle(filepath, filename)