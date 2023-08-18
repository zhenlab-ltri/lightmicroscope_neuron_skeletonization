
from typing import List, Union, Tuple
import numpy as np
import pandas as pd
import logging
from typing import Tuple
ArrayType = Union[np.ndarray, List[float], pd.Series]

def coords_to_id(coords : any):
    """
    Converts an array or list or tuple of three coordinates into a unique identifier id
    
    Parameters:
    ----------
    coords : tuple, list, np.ndarray
        set of three values representing the x,y,z coordinates of a point

    Returns:
    --------
    id : int
        integer 'id' for the coordinate point, identified by concatenating its x,y,z coordinates
        
    """
    id = int(str(int(coords[0])) + str(int(coords[1])) + str(int(coords[2])))
    return id

def find_parent(child_id : int,
                nodes_df : pd.DataFrame):
    '''
    Returns parent id give the id of a child node by searching through the dataframe for that nodeid and finding its corresponding 'parent id'
    
    Parameters
    ----------
    child_id : int
        Unique identifier of the child node
    nodes_df : pd.DataFrame
        Dataframe containing information on all nodes
    
    Returns
    -------
    parent_id : int
        Unique identifier of the parent node of child node
    '''
    row = nodes_df.loc[nodes_df['node_id'] == child_id]
    parent_id = row['parent_id'].values[0]
    return parent_id #returns parent ID

    

def find_children(parent_id : int, 
                  nodes_df : pd.DataFrame) -> list:
    '''
    Finds the children of a parent node by searching through the DataFrame to find all nodes with a
    matching 'parent_id' and returns a list of their corresponding child IDs.

    Parameters
    ----------
    parent_id : int
        Unique identifier of the parent node.
    nodes_df : pd.DataFrame
        DataFrame containing information on all nodes.

    Returns
    -------
    children : list
        List of child IDs.
    '''
    rows = nodes_df.loc[nodes_df['parent_id'] == parent_id]
    children_ids = list(rows['node_id'].values[:])
    return children_ids #returns list of child ids

def distance(x1 : float, y1 : float, z1 : float, 
             x2 : float, y2 : float, z2 : float, 
             scaling : np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points in a 3D space.

    Parameters
    ----------
    x1, y1, z1 : float
        Coordinates of the first point.
    x2, y2, z2 : float
        Coordinates of the second point.
    scaling : np.ndarray

    Returns
    -------
    distance : float
        Euclidean distance between the two points scaled by the size of a pixel in each dimension
    """
    dx = (x2 - x1)*scaling[0]
    dy = (y2 - y1)*scaling[1]
    dz = (z2 - z1)*scaling[2]
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def find_xyz(nodes_list : ArrayType, 
             nodes_df : pd.DataFrame) -> tuple:
    """
    Extract x, y, z coordinates for a given list of nodes from the nodes_df DataFrame.

    Parameters
    ----------
    nodes_list : array-like
        List of node identifiers.
    nodes_df : pd.DataFrame
        DataFrame containing node information.

    Returns
    -------
    x, y, z : array-like
        x, y, z coordinates for the given nodes.
    """
    # Filter the DataFrame and order it based on the 'nodes' list
    branch_nodes = nodes_df[nodes_df['node_id'].isin(nodes_list)]  #clip the nodes_df dataframe to just those nodes
    branch_nodes = branch_nodes.set_index('node_id').loc[nodes_list].reset_index() #order the RIH dataframe in the same order that nodes_list was in

    #define x,y,z values of nodes on branch
    x = branch_nodes['x']
    y = branch_nodes['y']
    z = branch_nodes['z']
    return x, y, z

def smooth_branch(x : ArrayType, y : ArrayType, z : ArrayType,
                  window_size : int) -> tuple:
    '''
    Apply a smoothing function to a 3D curve defined by x, y, z coordinates.

    Smoothing is achieved through convolution with a uniform window function.
    To prevent edge effects and to ensure that the smoothed curve includes the 
    start and end points of the input arrays, the input arrays are padded 
    at the beginning and the end.

    The size of the padding is determined by the window size for the smoothing operation.
    The first and last (window_size - 1) // 2 elements of the input arrays are reversed
    and appended to the beginning and end of the arrays, respectively. This effectively
    extends the input arrays for the convolution operation.

    Parameters
    ----------
    x, y, z : array-like
        x, y, z coordinates defining the curve.
    window_size : int
        Size of the smoothing window.

    Returns
    -------
    x_smooth, y_smooth, z_smooth : array-like
        Smoothed x, y, z coordinates.
    '''
    padding = (window_size - 1) // 2
    padded_x = np.concatenate([x[padding:0:-1], x, x[-2:-padding-2:-1]])
    padded_y = np.concatenate([y[padding:0:-1], y, y[-2:-padding-2:-1]])
    padded_z = np.concatenate([z[padding:0:-1], z, z[-2:-padding-2:-1]])

    #smooth out branch
    x_smooth = np.convolve(padded_x, np.ones(window_size)/window_size, mode='valid')
    y_smooth = np.convolve(padded_y, np.ones(window_size)/window_size, mode='valid')
    z_smooth = np.convolve(padded_z, np.ones(window_size)/window_size, mode='valid')   
    return(x_smooth, y_smooth, z_smooth)

def find_length(nodes_list : ArrayType, 
                nodes_df : pd.DataFrame, 
                scaling : np.ndarray) -> float:
    '''
    Calculate the length of a curve defined by a list of nodes in the nodes_df DataFrame.

    Parameters
    ----------
    nodes_list : array-like
        List of node identifiers.
    nodes_df : pd.DataFrame
        DataFrame containing node information.

    Returns
    -------
    length : float
        Length of the curve.
    '''
    num_nodes = len(nodes_list)
    x,y,z = find_xyz(nodes_list, nodes_df)

    #choosing window size
    proceed = True
    if num_nodes < 3:  #too short of a branch to smooth, just take the node values as length
        curve = np.array((x,y,z)).T
        proceed = False
    elif num_nodes < 11: # short branch, use a short smoothing window
        window_size = 3
    elif num_nodes < 33: #use a slightly larger smoothing window
        window_size = 5
    else:  #use largest smoothing window
        window_size = 11
    
    #if long enough to need smoothing, fit a smoothed curve to the node points
    if proceed == True:
        x_smooth, y_smooth, z_smooth = smooth_branch(x,y,z, window_size)
        curve = np.array((x_smooth, y_smooth, z_smooth)).T   
       
    #find length of curve
    length = 0.0
    for j in range(1, len(curve)):
        x1, y1, z1 = curve[j-1]
        x2, y2, z2 = curve[j]
        length += distance(x1, y1, z1, x2, y2, z2, scaling)
    return(length, curve)