import numpy as np
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms import minimum_spanning_tree
from skimage import io
import cv2 as cv
import pandas as pd
import json
import os
import logging
from GUIs.Useful_functions import coords_to_id, find_parent, find_length

def make_tree(coords : np.ndarray, 
              root_coords: list) -> dict:
    '''
    Builds a k-d tree for efficient nearest neighbour search, finds the k-nearest neighbours for each point and creates
    a graph with weighted edges between each point and its neighbours. finally, computes the minimum spanning tree using 
    those weighted edges
    
    Parameters
    ----------
    coords : np.ndarray
        3d array with coordinates of all points in the skeleton
    root_coords : np.ndarray
        array of x,y,z coordinates of root node to generate tree from
        
    Returns
    -------
    tree : dict
        returns a dictionary with every key (parent node) connected to its values (children)

    '''
    tree = KDTree(coords, copy_data=True)
    k = 3 #2 #the number of nearest neighbors (k) to consider
    distances, indices = tree.query(coords, k=k+1)  #query neighbours, k+1 because the point itself is included as a neighbor

    G = nx.Graph()
    for i in range(len(coords)):
        for j in range(1, k+1):  # Skip the first neighbor, which is the point itself
            G.add_edge(i, indices[i, j], weight=distances[i, j])
            
    mst = minimum_spanning_tree(G)
    tree = {}
    visited = set()

    def dfs(node):
        visited.add(node)
        neighbors = [n for n in mst.neighbors(node) if n not in visited]
        tree[tuple(coords[node])] = [tuple(coords[child]) for child in neighbors]
        for child in neighbors:
            dfs(child)

    root = np.where((coords == root_coords).all(axis=1))[0][0]
    dfs(root)
    return tree

def make_nodes_df(tree : dict, 
            root_coords : list, 
            start_coords : list) -> tuple[pd.DataFrame, list, list]:
    '''
    Converts a tree structure to a DataFrame representation.

    Parameters
    ----------
    tree : dict
        A dictionary representing the tree structure.
    root_coords : numpy.ndarray
        The coordinates of the root node.
    start_coords : numpy.ndarray
        The coordinates of the start node.

    Returns
    -------
    nodes_df : pd.dataframe
        The DataFrame representation of the tree nodes.
    end_nodes : list
        A list of end node IDs.
    branching_nodes : list
        A list of branching node IDs.

    Notes
    -----
    This function converts a tree structure, specified as a dictionary,
    to a DataFrame representation. The tree structure should have node
    coordinates as keys and lists of child coordinates as values.

    The function performs a depth-first search (DFS) traversal of the tree,
    starting from the root coordinates. It keeps track of visited nodes to
    avoid revisiting them. The resulting DataFrame contains columns for node
    ID, x, y, z coordinates, parent ID, and node tags. The parent ID is set
    to -1 for the start coordinates. The function also identifies end nodes
    and branching nodes, storing their IDs in separate lists.

    '''
    data = []
    visited = set()
    branching_nodes = []
    end_nodes = []
        
    def dfs(node_coords, parent_id, start_bool):
        node_id = coords_to_id(node_coords) 
        visited.add(tuple(node_coords))
        
        if (node_coords == start_coords).all():
            start_bool = False
            
        if start_bool == False: #add to dataframe if past start coords
            row = {'node_id': node_id, 'x': node_coords[0], 'y': node_coords[1], 'z': node_coords[2], 'parent_id':parent_id, 'node_tags':'NaN'}
            data.append(row)
        
        children = [c for c in tree[tuple(node_coords)] if tuple(c) not in visited] #returns a list of children
        
        if len(children) > 1 and start_bool == False:
            branching_nodes.append(node_id)
        elif len(children) == 0 and start_bool == False:
            end_nodes.append(node_id)
            
        for child_coords in children:
            dfs(child_coords, node_id, start_bool)
    
    dfs(root_coords, -1, True) #last one = is it before start node
    
    nodes_df = pd.DataFrame(data)
    nodes_df.at[0, 'parent_id'] = -1 #setting start coord parent id to -1
    return nodes_df, end_nodes, branching_nodes

def make_branches_df(nodes_df: pd.DataFrame, 
                        p_start_node: int, 
                        p_end_node: int, 
                        end_nodes: list, 
                        branchthreshold: float, 
                        nodethreshold: float,
                        scaling: np.ndarray) -> pd.DataFrame:
    """
    Constructs a pandas DataFrame representing branches of a given tree structure. 
    
    The function builds a primary branch starting from p_end_node and iteratively 
    adding parent nodes until it reaches p_start_node. Afterwards, secondary and 
    tertiary branches are added. A branch degree is then defined for each branch. 
    The function finally measures branch lengths and discards branches that don't 
    meet the specified threshold. The returned DataFrame includes additional 
    characteristics of the branches such as the number of nodes and the distance of 
    branching points along parent branches.
    
    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame with nodes information. This is expected to include a column
        for the node's parent and its spatial coordinates.
        
    p_start_node : int
        Node id where the primary branch starts.
        
    p_end_node : int
        Node id where the primary branch ends.
        
    end_nodes : List[int]
        List of node ids that represent the endpoints of all branches.
        
    branchthreshold : float
        Length threshold to consider a sequence of nodes as a branch.
        
    nodethreshold : int
        Minimum number of nodes for a sequence to be considered a branch.
    
    scaling : np.ndarray
        Array of three values representing the distance of each pixel in the x,y,z directions in microns
        
    Returns
    -------
    branches : pd.DataFrame
        DataFrame representing the branches in the structure. Each row represents a
        branch and columns provide information such as branch type (primary, secondary, etc),
        start and end nodes, branch length, and more.
        
    Notes
    -----
    This function assumes that the `nodes_df` DataFrame contains information that allows
    the construction of the tree structure (e.g., each node has a parent except for the root).
    It also assumes that branches are built in such a way that they never cross each other,
    i.e., the tree structure is well-defined.
    """
    
    #make primary branch
    primary_branch = []
    node = p_end_node
    while node != p_start_node:
        primary_branch.append(node) #adds the node to branch, starting with end node
        pnode = find_parent(node, nodes_df) #gets the parent of the node
        node = pnode #sets the next node to append as being the parent
    primary_branch.append(p_start_node)
    logging.info("Done making primary branch")
    
    #make other branches
    otherbranches = [] #getting all other secondary and tertiary branches
    for endnode in end_nodes:
        branch = []
        node = endnode
        while node not in primary_branch: 
            branch.append(node) 
            pnode = find_parent(node, nodes_df) 
            node = pnode
        branch.append(node) #adds the last node (on the primary branch, i.e. the root node) to the list
        if find_length(branch, nodes_df, scaling)[0] > branchthreshold and len(branch) > nodethreshold: #only add initially if longer than branch threshold (not a swelling)
            otherbranches.append([endnode, node, branch]) #endnode, startnode, branchnodes
    logging.info("Done making all branches")
        
    #setup branches dataframe
    branches = pd.DataFrame(otherbranches)
    branches.columns = ['branch_end_node', 'branch_start_node','branch_nodes']
    newcol = [2]*len(otherbranches) #will later be used to define secondary, tertiary, etc...
    branches['branch_type'] = newcol
    parents = [p_end_node]*len(branches) #setting all parents initially to the primary branch, will be changed if a tertiary branch is found
    logging.info("Done setting up branches df")
            
    #define each branch degree
    for b1, branch1 in enumerate(branches['branch_end_node']):
        for b2, branch2 in enumerate(branches['branch_end_node']):
            if branch2 != branch1: 
                #retrieve branch nodes
                branch1nodes = branches.at[b1, 'branch_nodes'] 
                branch2nodes = branches.at[b2,'branch_nodes']
                    
                if len(set(branch1nodes).intersection(branch2nodes)) > 1: #if they overlap at more than one point (which could just be branching from the same point on the parent branch)

                    if find_length(branch1nodes, nodes_df, scaling)[0] < find_length(branch2nodes, nodes_df, scaling)[0]: #if branch 1 is smaller it is the branch
                        #set branch1 as being 1 order higher than branch2 (which it branches off of) and change its parent to branch 2
                        branch2type = branches.at[b2,'branch_type'] 
                        branch1type = branch2type + 1
                        branches.at[b1, 'branch_type'] = branch1type 
                        parents[b1] = branch2
                        
                        #remove all overlap between the two from branch 1 nodes  and set new startnode for branch 1
                        newbranch1nodes = [x for x in branch1nodes if x not in branch2nodes] 
                        branch_rootnode = find_parent(newbranch1nodes[-1], nodes_df) #readding the 'root node' of the branch that is on the secondary branch
                        newbranch1nodes.append(branch_rootnode)
                        branches.at[b1, 'branch_nodes'] = newbranch1nodes
                        branches.at[b1, 'branch_start_node'] = branch_rootnode #set new startnode

                        #finally, find all branches that have branch1 as a parent (i.e. have already been processed) and change their designations
                        indexes = [i for i, x in enumerate(parents) if x == branch1]
                        
                        for ind in indexes:
                            #if the branch was not connected to branch1 but really branch2 (remember we just cut the overlap between branch1 and branch2 and gave it to branch2)
                            if branches.at[ind, 'branch_start_node'] not in newbranch1nodes:
                                parents[ind] = branch2
                                print("FIX: Changed parent ", ind, branch1, branch2)
                            #otherwise, increase the order of the branch by 1
                            else:
                                childbranchtype = branch1type + 1
                                print("FIX: Branch ", ind, " is degree ", childbranchtype, " and has branch ", branch1, " as a parent")
                                branches.at[ind, 'branch_type'] = childbranchtype
                        
    #add parent column to branch_df
    branches['parent_branch_end_node'] = parents
    
    #add primary branch to branch_df
    new_row = [p_end_node, p_start_node, primary_branch, 1, 'NaN'] 
    branches.loc[len(branches)] = new_row
    
    #find branch lengths and dropping branches that don't meet the threshold
    branchlengths = []
    numberofnodes = []
    branchestodrop = []
    for i in range(len(branches)): 
        branchnodes = branches.at[i, 'branch_nodes'] #retrieve all the node ids of nodes in the branch
        if find_length(branchnodes, nodes_df, scaling)[0] > branchthreshold and len(branchnodes) > nodethreshold: #checking to see if it qualifies as a branch
            numberofnodes.append(len(branchnodes))
            branchlengths.append(find_length(branchnodes, nodes_df, scaling)[0])
        else:
            branchestodrop.append(i)
    branches.drop(labels= branchestodrop, axis=0, inplace=True)
    branches.reset_index(drop=True, inplace=True)
    branches["number_nodes"] = numberofnodes
    branches["branch_length"] = branchlengths
    
    #find distance of branching point along parent branch
    parentlengths_tochild = [] #lengths to the branching point on the parent branch for each child branch
    parentlengths = [] #lengths of the parent branch for each child branch

    allbranches = list(branches['branch_end_node'])[:-1] #we don't care about the primary branch

    for i,branch in enumerate(allbranches):
        branchingpoint = branches.at[i, 'branch_start_node'] 
        parent_end_node = branches.at[i, 'parent_branch_end_node'] 
        p_ind = branches.loc[branches['branch_end_node'] == parent_end_node].index[0]
        parentnodes = branches.at[p_ind, 'branch_nodes'] 

        index = parentnodes.index(branchingpoint)
        parentnodes_tochild = parentnodes[index:]

        parentlength_tochild = find_length(parentnodes_tochild, nodes_df, scaling)[0]
        parentlength = branches.at[p_ind, 'branch_length'] 

        parentlengths_tochild.append(parentlength_tochild)
        parentlengths.append(parentlength)

    percentparentlength_tochild = list(np.array(parentlengths_tochild)/np.array(parentlengths))
    percentparentlength_tochild.append("NaN") #for primary branch

    branches["percentparentlength_tochild"] = percentparentlength_tochild
    
    #return final data table    
    return branches



def Run_branching(results: dict, 
                  filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the branching analysis pipeline on a set of 3D coordinates. 
    
    This function first loads coordinates from a .npy file and removes unwanted coordinates. 
    It then creates a minimum spanning tree and a DataFrame containing node information based 
    on the tree. Finally, a DataFrame of branches is generated based on node and endpoint information.
    
    Parameters
    ----------
    results : dict
        Dictionary that includes several parameters and values for the analysis:
        'results_coords': array-like, coordinates to remove from the analysis.
        'root_coords': array-like, coordinates of the root node.
        'start_coords': array-like, coordinates of the start node.
        'end_coords': array-like, coordinates of the end node.
        'scaling': float, scaling factor for the coordinates.
        'node_threshold': int, minimum number of nodes for a sequence to be considered a branch.
        'length_threshold': float, length threshold to consider a sequence of nodes as a branch.
    
    filepath : str
        The path to the .npy file containing the 3D coordinates.
        
    Returns
    -------
    branches : pd.DataFrame
        DataFrame representing the branches in the structure. Each row represents a branch and 
        columns provide information such as branch type (primary, secondary, etc), start and end 
        nodes, branch length, and more.
        
    nodes : pd.DataFrame
        DataFrame containing information about all nodes in the tree. Each row represents a node 
        and columns provide information such as node coordinates and its parent node.
        
    Notes
    -----
    This function assumes that the input .npy file contains a structured array of 3D coordinates. 
    The root, start, and end coordinates should all be present in the original .npy file and 
    not included in the remove_coords array.
    """
    #load useful stuff
    coords = results['coords']
    root_coords = results['root_coords']
    start_coords = results['start_coords']
    end_coords = results['end_coords']
    scaling = results['scaling']
    node_threshold = results['node_threshold']
    length_threshold = results['length_threshold']

    #make minimum spanning tree and nodes df
    tree = make_tree(coords, root_coords)
    logging.info("Done making tree")
    nodes, end_nodes, branching_nodes = make_nodes_df(tree, root_coords, start_coords)
    logging.info("Done making nodes df")
    
    #make branches df
    startnode = coords_to_id(start_coords) 
    endnode = coords_to_id(end_coords) 
    branches = make_branches_df(nodes, startnode, endnode, end_nodes, length_threshold, node_threshold, scaling)
    logging.info(f"Done making branches df with {len(branches)} branches")
    
    return branches, nodes