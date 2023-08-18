#Alternate version in case it is useful to save the processed skeleton as a swc file that can then be processed!!

import numpy as np
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms import minimum_spanning_tree
import logging

from networkx.algorithms.tree import Tree

def save_as_swc(mst, coords, filepath):
    tree = Tree(mst) # Convert to a tree
    root = tree.get_root() # Define root as needed
    with open(filepath, "w") as file:
        file.write("# index type x y z r parent\n")
        for node in nx.dfs_preorder_nodes(tree, source=root):
            x, y, z = coords[node]
            parent = list(tree.neighbors(node))
            parent.remove(root) if root in parent else None
            parent = parent[0] if parent else -1
            file.write(f"{node} 3 {x} {y} {z} 1.0 {parent}\n")

def make_swc(coords : np.ndarray, 
              root_coords: list,
              branching_parameters : dict) -> dict:
    '''
    Builds a k-d tree for efficient nearest neighbour search, finds the k-nearest neighbours for each point and creates
    a graph with weighted edges between each point and its neighbours. Then computes the minimum spanning tree using 
    those weighted edges. If the minimum spanning tree is not fully connected, this function will connect the disconnected
    components by adding edges between the nearest points in each component.
    
    Parameters
    ----------
    coords : np.ndarray
        3d array with coordinates of all points in the skeleton
    root_coords : np.ndarray
        array of x,y,z coordinates of root node to generate tree from
    branching_parameters : dict
        dictionary of thresholds for branching used below
        
    Returns
    -------
    tree : dict
        returns a dictionary with every key (parent node) connected to its values (children)
    '''

    tree = KDTree(coords, copy_data=True)
    k = branching_parameters['num_nearestneighbours'] # The number of nearest neighbors (k) to consider
    distances, indices = tree.query(coords, k=k+1)  # Query neighbours (k+1 because the point itself is included as a neighbor)

    G = nx.Graph()
    for i in range(len(coords)):
        for j in range(1, k+1):  # Skip the first neighbor, which is the point itself
            G.add_edge(i, indices[i, j], weight=distances[i, j]) #Add weighted edge between neighbours
            
    mst = minimum_spanning_tree(G) #Construct minimum spanning tree from weights
    
    #Check if the minimum spanning tree is fully connected
            
    dist_threshold = branching_parameters['max_unconnected_distance']
    size_threshold = branching_parameters['min_unconnected_nodes']
    
    if not nx.is_connected(mst):
        components = list(nx.connected_components(mst))
        logging.info(f"Tree is not connected. Attempting correct connection of {len(components)} tree components")
        root_index = np.where((coords == root_coords).all(axis=1))[0][0]
        main_component = [c for c in components if root_index in c][0]
        other_components = [c for c in components if c != main_component]
        for component in other_components: 
            if len(component) > size_threshold: # Check size threshold
                min_distance = np.inf # Initialize minimum distance with infinity
                for node1 in main_component:
                    for node2 in component:
                        distance = np.linalg.norm(coords[node1] - coords[node2])
                        if distance < min_distance:
                            min_distance = distance
                            min_pair = (node1, node2)
                # Check distance threshold
                if min_distance <= dist_threshold:
                    # Add an edge between the pair of nodes with the smallest distance
                    mst.add_edge(min_pair[0], min_pair[1], weight=min_distance)
                    logging.info("Connected component")
    
    save_as_swc(mst, coords, "trialswc.swc")
    print("saved")
    
    
def Run_branching(results: dict, 
                  branching_parameters): 
    
    #load useful stuff
    coords = results['coords']
    root_coords = results['root_coords']
    start_coords = results['start_coords']

    #make minimum spanning tree and nodes df
    tree = make_swc(coords, root_coords, branching_parameters)
    logging.info("Done making swc file")
    
    
# #example analysis:
# import neurom as nm
# from neurom import viewer
# from neurom.fst import get, load_neuron

# def analyze_morphology(filepath):
#     # Load the neuron from the SWC file
#     neuron = load_neuron(filepath)

#     # Some morphometric features
#     total_length = get('total_length', neuron)
#     num_sections = get('number_of_sections', neuron)
#     num_neurites = get('number_of_neurites', neuron)
#     soma_radius = get('soma_radius', neuron)

#     print('Total length of all sections:', total_length)
#     print('Number of sections:', num_sections)
#     print('Number of neurites:', num_neurites)
#     print('Soma radius:', soma_radius)

#     # Visualize the neuron
#     viewer.draw(neuron)

#     # You can also extract more features using the get() function and the appropriate feature name

#     return neuron

# # Analyze the morphology of the given SWC file
# neuron = analyze_morphology('trialswc.swc')

    
