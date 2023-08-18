import os
from GUIs.GUI3_measurements import GUI3
import pandas as pd
import numpy as np
import ast
from GUIs.Useful_functions import find_xyz, find_length, coords_to_id

def find_files(path, word):
    matching_files = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            if word in filename and '0402' not in filename:
                full_path = os.path.join(root, filename)
                matching_files.append([root, filename, full_path])

    return matching_files

path_to_search = '/home/zhenlab/shuyu/PyQt_related/LMS_Pipeline/IMAGES'
word_to_find = "branches"
matching_files = find_files(path_to_search, word_to_find)

print(f"there are {len(matching_files)} processed files")

matching_files = matching_files[170:]


#CYCLE
for i, items in enumerate(matching_files):
    
    root, branches_filename, full_path = items
    print(i, branches_filename)
    nodes_filename = branches_filename.replace("branches", "nodes")
    
    try:
        #open files and run GUI3
        branches = pd.read_excel(full_path)
        branches['branch_nodes'] = branches['branch_nodes'].apply(lambda x: ast.literal_eval(x))
        nodes_df = pd.read_excel(os.path.join(root, nodes_filename))
        scaling = np.array([0.065,0.065,0.25])
        start_node = branches.at[len(branches)-1, 'branch_start_node']
        row = nodes_df.query('node_id == @start_node')
        start_coords = [row['x'].values[0], row['y'].values[0], row['z'].values[0]]
        data3 = GUI3(branches, nodes_df, scaling, start_coords, branches_filename[:-13])
    except KeyError or pd.errors.UndefinedVariableError:
        pass
#     print(data3) #returns a dictionary with 'T1' and 'T2' as keys
    
#     #defining a plane by the start coord, end coord, and branch point of T1
#     p1 = np.array(start_coords)
#     end_node = branches.at[len(branches)-1, 'branch_end_node']
#     row = nodes_df.query('node_id == @end_node')
#     p2 = np.array([row['x'].values[0], row['y'].values[0], row['z'].values[0]])
    
#     T1_start_node = branches.loc[branches['branch_name']=='T1']['branch_start_node'].values[0]
#     row = nodes_df.query('node_id == @T1_start_node')
#     p3 = np.array([row['x'].values[0], row['y'].values[0], row['z'].values[0]])
    
#     # Calculate two vectors that lie on the plane
#     v1 = p2 - p1
#     v2 = p3 - p1

#     # Calculate the normal vector of the plane
#     normal_vector = np.cross(v1, v2)
#     # Define the vectors representing DBs
#     T1vec = np.array(data3['T1'][1]) - np.array(data3['T1'][0])
#     T2vec = np.array(data3['T2'][1]) - np.array(data3['T2'][0]) #head - tail, so select DB root first then the tip
    
#     # If the dot product is negative, negate the normal vector (want the normal pointing towards T1vec and T2vec)
#     dot_product = np.dot(normal_vector, T1vec)
#     if dot_product < 0:
#         normal_vector = -normal_vector

#     n_norm = np.sqrt(sum(normal_vector**2))  
#     T1vec_proj = T1vec - (np.dot(T1vec, normal_vector)/n_norm**2)*normal_vector # * normal_vector / np.linalg.norm(normal_vector)**2
#     T2vec_proj = T2vec - (np.dot(T2vec, normal_vector)/n_norm**2)*normal_vector #np.dot(T2vec, normal_vector) * normal_vector / np.linalg.norm(normal_vector)**2

#     # Calculate the percentage of the original vectors' lengths that are on the plane
#     T1vec_percentage = np.linalg.norm(T1vec_proj) / np.linalg.norm(T1vec)
#     T2vec_percentage = np.linalg.norm(T2vec_proj) / np.linalg.norm(T2vec)

#     print('T1vec percentage: ', T1vec_percentage)
#     print('T2vec percentage: ', T2vec_percentage)
    
#     # Calculate dot products to determine position relative to the plane
#     T1vec_dot = np.dot(T1vec, normal_vector) #if dot product is positive, means angle between the two vectors is smaller than 90 so they are both pointing 'up'
#     T2vec_dot = np.dot(T2vec, normal_vector)

#     # Interpret dot products
#     T1vec_position = 'above' if T1vec_dot > 0 else 'below' if T1vec_dot < 0 else 'in'
#     T2vec_position = 'above' if T2vec_dot > 0 else 'below' if T2vec_dot < 0 else 'in'

#     # Print results
#     print('T1vec is {} the plane'.format(T1vec_position))
#     print('T2vec is {} the plane'.format(T2vec_position))

    

    
#     #also find how much the projections are pointing towards the soma
#     # Calculate vectors pointing towards p1
#     vec_p1_T1tail = p1 - np.array(data3['T1'][0]).astype(float) #head - tail TODO wait this isn't along the plane anymore bruhhhh, JUST PROJECT IT ONTO PLANE DUH
#     vec_p1_T2tail = p1 - np.array(data3['T2'][0]).astype(float)
    
#     vec_p1_T1tail_proj = vec_p1_T1tail - (np.dot(vec_p1_T1tail, normal_vector)/n_norm**2)*normal_vector
#     vec_p1_T2tail_proj = vec_p1_T2tail - (np.dot(vec_p1_T2tail, normal_vector)/n_norm**2)*normal_vector

#     # Normalize vectors to unit vectors
#     vec_p1_T1tail_proj_hat = vec_p1_T1tail_proj / np.linalg.norm(vec_p1_T1tail_proj)
#     vec_p1_T2tail_proj_hat = vec_p1_T2tail_proj / np.linalg.norm(vec_p1_T2tail_proj)
#     T1vec_proj_hat = T1vec_proj / np.linalg.norm(T1vec_proj)
#     T2vec_proj_hat = T2vec_proj / np.linalg.norm(T2vec_proj)

#     # Calculate cosine of the angles
#     cos_angle_T1 = np.dot(vec_p1_T1tail_proj_hat, T1vec_proj_hat)
#     cos_angle_T2 = np.dot(vec_p1_T2tail_proj_hat, T2vec_proj_hat)

#     print('Cosine of the angle between T1vec_proj and vector pointing towards p1: ', cos_angle_T1)
#     print('Cosine of the angle between T2vec_proj and vector pointing towards p1: ', cos_angle_T2)
#     import math
#     angle_T1 = math.acos(cos_angle_T1)
#     print('The angle between T1vec_proj and vector pointing towards p1 is {:.2f} degrees'.format(math.degrees(angle_T1)))
#     angle_T2 = math.acos(cos_angle_T2)
#     print('The angle between T2vec_proj and vector pointing towards p1 is {:.2f} degrees'.format(math.degrees(angle_T2)))
    
#     #for T1
#     vec_T1tail_T2tail = np.array(data3['T2'][0]) - np.array(data3['T1'][0]) #vector pointing from tail of T1vec to tail of T2vec
#     vec_T1tail_T2tail_proj = vec_T1tail_T2tail - (np.dot(vec_T1tail_T2tail, normal_vector)/n_norm**2)*normal_vector #projected onto plane, anchored at T1 tail
#     vec_T1tail_T2tail_proj_hat = vec_T1tail_T2tail_proj / np.linalg.norm(vec_T1tail_T2tail_proj)
    
#     #angle 1 between vec_p1_T1tail_proj_hat and vec_T1tail_T2tail_proj_hat
#     cos_angle_T1_1 = math.degrees(math.acos(np.dot(vec_p1_T1tail_proj_hat, vec_T1tail_T2tail_proj_hat)))
    
#     #angle 2 between T1vec_proj_hat and vec_T1tail_T2tail_proj_hat
#     cos_angle_T1_2 = math.degrees(math.acos(np.dot(T1vec_proj_hat, vec_T1tail_T2tail_proj_hat)))
    
#     #comparing the two
#     if cos_angle_T1_1 > cos_angle_T1_2:
#         print("T1 DB is inside of soma vec")
#     elif cos_angle_T1_1 < cos_angle_T1_2:
#         print("T1 DB is outside of soma vec")
#     else:
#         print("uh idk")
#     print(f"the two angles: {cos_angle_T1_1} and {cos_angle_T1_2}")
          
#     #for T2
#     vec_T2tail_T1tail = np.array(data3['T1'][0]) - np.array(data3['T2'][0]) #vector pointing from tail of T2vec to tail of T1vec
#     vec_T2tail_T1tail_proj = vec_T2tail_T1tail - (np.dot(vec_T2tail_T1tail, normal_vector)/n_norm**2)*normal_vector #projected onto plane, anchored at T2 tail
#     vec_T2tail_T1tail_proj_hat = vec_T2tail_T1tail_proj / np.linalg.norm(vec_T2tail_T1tail_proj)
    
#     #angle 1 between vec_p1_T1tail_proj_hat and vec_T1tail_T2tail_proj_hat
#     cos_angle_T2_1 = math.degrees(math.acos(np.dot(vec_p1_T2tail_proj_hat, vec_T2tail_T1tail_proj_hat)))
    
#     #angle 2 between T1vec_proj_hat and vec_T1tail_T2tail_proj_hat
#     cos_angle_T2_2 = math.degrees(math.acos(np.dot(T2vec_proj_hat, vec_T2tail_T1tail_proj_hat)))
    
#     #comparing the two
#     if cos_angle_T2_1 > cos_angle_T2_2:
#         print("T2 DB is inside of soma vec")
#     elif cos_angle_T2_1 < cos_angle_T2_2:
#         print("T2 DB is outside of soma vec")
#     else:
#         print("uh idk")
#     print(f"the two angles: {cos_angle_T2_1} and {cos_angle_T2_2}")
    
    
    
    
#     #find angle between vec_p1_T1tail_proj_hat and vec_T1tail_T2tail_proj_hat
#     #find angle between T1vec_proj_hat and vec_T1tail_T2tail_proj_hat
#     #if second angle is larger than the first one then the vector is on the outside, otherwise on the inside
    
    
#     # T1_direction_vector = start_coords - T1vec  
#     # T2_direction_vector = start_coords - T2vec
    
#     # T1_cosine_similarity = np.dot(T1vec, T1_direction_vector) / (np.linalg.norm(T1vec) * np.linalg.norm(T1_direction_vector))
#     # T2_cosine_similarity = np.dot(T2vec, T2_direction_vector) / (np.linalg.norm(T2vec) * np.linalg.norm(T2_direction_vector))

#     # print("T1: ", T1_cosine_similarity, "T2: ", T2_cosine_similarity)
    
#    # {'T1': [(379, 803, 32), (418, 827, 59)], 'T2': [(396, 749, 28), (390, 760, 28)]}
    

    