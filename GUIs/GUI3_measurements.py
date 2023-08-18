import sys
import numpy as np
import math
import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QListWidget, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from .Useful_functions import find_parent, find_xyz, find_length, coords_to_id
import logging


def DB_analysis(T1tail, T1head, T2tail, T2head, branches, nodes_df, scaling, start_coords):
    """ **RIH SPECIFIC** Completes analysis of the T1 and T2 dauer branches given two sets of points representing the head and tail of each dauer branch

    Args:
        T1tail (np.ndarray): Coordinates of T1 tail point
        T1head (_type_): _description_
        T2tail (_type_): _description_
        T2head (_type_): _description_
        branches (_type_): _description_
        nodes_df (_type_): _description_
        scaling (_type_): _description_
        start_coords (_type_): _description_
    """
    try: 
        def DB_length(tail, head, nodes_df, scaling, start_coords):
            tail_id = int(str(int(tail[0])) + str(int(tail[1])) + str(int(tail[2])))
            head_id = int(str(int(head[0])) + str(int(head[1])) + str(int(head[2])))
            startnode = coords_to_id(start_coords) 

            nodelist = []
            curnode = head_id
            while curnode!=startnode and curnode!=tail_id:
                nodelist.append(curnode)
                curnode = find_parent(curnode,nodes_df)
            nodelist.append(curnode)

            if startnode not in nodelist:
                length, curve = find_length(nodelist, nodes_df, scaling)
                return length
            else:
                return("FAILED")
            
        
        #branch lengths
        T1_length = DB_length(T1tail, T1head, nodes_df, scaling, start_coords) 
        T2_length = DB_length(T2tail, T2head, nodes_df, scaling, start_coords)

        #defining a plane by the start coord, end coord, and branch point of T1
        end_node = branches.at[len(branches)-1, 'branch_end_node']
        end_node_row = nodes_df.loc[nodes_df['node_id']==end_node]
        T1_startnode = branches.loc[branches['branch_name']=='T1']['branch_start_node'].values[0]
        T1_startnode_row = nodes_df.loc[nodes_df['node_id']==T1_startnode]
        
        p1 = np.array(start_coords)
        p2 = np.array([end_node_row['x'].values[0], end_node_row['y'].values[0], end_node_row['z'].values[0]])
        p3 = np.array([T1_startnode_row['x'].values[0], T1_startnode_row['y'].values[0], T1_startnode_row['z'].values[0]])
        
        # Calculate two vectors that lie on the plane and the normal vector that defines the plane
        v1 = p2 - p1
        v2 = p3 - p1
        normal_vector = np.cross(v1, v2)
        
        # Define the vectors representing DBs
        T1vec = T1head - T1tail #np.array(data3['T1'][1]) - np.array(data3['T1'][0])
        T2vec = T2head - T2tail #np.array(data3['T2'][1]) - np.array(data3['T2'][0]) #head - tail, so select DB root first then the tip
        
        # If the dot product is negative, negate the normal vector (want the normal pointing 'up' towards T1vec and T2vec)
        dot_product = np.dot(normal_vector, T1vec)
        if dot_product < 0:
            normal_vector = -normal_vector 

        # Calculate the projections of the two DB vectors onto the generated plane, and the percentage of og vectors' length on the plane
        T1vec_proj = T1vec - (np.dot(T1vec, normal_vector)/np.linalg.norm(normal_vector)**2)*normal_vector # * normal_vector / np.linalg.norm(normal_vector)**2
        T2vec_proj = T2vec - (np.dot(T2vec, normal_vector)/np.linalg.norm(normal_vector)**2)*normal_vector #np.dot(T2vec, normal_vector) * normal_vector / np.linalg.norm(normal_vector)**2
        T1vec_percentage = np.linalg.norm(T1vec_proj) / np.linalg.norm(T1vec)
        T2vec_percentage = np.linalg.norm(T2vec_proj) / np.linalg.norm(T2vec)

        # Calculate dot products to determine position relative to the plane
        T1vec_dot = np.dot(T1vec, normal_vector) #if dot product is positive, means angle between the two vectors is smaller than 90 so they are both pointing 'up'
        T2vec_dot = np.dot(T2vec, normal_vector)
        T1vec_position = 'above' if T1vec_dot > 0 else 'below' if T1vec_dot < 0 else 'in'
        T2vec_position = 'above' if T2vec_dot > 0 else 'below' if T2vec_dot < 0 else 'in'
        
        #Find how much the DB are pointing towards the soma (defined as p1)
        # Calculate vectors pointing towards p1
        vec_p1_T1tail = p1 - T1tail.astype(float) 
        vec_p1_T2tail = p1 - T2tail.astype(float)
        
        vec_p1_T1tail_proj = vec_p1_T1tail - (np.dot(vec_p1_T1tail, normal_vector)/np.linalg.norm(normal_vector)**2)*normal_vector
        vec_p1_T2tail_proj = vec_p1_T2tail - (np.dot(vec_p1_T2tail, normal_vector)/np.linalg.norm(normal_vector)**2)*normal_vector

        # Normalize vectors to unit vectors
        vec_p1_T1tail_proj_hat = vec_p1_T1tail_proj / np.linalg.norm(vec_p1_T1tail_proj)
        vec_p1_T2tail_proj_hat = vec_p1_T2tail_proj / np.linalg.norm(vec_p1_T2tail_proj)
        T1vec_proj_hat = T1vec_proj / np.linalg.norm(T1vec_proj)
        T2vec_proj_hat = T2vec_proj / np.linalg.norm(T2vec_proj)

        # Calculate cosine of the angles
        cos_angle_T1 = np.dot(vec_p1_T1tail_proj_hat, T1vec_proj_hat)
        cos_angle_T2 = np.dot(vec_p1_T2tail_proj_hat, T2vec_proj_hat)

        #Figuring out if the DB is inside or outside a line connecting T1/T2 tail to the soma (i.e. if they go inwards or outwards)
        #for T1
        vec_T1tail_T2tail = T2tail - T1tail #vector pointing from tail of T1vec to tail of T2vec (head - tail)
        vec_T1tail_T2tail_proj = vec_T1tail_T2tail - (np.dot(vec_T1tail_T2tail, normal_vector)/np.linalg.norm(normal_vector)**2)*normal_vector #projected onto plane, anchored at T1 tail
        vec_T1tail_T2tail_proj_hat = vec_T1tail_T2tail_proj / np.linalg.norm(vec_T1tail_T2tail_proj)
        
        #angle 1 between vec_p1_T1tail_proj_hat and vec_T1tail_T2tail_proj_hat
        cos_angle_T1_1 = math.degrees(math.acos(np.dot(vec_p1_T1tail_proj_hat, vec_T1tail_T2tail_proj_hat)))
        
        #angle 2 between T1vec_proj_hat and vec_T1tail_T2tail_proj_hat
        cos_angle_T1_2 = math.degrees(math.acos(np.dot(T1vec_proj_hat, vec_T1tail_T2tail_proj_hat)))
        
        T1_gen_direction = 'inside' if cos_angle_T1_1 > cos_angle_T1_2 else 'outside' if cos_angle_T1_1 < cos_angle_T1_2 else 'along'
            
        #for T2
        vec_T2tail_T1tail = T1tail - T2tail #vector pointing from tail of T2vec to tail of T1vec (head - tail)
        vec_T2tail_T1tail_proj = vec_T2tail_T1tail - (np.dot(vec_T2tail_T1tail, normal_vector)/np.linalg.norm(normal_vector)**2)*normal_vector #projected onto plane, anchored at T2 tail
        vec_T2tail_T1tail_proj_hat = vec_T2tail_T1tail_proj / np.linalg.norm(vec_T2tail_T1tail_proj)
        
        #angle 1 between vec_p1_T2tail_proj_hat and vec_T2tail_T1tail_proj_hat
        cos_angle_T2_1 = math.degrees(math.acos(np.dot(vec_p1_T2tail_proj_hat, vec_T2tail_T1tail_proj_hat)))
        
        #angle 2 between T2vec_proj_hat and vec_T2tail_T1tail_proj_hat
        cos_angle_T2_2 = math.degrees(math.acos(np.dot(T2vec_proj_hat, vec_T2tail_T1tail_proj_hat)))
        
        T2_gen_direction = 'inside' if cos_angle_T2_1 > cos_angle_T2_2 else 'outside' if cos_angle_T2_1 < cos_angle_T2_2 else 'along'

        # # Print results
        # print('T1vec percentage: ', T1vec_percentage)
        # print('T2vec percentage: ', T2vec_percentage)
        # print('T1vec is {} the plane'.format(T1vec_position))
        # print('T2vec is {} the plane'.format(T2vec_position))
        # print('Cosine of the angle between T1vec_proj and vector pointing towards p1: ', cos_angle_T1) #1 would be very much pointing same direction, -1 is complete opposite
        # print('Cosine of the angle between T2vec_proj and vector pointing towards p1: ', cos_angle_T2)
        # print('The angle between T1vec_proj and vector pointing towards p1 is {:.2f} degrees'.format(math.degrees(math.acos(cos_angle_T1))))
        # print('The angle between T2vec_proj and vector pointing towards p1 is {:.2f} degrees'.format(math.degrees(math.acos(cos_angle_T2))))
        # print('T1vec is on the {} of line between T1 tail and soma'.format(T1_gen_direction))
        # print('T2vec is on the {} of line between T2 tail and soma'.format(T2_gen_direction))
        
        result_dict = {
            'T1_length' : T1_length,
            'T2_length' : T2_length,
            'T1vec_percentage' : T1vec_percentage,
            'T2vec_percentage' : T2vec_percentage,
            'T1_updown' : T1vec_position,
            'T2_updown' : T2vec_position,
            'T1vec_soma_cosine' : cos_angle_T1,
            'T2vec_soma_cosine' : cos_angle_T2,
            'T1vec_soma_angle' : math.degrees(math.acos(cos_angle_T1)),
            'T2vec_soma_angle' : math.degrees(math.acos(cos_angle_T2)),
            'T1_inout' : T1_gen_direction,
            'T2_inout' : T2_gen_direction   
        }
        return result_dict
    except KeyError:
        return f"There was a key error somehow {KeyError}"
    
    

class GUI3Plot(FigureCanvas):
    """Creates a 3D scatter plot and responds to user interactions.

    Parameters
    ----------
    distance_widget : QLabel
        The widget to display Euclidean distance.
    angle_widget : QLabel
        The widget to display the angle.
    length_widget : QLabel
        The widget to display the branch length.
    """
    def __init__(self, distance_widget, angle_widget, length_widget, T1_widget, T2_widget):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111,projection='3d')

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.ax.mouse_init()
        self.current_button = None
        self.drawn_items = []

        self.distance_points = []
        self.angle_points = []
        self.length_points = []
        self.distance_widget = distance_widget
        self.angle_widget = angle_widget
        self.length_widget = length_widget
        self.T1_widget, self.T2_widget = T1_widget, T2_widget
        self.T1_list, self.T2_list = [],[]

        self.fig.canvas.mpl_connect('pick_event',self.on_pick)

    #setting current status/task
    def select_distance(self):
        self.current_button = 1
    def select_angle(self):
        self.current_button = 2
    def select_length(self):
        self.current_button = 3
    def select_T1(self):
        self.current_button = 'T1'
    def select_T2(self):
        self.current_button = 'T2'

    #plotting all datapoints
    def plot(self, branches, nodes_df, scaling, start_coords):
        """Plots the scatter plot with the given data. 

        Parameters
        ----------
        branches : pd.DataFrame
            The branches to plot.
        nodes_df : pd.DataFrame
            The nodes to plot.
        scaling : np.ndarray
            The scaling factors.
        start_coords : list
            The start coordinates.
        """
        self.branches = branches
        self.nodes_df = nodes_df
        self.scaling = scaling
        self.start_coords = start_coords
        
        datax, datay, dataz = [],[],[]
        colorslist = ["#0000FF", "#008000", "#800080", "#FFA500", "#000080", "#800000", "#00FFFF", "#FF00FF", "#808000", "#008080", 
            "#0000A0", "#8000A0", "#00A080", "#808080", "#006400", "#8B0000", "#2F4F4F", "#800080", "#00008B", "#8B008B", 
            "#008B8B", "#696969", "#556B2F", "#4B0082", "#483D8B"]
        colorlabels = []
        for i, branch_nodes in enumerate(branches['branch_nodes']):
            x,y,z = find_xyz(branch_nodes, nodes_df)
            datax.extend(x)
            datay.extend(y)
            dataz.extend(z)
            colorlabels.extend([colorslist[i]]*len(x))
        data = np.array([np.array(datax),np.array(datay),np.array(dataz)])
        self.data = data.T
        self.ax.scatter(data[0], data[1], data[2], c=colorlabels, alpha=0.4, picker=5)

    def euclidean_distance(self, coord1, coord2):
        """Calculates Euclidean distance between two coordinates.

        Parameters
        ----------
        coord1 : list or np.ndarray
            First coordinate.
        coord2 : list or np.ndarray
            Second coordinate.

        Returns
        -------
        float
            Euclidean distance between the two coordinates.
        """
        dx = (coord2[0] - coord1[0])*self.scaling[0]
        dy = (coord2[1] - coord1[1])*self.scaling[1]
        dz = (coord2[2] - coord1[2])*self.scaling[2]
        dis = np.sqrt(dx**2 + dy**2 + dz**2)
        return dis

    def threepoint_angle(self, coord1, coord2, coord3):
        """Calculates the angle formed by three points.

        Parameters
        ----------
        coord1 : list or np.ndarray
            First coordinate.
        coord2 : list or np.ndarray
            Second coordinate.
        coord3 : list or np.ndarray
            Third coordinate.

        Returns
        -------
        float
            Angle in degrees.
        """
        a = np.array(coord1)*self.scaling
        b = np.array(coord2)*self.scaling
        c = np.array(coord3)*self.scaling
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def partbranch_length(self, coord1, coord2):
        """Calculates the branch length between two coordinates.

        Parameters
        ----------
        coord1 : list or np.ndarray
            First coordinate.
        coord2 : list or np.ndarray
            Second coordinate.

        Returns
        -------
        float or str
            Length and curve if the nodes are on the same branch, else "FAILED".
        """
        nodeid1 = int(str(int(coord1[0])) + str(int(coord1[1])) + str(int(coord1[2])))
        nodeid2 = int(str(int(coord2[0])) + str(int(coord2[1])) + str(int(coord2[2])))
        startnode = coords_to_id(self.start_coords) 

        #with nodeid1 first
        nodelist1 = []
        curnode = nodeid1
        while curnode!=startnode and curnode!=nodeid2:
            nodelist1.append(curnode)
            curnode = find_parent(curnode, self.nodes_df)
        nodelist1.append(curnode)

        #now with nodeid2
        nodelist2 = []
        curnode = nodeid2
        while curnode!=startnode and curnode!=nodeid1:
            nodelist2.append(curnode)
            curnode = find_parent(curnode, self.nodes_df)
        nodelist2.append(curnode)
        
        if startnode not in nodelist1:
            realnodelist = nodelist1
        elif startnode not in nodelist2:
            realnodelist = nodelist2
        else:
            return("FAILED")
        
        length, curve = find_length(realnodelist, self.nodes_df, self.scaling)
        return length, curve

    def erase_all(self):
        """Erases all drawn items on the plot."""
        for item in self.drawn_items:
            item.remove()
            self.drawn_items = []
            self.draw()
            self.angle_widget.setText("")
            self.distance_widget.setText("")
            self.length_widget.setText("")
            self.T1_widget.clear()
            self.T2_widget.clear()
            self.distance_points = []
            self.angle_points = []
            self.length_points = []
            self.T1_list = []
            self.T2_list = []

    def start_DB_analysis(self):
        results = DB_analysis(np.array(self.T1_list[0]), np.array(self.T1_list[1]), np.array(self.T2_list[0]), np.array(self.T2_list[1]), self.branches, self.nodes_df, self.scaling, self.start_coords)
        print(results)
        self.T1_list = []
        self.T2_list = []
        self.T1_widget.clear()
        self.T2_widget.clear()
        
    def on_pick(self, event):
        """Responds to pick event. Will draw lines or points and call the selected measurement function (see functions above).

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event.
        """
        if self.current_button is not None:
            ind = event.ind[0]
            coords = self.data[ind]

            if self.current_button == 1: #euclidean distance
                self.distance_points.append(coords)
                picked_point = self.ax.scatter(coords[0],coords[1],coords[2], color='r',s=50, alpha=1)
                self.drawn_items.append(picked_point)
                self.draw()
                if len(self.distance_points) == 2:
                    coords1 = self.distance_points[0]
                    coords2 = self.distance_points[1]
                    line, = self.ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], [coords1[2], coords2[2]], c='r', lw = 2, alpha=1)
                    self.drawn_items.append(line)
                    self.draw()
                    dist = self.euclidean_distance(coords1, coords2)
                    self.distance_widget.setText(str(dist))
                    self.distance_points = []

            if self.current_button == 2: #angle
                self.angle_points.append(coords)
                picked_point = self.ax.scatter(coords[0],coords[1],coords[2], color='b',s=50, alpha=1)
                self.drawn_items.append(picked_point)
                self.draw()
                if len(self.angle_points) == 2:
                    line, = self.ax.plot([self.angle_points[0][0], self.angle_points[1][0]], [self.angle_points[0][1], self.angle_points[1][1]], [self.angle_points[0][2], self.angle_points[1][2]], c='b', lw = 2, alpha=1)
                    self.drawn_items.append(line)
                    self.draw()
                elif len(self.angle_points) == 3:
                    line, = self.ax.plot([self.angle_points[1][0], self.angle_points[2][0]], [self.angle_points[1][1], self.angle_points[2][1]], [self.angle_points[1][2], self.angle_points[2][2]], c='b', lw = 2, alpha=1)
                    self.drawn_items.append(line)
                    self.draw()
                    coords1 = self.angle_points[0]
                    coords2 = self.angle_points[1]
                    coords3 = self.angle_points[2]
                    angle = self.threepoint_angle(coords1, coords2, coords3)
                    self.angle_widget.setText(str(angle))
                    self.angle_points = []
            
            if self.current_button == 3: #branch length
                self.length_points.append(coords)
                picked_point = self.ax.scatter(coords[0],coords[1],coords[2], color='green',s=50, alpha=0.5)
                self.drawn_items.append(picked_point)
                self.draw()
                if len(self.length_points) == 2:
                    coords1 = self.length_points[0]
                    coords2 = self.length_points[1]
                    returns = self.partbranch_length(coords1, coords2)
                    if returns != 'FAILED':
                        self.length, self.curve = returns
                        for i in range(len(self.curve)-1):
                            line, = self.ax.plot([self.curve[i][0], self.curve[i+1][0]], [self.curve[i][1], self.curve[i+1][1]], [self.curve[i][2], self.curve[i+1][2]], c='green', lw = 2, alpha=1)
                            self.drawn_items.append(line)
                        self.draw()
                        self.length_widget.setText(str(self.length))
                        self.length_points = []
                    else:
                        logging.info("NOT ON SAME BRANCH -- CANNOT FIND BRANCH LENGTH")
                        self.length_points = []
                        
            if self.current_button == 'T1': #setT1 nodes
                    picked_point = self.ax.scatter(coords[0],coords[1],coords[2], color='purple',s=50, alpha=0.5)
                    self.drawn_items.append(picked_point)
                    self.draw()
                    self.T1_widget.addItem(str(coords))
                    self.T1_list.append(tuple(coords))
                    
            if self.current_button == 'T2': #setT1 nodes
                    picked_point = self.ax.scatter(coords[0],coords[1],coords[2], color='purple',s=50, alpha=0.5)
                    self.drawn_items.append(picked_point)
                    self.draw()
                    self.T2_widget.addItem(str(coords))
                    self.T2_list.append(tuple(coords))


class GUI3ApplicationWindow(QMainWindow):
    close_ready = pyqtSignal(object)  # Create a custom signal
    def __init__(self):
        """Creates the main application window.

        Signals
        -------
        close_ready : pyqtSignal
            Signal to indicate that the window is ready to be closed.
        """
        QMainWindow.__init__(self)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        #widgets
        self.distance_label = QLabel("Distance: ")
        self.distance_widget = QLabel("")
        self.distance_widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.angle_label = QLabel("Angle: ")
        self.angle_widget = QLabel("")
        self.angle_widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.length_label = QLabel("Length: ")
        self.length_widget = QLabel("")
        self.length_widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.distance_button = QPushButton("Find Euclidean Distance")
        self.angle_button = QPushButton("Find 3-point Angle")
        self.length_button = QPushButton("Find Branch Length")
        self.erase_button = QPushButton("Erase All")
        self.close_button = QPushButton("CLOSE")
        self.close_button.clicked.connect(self.close_window)
        self.T1_button = QPushButton("T1 nodes")
        self.T2_button = QPushButton("T2 nodes")
        self.T1_widget = QListWidget()
        self.T2_widget = QListWidget()
        self.DBAnalysis_button = QPushButton("Analyze T1/T2")

        #create plot and navigation toolbar
        self.plot_widget = GUI3Plot(self.distance_widget, self.angle_widget, self.length_widget, self.T1_widget, self.T2_widget) #, self.scale_label)
        self.file_name = QLabel("") 
        self.toolbar = NavigationToolbar(self.plot_widget, self)
        self.distance_button.clicked.connect(self.plot_widget.select_distance)
        self.angle_button.clicked.connect(self.plot_widget.select_angle)
        self.length_button.clicked.connect(self.plot_widget.select_length)
        self.erase_button.clicked.connect(self.plot_widget.erase_all)
        self.T1_button.clicked.connect(self.plot_widget.select_T1)
        self.T2_button.clicked.connect(self.plot_widget.select_T2)
        self.DBAnalysis_button.clicked.connect(self.plot_widget.start_DB_analysis)

        vbox = QVBoxLayout()
        vbox.addWidget(self.file_name)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.plot_widget)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.distance_button)
        vbox2.addWidget(self.angle_button)
        vbox2.addWidget(self.length_button)
        vbox2.addWidget(self.erase_button)

        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.distance_label)
        vbox3.addWidget(self.distance_widget)
        vbox3.addWidget(self.angle_label)
        vbox3.addWidget(self.angle_widget)
        vbox3.addWidget(self.length_label)
        vbox3.addWidget(self.length_widget)
        vbox3.addWidget(self.close_button)
        
        vbox4 = QVBoxLayout()
        vbox4.addWidget(self.T1_button)
        vbox4.addWidget(self.T1_widget)
        vbox4.addWidget(self.T2_button)
        vbox4.addWidget(self.T2_widget)
        vbox4.addWidget(self.DBAnalysis_button)

        hbox = QHBoxLayout(self.main_widget)
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        hbox.addLayout(vbox4)
            
    def setting_filename(self, filename):
        """
        Set the filename label in the GUI.

        Parameters
        ----------
        filename : str
            The name of the file.
        """
        self.file_name.setText(filename)
        
    def close_window(self):
        """Closes the window and emits the close_ready signal."""
        
        self.results = {
            'T1' : self.plot_widget.T1_list,
            'T2' : self.plot_widget.T2_list
        }
        self.close_ready.emit(self.results)  # Emit the signal with the results
        self.close()
        
        
def GUI3(branches : pd.DataFrame,
         nodes_df : pd.DataFrame,
         scaling: np.ndarray,
         start_coords : list,
         filename : str = ""):
    """Starts the main event loop for the application.

    Parameters
    ----------
    branches : pd.DataFrame
        The branches to plot.
    nodes_df : pd.DataFrame
        The nodes to plot.
    scaling : np.ndarray
        The scaling factors.
    start_coords : list
        The start coordinates.
    """
    app = QApplication.instance()  # checks if QApplication already exists
    if not app:  # create QApplication if it doesnt exist 
        app = QApplication(sys.argv)
        
    # app = QApplication(sys.argv) 
    window = GUI3ApplicationWindow()
    window.show()
    window.plot_widget.plot(branches, nodes_df, scaling, start_coords)
    window.setting_filename(filename)
        
    data = {}

    def receive_results(results):
        data.update(results)
        app.quit()

    window.close_ready.connect(receive_results)  # Connect the signal to the receiving function

    app.exec_()  # Start the application event loop
    
    return data