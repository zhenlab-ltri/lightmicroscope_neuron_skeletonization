import sys
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QListWidget, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
plugins_path = os.path.join(dir_path, 'env', 'lib', 'python3.10', 'site-packages', 'PyQt5', 'Qt', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path

class GUI1Plot(FigureCanvas):
    def __init__(self, remove_coords_widget, root_coords_widget, start_coords_widget, end_coords_widget, scale_label, bomb_label, threshold_label):
        """
        Initialize the 3D plot GUI.

        Parameters
        ----------
        remove_coords_widget : QListWidget
            Widget to display coordinates to be removed.
        root_coords_widget : QListWidget
            Widget to display the root coordinates.
        start_coords_widget : QListWidget
            Widget to display the starting coordinates.
        end_coords_widget : QListWidget
            Widget to display the end coordinates.
        scale_label : QLabel
            Label to display current scaling values.
        bomb_label : QLabel
            Label to display current bomb radius.
        threshold_label : QLabel
            Label to display current threshold values.
        """
        #intialize plot
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        #initialize lists to hold all the values
        self.remove_coords, self.root_coords, self.start_coords, self.end_coords, self.draw_points_list = [], [], [], [], []

        #intialize widgets and other stuff
        self.remove_coords_widget, self.root_coords_widget, self.start_coords_widget, self.end_coords_widget = remove_coords_widget, root_coords_widget, start_coords_widget, end_coords_widget
        self.bomb_label, self.scale_label, self.threshold_label = bomb_label, scale_label, threshold_label
        self.scaling = np.array((0.065,0.065,0.25))
        self.thresholds = np.array((0.1,2))
        self.current_radius = 5
        self.ax.mouse_init() #connect mouse
        self.current_button = None #store the last clicked choice button
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def plot(self, data):
        """
        Plot the 3D scatter plot.

        Parameters
        ----------
        data : ndarray
            Array of 3D coordinates to plot.
        """
        self.data = data
        self.prev_data = data
        self.scatter_plot = self.ax.scatter(data[:, 0], data[:, 1], data[:, 2], picker=5)  # 5 points tolerance

    def set_scale(self, x_scale, y_scale, z_scale):
        """
        Set the pixel scales of the 3D plot.

        Parameters
        ----------
        x_scale : QLineEdit
            Input for X axis scaling.
        y_scale : QLineEdit
            Input for Y axis scaling.
        z_scale : QLineEdit
            Input for Z axis scaling.
        """
        try:
            self.scaling = np.array((float(x_scale.text()), float(y_scale.text()), float(z_scale.text())))
            self.scale_label.setText(f"Current scale: {self.scaling}")
        except ValueError:
            print("Error: Scaling values must be numeric.")
            
    def set_threshold(self, length_threshold, node_threshold):
        """
        Set the thresholds for branching

        Parameters
        ----------
        length_threshold : QLineEdit
            Input for length threshold.
        node_threshold : QLineEdit
            Input for node threshold.
        """
        try:
            self.thresholds = np.array((float(length_threshold.text()), float(node_threshold.text())))
            self.threshold_label.setText(f"Current thresholds: {self.thresholds}")
        except ValueError:
            print("Error: Threshold values must be numeric.")

    def set_radius(self, radius_input):
        """
        Set the radius for the bomb button.

        Parameters
        ----------
        radius_input : QLineEdit
            Input for bomb radius.
        """
        try:
            radius_value = float(radius_input.text())
            self.current_radius = radius_value
            self.bomb_label.setText(f"Current radius: {self.current_radius}")
        except ValueError:
            print("Error: Radius must be a numeric value.")
    
    def find_nearby_points(self, point, radius):
        """
        Find points within a given radius of a reference point.

        Parameters
        ----------
        point : ndarray
            The reference point to find nearby points to.
        radius : float
            The radius within which to search for nearby points.

        Returns
        -------
        list
            List of nearby points.
        """
        distances = np.linalg.norm(self.data - point, axis=1)
        nearby_points = self.data[distances < radius]
        return [list(point) for point in nearby_points]  # Convert the points to lists to make them hashable

    '''these are all functions to connect buttons being pressed to plot settings'''
    def select_red(self):
        self.current_button = 1
    def select_green(self):
        self.current_button = 2
    def select_blue(self):
        self.current_button = 3
    def select_purple(self):
        self.current_button = 4
    def select_bomb(self):
        self.current_button = 'BOMB'
    def select_draw_points(self):
        self.current_button = 'DRAW'

    def reset_scale(self):
        """
        Reset the scale of the 3D plot to the new data range.
        """
        self.ax.set_xlim(self.data[:, 0].min(), self.data[:, 0].max())
        self.ax.set_ylim(self.data[:, 1].min(), self.data[:, 1].max())
        self.ax.set_zlim(self.data[:, 2].min(), self.data[:, 2].max())
        self.draw()

    def generate_line(self, point1, point2, step=1):
        """
        Generates a straight line of points between two points in 3D space.

        Parameters
        ----------
        point1 : array_like
            The coordinates of the first point. For example: [x1, y1, z1]
        point2 : array_like
            The coordinates of the second point. For example: [x2, y2, z2]
        step : int or float, optional
            The distance between each generated point. Default is 1.

        Returns
        -------
        points : ndarray
            An array of shape (n, 3) where n is the number of generated points. Each row is a point on the line.
        """
        point1, point2 = np.array(point1), np.array(point2)
        distance = np.linalg.norm(point2 - point1)  # Euclidean distance
        num_points = int(np.ceil(distance / step))
        
        # If the points are coincident, return the point itself
        if num_points == 0:
            return point1[None, :]
        
        points = [point1 + (point2 - point1) * t for t in np.linspace(0, 1, num_points)]
        for i,point in enumerate(points):
            point = np.array([float(int(j+0.5)) for j in point]) #basically just rounding them to nearest int
            points[i] = point #replacing values before
        points = np.array(points)
        points = np.unique(points, axis=0) #getting rid of any possible duplicates        
        return points

    def on_pick(self, event):
        """
        Handle the pick event based on the current active button.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event that contains information about the picked point.

        Notes
        -----
        The action taken depends on the current active button:
        - 'BOMB': Removes points within the current radius of the picked point.
        - 'DRAW' : Creates points in a line between two points
        - 1: Removes the single picked point.
        - 2: Adds the picked point as a root coordinate (green).
        - 3: Adds the picked point as a start coordinate (blue).
        - 4: Adds the picked point as an end coordinate (purple).
        """
        if self.current_button is not None:
            ind = event.ind[0]  # Get the first index if multiple points are picked
            coords = self.data[ind]

            if self.current_button == 'BOMB': #bomb destroy function
                nearby_points = self.find_nearby_points(coords, self.current_radius)
                for point in nearby_points:
                    if tuple(point) not in self.remove_coords:
                        self.remove_coords.append(tuple(point))
                        self.remove_coords_widget.addItem(str(np.array(point)))
                self.prev_data = self.data #for if undo button is clicked
                self.data = np.array([point for point in self.data if list(point) not in nearby_points])
                self.scatter_plot._offsets3d = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
                self.draw()
            elif self.current_button == 1: #remove single point
                self.prev_data = self.data
                self.data = np.delete(self.data, ind, axis=0)
                self.scatter_plot._offsets3d = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
                self.draw()
                if tuple(coords) not in self.remove_coords:
                    self.remove_coords.append(tuple(coords))
                    self.remove_coords_widget.addItem(str(coords))
            
            elif self.current_button == 'DRAW': #draw points between two points
                self.draw_points_list.append(coords)
                self.ax.scatter(coords[0],coords[1],coords[2], color='black',s=30, alpha=1)
                self.draw()
                if len(self.draw_points_list) == 2:
                    coords1 = self.draw_points_list[0]
                    coords2 = self.draw_points_list[1]
                    linepoints = self.generate_line(coords1, coords2, 1)
                    self.prev_data = self.data
                    self.data = np.concatenate((self.data, linepoints), axis=0)
                    self.data = np.unique(self.data, axis=0)
                    self.scatter_plot._offsets3d = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
                    self.draw_points_list = [] #reset list to empty
                    self.draw()        
            
            else:
                if self.current_button ==2: #set root coord
                    color = 'g'
                    self.root_coords.append(tuple(coords))
                    self.root_coords_widget.addItem(str(coords))
                elif self.current_button ==3: #set start coord
                    color = 'b'
                    self.start_coords.append(tuple(coords))
                    self.start_coords_widget.addItem(str(coords))
                elif self.current_button == 4: #set end coord
                    color = 'purple'
                    self.end_coords.append(tuple(coords))
                    self.end_coords_widget.addItem(str(coords))
                self.ax.scatter(*coords, color=color, s=100) # Add a larger  point at the picked location
                self.draw()
    
    def undo(self):
        self.data = self.prev_data
        self.scatter_plot._offsets3d = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
        self.draw()
         
    def save_results(self):
        #NOTE: Right now only the first item of root_coords, start_coords, end_coords are being saved and returned. This is not a problem for our purpose but the code could be improved.
        """
        Save the current state of the 3D plot.

        Returns
        -------
        tuple
            Tuple containing lists of removed points, root points, start points, end points, 
            and the length and node thresholds and scaling values.
        """
        
        def str_to_array(array_str):
            return np.array(list(map(float, array_str.strip('[]').split())))
        try:
            coords = np.unique(self.data, axis=0) #getting rid of any possible duplicates
            root_coords = [str_to_array(item.text()) for item in self.root_coords_widget.findItems("*",Qt.MatchWildcard)][0]
            start_coords = [str_to_array(item.text()) for item in self.start_coords_widget.findItems("*",Qt.MatchWildcard)][0]
            end_coords = [str_to_array(item.text()) for item in self.end_coords_widget.findItems("*",Qt.MatchWildcard)][0]
            length_threshold = float(self.thresholds[0])
            node_threshold = float(self.thresholds[1])
            scaling = self.scaling
        except IndexError:
            coords = []
            root_coords = []
            start_coords = []
            end_coords = []
            length_threshold = []
            node_threshold = []
            scaling = []
        return coords, root_coords, start_coords, end_coords, length_threshold, node_threshold, scaling

class GUI1ApplicationWindow(QMainWindow):
    
    results_ready = pyqtSignal(object)  # Create a custom signal to return data when close buttons are pressed
    
    def __init__(self):
        """
        Initialize the application window.
        """
        QMainWindow.__init__(self)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        labelsizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        #create variables to store thresholds
        self.threshold_label = QLabel("Set length and node threshold for branches (default = 0.1, 2")
        self.length_threshold = QLineEdit(self)
        self.node_threshold = QLineEdit(self)
        self.set_threshold_button = QPushButton("Set")
        self.cur_threshold_label = QLabel("Current thresholds: [0.1 2.]")
        self.cur_threshold_label.setSizePolicy(labelsizePolicy)
        self.set_threshold_button.clicked.connect(lambda: self.plot_widget.set_threshold(self.length_threshold, self.node_threshold))
        
        #create QListWidgets for the removed coords, root coords, start coords, end coords and corresponding labels
        self.remove_coords_widget = QListWidget()
        self.root_coords_widget = QListWidget()
        self.start_coords_widget = QListWidget()
        self.end_coords_widget = QListWidget()
        self.remove_coords_label = QLabel("Points to be removed")
        self.root_coords_label = QLabel("Root point (one only)")
        self.start_coords_label = QLabel("Start point (one only")
        self.end_coords_label = QLabel("End point (one only")
        
        #create widgets to allow for settable scaling
        self.scaling_label = QLabel("Set scaling below \n order: x,y,z\nenter all 3")
        self.x_scale = QLineEdit()
        self.y_scale = QLineEdit()
        self.z_scale = QLineEdit()
        self.set_scale_button = QPushButton("Set")
        self.scale_label = QLabel("Current Scale: [0.65 0.65 0.25]")
        self.scale_label.setSizePolicy(labelsizePolicy)
        self.set_scale_button.clicked.connect(lambda: self.plot_widget.set_scale(self.x_scale, self.y_scale, self.z_scale))
        
        #create widgets to handle 'bomb' effect
        self.radius_input = QLineEdit()  # Let the user input a radius
        self.set_radius_button = QPushButton("Set Bomb Radius")
        self.set_radius_button.clicked.connect(lambda: self.plot_widget.set_radius(self.radius_input))
        self.bomb_button = QPushButton('BOMB DESTROY')
        self.bomb_label = QLabel("Current Radius: 5")
        self.bomb_label.setSizePolicy(labelsizePolicy)
        self.reset_scale_button = QPushButton('Reset Scale')
        self.undo_button = QPushButton("Undo Last Change")
        
        #create 3D plot and toolbar, as well as filename widget
        self.plot_widget = GUI1Plot(self.remove_coords_widget, self.root_coords_widget, self.start_coords_widget, self.end_coords_widget, self.scale_label, self.bomb_label, self.cur_threshold_label)
        self.toolbar = NavigationToolbar(self.plot_widget, self)
        self.file_name = QLabel("") 

        #create buttons that choose the current function
        self.red_button = QPushButton('Remove Point')
        self.green_button = QPushButton('Select Root Point')
        self.blue_button = QPushButton('Select Start Point')
        self.purple_button = QPushButton('Select End Point')
        self.draw_points_button = QPushButton("Connect Two Points")
        self.red_button.clicked.connect(self.plot_widget.select_red)
        self.green_button.clicked.connect(self.plot_widget.select_green)
        self.blue_button.clicked.connect(self.plot_widget.select_blue)
        self.purple_button.clicked.connect(self.plot_widget.select_purple)
        self.draw_points_button.clicked.connect(self.plot_widget.select_draw_points)
        self.bomb_button.clicked.connect(self.plot_widget.select_bomb)
        self.reset_scale_button.clicked.connect(self.plot_widget.reset_scale)
        self.undo_button.clicked.connect(self.plot_widget.undo)

        #create three buttons to close the GUI
        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.save_and_close)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_stack)
        self.previous_button = QPushButton("PREVIOUS STACK")
        self.previous_button.clicked.connect(self.previous_stack)
        self.next_button = QPushButton("NEXT STACK")
        self.next_button.clicked.connect(self.next_stack)
        
        '''ALL LAYOUT STUFF BELOW'''        
        
        #vertical layout with toolbar and plot
        vbox = QVBoxLayout()
        vbox.addWidget(self.file_name)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.plot_widget)

        #vertical layout with buttons and bomb functionalities
        vbox2 = QVBoxLayout()
        vbox21 = QVBoxLayout()
        vbox21.addWidget(self.green_button)
        vbox21.addWidget(self.blue_button)
        vbox21.addWidget(self.purple_button)
        vbox21.addWidget(self.draw_points_button)
        vbox21.setSpacing(10)
        vbox22 = QVBoxLayout()
        vbox22.addWidget(self.red_button)
        vbox22.addWidget(self.bomb_button) 
        vbox22.addWidget(self.radius_input) 
        vbox22.addWidget(self.set_radius_button)  
        vbox22.addWidget(self.bomb_label)
        vbox22.setSpacing(10)
        vbox2.addLayout(vbox21)
        vbox2.addLayout(vbox22)
        vbox2.addWidget(self.reset_scale_button)
        vbox2.addWidget(self.undo_button)
        vbox2.setSpacing(60)

        #vertical layout with removed points label + coords list, both threshold boxes, and ugly button
        vbox3 = QVBoxLayout()
        vbox31 = QVBoxLayout()
        vbox31.addWidget(self.remove_coords_label)
        vbox31.addWidget(self.remove_coords_widget)
        vbox31.setSpacing(10)
        vbox32 = QVBoxLayout()
        vbox32.addWidget(self.threshold_label)
        vbox32.addWidget(self.length_threshold)
        vbox32.addWidget(self.node_threshold)
        vbox32.addWidget(self.set_threshold_button)
        vbox32.addWidget(self.cur_threshold_label)
        vbox32.setSpacing(10)
        vbox3.addLayout(vbox31)
        vbox3.addLayout(vbox32)
        hbox33 = QHBoxLayout()
        hbox33.addWidget(self.previous_button)
        hbox33.addWidget(self.next_button)
        vbox3.addLayout(hbox33)
        vbox3.setSpacing(30)

        #vertical layout with root coords, start coords, end coords, and done button
        vbox4 = QVBoxLayout()
        vbox41 = QVBoxLayout()
        vbox41.addWidget(self.root_coords_label)
        vbox41.addWidget(self.root_coords_widget)
        vbox41.addWidget(self.start_coords_label)
        vbox41.addWidget(self.start_coords_widget)
        vbox41.addWidget(self.end_coords_label)
        vbox41.addWidget(self.end_coords_widget)
        vbox41.setSpacing(10)
        vbox42 = QVBoxLayout()
        vbox42.addWidget(self.scaling_label)
        vbox42.addWidget(self.x_scale)
        vbox42.addWidget(self.y_scale)
        vbox42.addWidget(self.z_scale)
        vbox42.addWidget(self.set_scale_button)
        vbox42.addWidget(self.scale_label)
        vbox42.setSpacing(10)
        vbox4.addLayout(vbox41)
        vbox4.addLayout(vbox42)
        vbox4.addWidget(self.reset_button)
        vbox4.addWidget(self.done_button)
        vbox4.setSpacing(60)

        #horizontal layout with all the vertical boxes
        hbox = QHBoxLayout(self.main_widget)
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        hbox.addLayout(vbox4)

    def plotting_data(self, datafile):
        """
        Plot the 3D scatter plot with the data from the specified file.

        Parameters
        ----------
        datafile : ndarray
            Array of 3D coordinates to plot.
        """
        self.plot_widget.plot(datafile)
        
    def setting_filename(self, filename):
        """
        Set the filename label in the GUI.

        Parameters
        ----------
        filename : str
            The name of the file.
        """
        self.file_name.setText(filename)
        
    def reset_stack(self):
        """
        Close the application window and emit signal

        Notes
        -----
        This method prepares the result dictionary with 'continue' key set to False, emits the results_ready signal with 
        the results and then closes the window. The continue key being False is picked up in Manual_processing.py as well as the direction key
        """
        self.results = {
            'continue' : False,
            'direction' : 'reset'
        }
        self.results_ready.emit(self.results)  # Emit the signal with the results
        self.close()
        
    def previous_stack(self):
        """
        Close the application window and emit signal

        Notes
        -----
        This method prepares the result dictionary with 'continue' key set to False, emits the results_ready signal with 
        the results and then closes the window. The continue key being False is picked up in Manual_processing.py as well as the direction key
        """
        self.results = {
            'continue' : False,
            'direction' : 'previous'
        }
        self.results_ready.emit(self.results)  # Emit the signal with the results
        self.close()
    
    def next_stack(self):
        """
        Close the application window and emit signal

        Notes
        -----
        This method prepares the result dictionary with 'continue' key set to False, emits the results_ready signal with 
        the results and then closes the window. The continue key being False is picked up in Manual_processing.py as well as the direction key
        """
        self.results = {
            'continue' : False,
            'direction' : 'next'
        }
        self.results_ready.emit(self.results)  # Emit the signal with the results
        self.close()
        
    def save_and_close(self):
        """
        Save the results, close the application window, and emit signal with the results.

        Notes
        -----
        This method calls the save_results method of the plot_widget object, prepares the result dictionary with the 
        acquired results and the 'continue' key set to True, emits the results_ready signal with the results and then 
        closes the window. The continue key being True is picked up in Manual_processing.py
        """
        coords, root_coords, start_coords, end_coords, length_threshold, node_threshold, scaling = self.plot_widget.save_results()
        self.results = {
            'continue' : True,
            'coords' : coords,
            'root_coords': root_coords,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'length_threshold': length_threshold,
            'node_threshold': node_threshold,
            'scaling': scaling
        }            
        self.results_ready.emit(self.results)  # Emit the signal with the results
        self.close()
            
from scipy.spatial import cKDTree
import numpy as np

def remove_duplicates(datafile, tol=0.1):
    '''
    Removes duplicates from a numpy array of 3D points.
    
    Parameters
    ----------
    datafile : np.ndarray
        2D array with coordinates of all points.
    tol : float
        The tolerance for considering points as duplicates.

    Returns
    -------
    datafile : np.ndarray
        2D array with coordinates of all points with duplicates removed.
    '''
    tree = cKDTree(datafile)
    pairs = tree.query_pairs(tol)
    keep = np.ones((datafile.shape[0],), dtype=bool)

    for i, j in pairs:
        if keep[i] and keep[j]:
            keep[j] = False

    return datafile[keep]



def GUI1(filepath: str, filename: str, branching_parameters, existing_coords : None):
    """
    Start the GUI1 application.

    Parameters
    ----------
    filepath : str
        The path of the file from which data will be loaded.

    filename : str
        The name of the file from which data will be loaded.

    Returns
    -------
    dict
        The results obtained from the GUI1 application. If 'reset','next','back' are pressed the 
        dictionary will contain the 'continue' key set to False.
    """
    if existing_coords is not None:
        all_coords = existing_coords
        print("LOADING EXISTING COORDS")
    else:
        datafile = np.load(filepath)
        print("OG COORDS SHAPE: ", datafile.shape)
        all_coords = remove_duplicates(datafile, branching_parameters['remove_overlap']) 
        print("NEW COORDS SHAPE: ", all_coords.shape)
        

    app = QApplication.instance()  # checks if QApplication already exists
    if not app:  # create QApplication if it doesnt exist 
        app = QApplication(sys.argv)
        
    window = GUI1ApplicationWindow()
    window.plotting_data(all_coords) #
    window.setting_filename(filename)

    window.show()
    data = {}

    def receive_results(results):
        data.update(results)
        app.quit()

    window.results_ready.connect(receive_results)  # Connect the signal to the receiving function

    app.exec_()  # Start the application event loop

    return data