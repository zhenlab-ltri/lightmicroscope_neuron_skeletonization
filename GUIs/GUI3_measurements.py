import sys
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QListWidget, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from .Useful_functions import find_parent, find_xyz, find_length, coords_to_id
import logging


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
    def __init__(self, distance_widget, angle_widget, length_widget):
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

        self.fig.canvas.mpl_connect('pick_event',self.on_pick)

    #setting current status/task
    def select_distance(self):
        self.current_button = 1
    def select_angle(self):
        self.current_button = 2
    def select_length(self):
        self.current_button = 3

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


class GUI3ApplicationWindow(QMainWindow):
    close_ready = pyqtSignal()  # Create a custom signal
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
        self.angle_label = QLabel("Angle: ")
        self.angle_widget = QLabel("")
        self.length_label = QLabel("Length: ")
        self.length_widget = QLabel("")
        self.distance_button = QPushButton("Find Euclidean Distance")
        self.angle_button = QPushButton("Find 3-point Angle")
        self.length_button = QPushButton("Find Branch Length")
        self.erase_button = QPushButton("Erase All")
        self.close_button = QPushButton("CLOSE (NO SAVE BTW)")
        self.close_button.clicked.connect(self.close_window)

        #create plot and navigation toolbar
        self.plot_widget = GUI3Plot(self.distance_widget, self.angle_widget, self.length_widget) #, self.scale_label)
        
        self.toolbar = NavigationToolbar(self.plot_widget, self)
        self.distance_button.clicked.connect(self.plot_widget.select_distance)
        self.angle_button.clicked.connect(self.plot_widget.select_angle)
        self.length_button.clicked.connect(self.plot_widget.select_length)
        self.erase_button.clicked.connect(self.plot_widget.erase_all)

        vbox = QVBoxLayout()
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

        hbox = QHBoxLayout(self.main_widget)
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)

    def close_window(self):
        """Closes the window and emits the close_ready signal."""
        self.close_ready.emit()  
        self.close()


def GUI3(branches : pd.DataFrame,
         nodes_df : pd.DataFrame,
         scaling: np.ndarray,
         start_coords : list):
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
        
    data = {}

    def close_app():
        app.quit()

    window.close_ready.connect(close_app)  # Connect the signal to the receiving function

    app.exec_()  # Start the application event loop