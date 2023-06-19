import sys
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QListWidget, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from .Useful_functions import find_xyz

class GUI2Plot(FigureCanvas):
    """
    This is a subclass of the FigureCanvas, a widget for embedding plots 
    into applications using general-purpose GUI toolkits like Qt. This class 
    handles all the 3D plotting functionalities and user interactions such 
    as clicking on points.

    Parameters
    ----------
    branches_widget : QListWidget
        A widget for displaying and handling the branches to be removed.
    named_branches_widget : QListWidget
        A widget for displaying and handling the named branches.
    named_branches : dict
        A dictionary to store branch numbers and their associated names.
    """
    def __init__(self, branches_widget, named_branches_widget, named_branches):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.branches_list = []
        self.branches_widget = branches_widget
        self.named_branches_widget = named_branches_widget
        self.named_branches = named_branches

        self.ax.mouse_init()
        self.current_button = None #store the last clicked choice button
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        
    def plot(self, branches, nodes_df):
        """
        Function to plot the branches in 3D space.

        Parameters
        ----------
        branches : pd.DataFrame
            A DataFrame containing information about the branches.
        nodes_df : pd.DataFrame
            A DataFrame containing information about the nodes.
        """
        self.branches = branches
        self.nodes_df = nodes_df
        text = ""
        datax = []
        datay = []
        dataz = []
        colorslist = ["#0000FF", "#008000", "#800080", "#FFA500", "#000080", "#800000", "#00FFFF", "#FF00FF", "#808000", "#008080", 
                "#0000A0", "#8000A0", "#00A080", "#808080", "#006400", "#8B0000", "#2F4F4F", "#800080", "#00008B", "#8B008B", 
                "#008B8B", "#696969", "#556B2F", "#4B0082", "#483D8B",
                "#DC143C", "#8B4513", "#1E90FF", "#D2691E", "#CD5C5C", "#5F9EA0", "#7B68EE", "#B8860B", "#20B2AA", "#FF4500",
                "#ADFF2F", "#FF6347", "#7FFF00", "#DB7093", "#4682B4", "#9ACD32", "#40E0D0", "#6B8E23", "#FF8C00", "#00BFFF",
                "#B22222", "#228B22", "#BA55D3", "#CD853F", "#5D8AA8"]
        color_labels = []
        for i, branch_nodes in enumerate(self.branches['branch_nodes']):
            color = colorslist[i]
            x,y,z = find_xyz(branch_nodes, self.nodes_df)
            datax.extend(x)
            datay.extend(y)
            dataz.extend(z)
            color_labels.extend([color]*len(x))  # assigning color for each point

            text = text + f"<p style=\" color:{colorslist[i]};\">Branch number {i} with degree {self.branches.at[i,'branch_type']}</p>\n"

        data = np.array([np.array(datax),np.array(datay),np.array(dataz)])
        self.branches_description_text = text
        self.data = data.T
        self.ax.scatter(data[0], data[1], data[2], c=color_labels, picker=5)  # 5 points tolerance

    def select_remove(self):
        self.current_button = 1
    def select_name(self):
        self.current_button = 2

    def on_pick(self, event):
        """
        Function to handle the event when a point is clicked. Will either remove a branch or add a branch name to respective widgets. 

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            An event triggered when a data point is clicked.
        """
        if self.current_button is not None:
            ind = event.ind[0]  # Get the first index if multiple points are picked
            coords = self.data[ind]
            nodeid = int(str(int(coords[0])) + str(int(coords[1])) + str(int(coords[2])))
            print(f'You picked point {ind} at coordinates {coords} with {self.current_button}')

            if self.current_button == 1:

                for i, branch_nodes in enumerate(self.branches['branch_nodes']):
                    if nodeid in branch_nodes:
                        print(f"branch chosen is {i}")
                        branchcoords = find_xyz(branch_nodes, self.nodes_df)
                        self.ax.scatter(branchcoords[0],branchcoords[1],branchcoords[2], color='r',s=50, alpha=0.5)
                        self.draw()
                        self.branches_list.append(i)
                        self.branches_widget.clear()
                        for b in self.branches_list:
                            self.branches_widget.addItem(str(b))

            if self.current_button == 2:
                for i, branch_nodes in enumerate(self.branches['branch_nodes']):
                    if nodeid in branch_nodes:
                        print(f"branch chosen is {i}")
                        if i not in self.named_branches:
                            self.named_branches[i] = ''
                        self.named_branches_widget.addItem(f"{i}: {self.named_branches[i]}")


class GUI2ApplicationWindow(QMainWindow):
    """
    This class represents the main application window and handles all GUI 
    operations such as setting up widgets, layouts, and buttons.
    
    Signals
    -------
    results_ready : pyqtSignal
        Signal to indicate that the window is ready to be closed and send back all the collected results.
    """
    results_ready = pyqtSignal(object)  # Create a custom signal
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        #create QListWidgets for branches lists, create 3D plot and toolbar
        self.named_branches_widget = QListWidget()
        self.named_branches_label = QLabel("Named Branches")
        self.named_branches  = {}
        self.branches_widget = QListWidget()
        self.branches_label = QLabel("Branches to be removed")
        self.plot_widget = GUI2Plot(self.branches_widget, self.named_branches_widget, self.named_branches)
        self.toolbar = NavigationToolbar(self.plot_widget, self)

        #create buttons that choose to remove and name
        self.branches_button = QPushButton('Remove Branch')
        self.branches_button.clicked.connect(self.plot_widget.select_remove)
        self.name_button = QPushButton('Name Branch')
        self.name_button_label = QLabel("Type branch name below:\n\n(main:[p,O1,O2,T1,T2], \nadd [child,grandchild] \nsuffix for tertiary/quaternary)")
        self.name_button.clicked.connect(self.plot_widget.select_name)
        self.name_branch_input = QLineEdit()
        # self.name_branch_input.textChanged.connect(self.update_branch_name)
        self.name_branch_input.returnPressed.connect(self.update_branch_name)

        #create redo button
        self.redo_button = QPushButton('FAILED!!! \nGo back to point selection')
        self.redo_button.clicked.connect(self.goback_and_close)

        #create done button
        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.save_and_close)

        #create label for all the branches
        self.branches_description = QLabel("")

        #vertical layout with toolbar and plot
        vbox = QVBoxLayout()
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.plot_widget)
        

        #vertical layout with buttons and button text thing
        vbox4 = QVBoxLayout()
        vbox41 = QVBoxLayout()
        vbox41.addWidget(self.name_button)
        vbox41.addWidget(self.name_button_label)
        vbox41.addWidget(self.name_branch_input)
        vbox41.setSpacing(10)
        vbox4.addWidget(self.branches_button)
        vbox4.addLayout(vbox41)
        vbox4.setSpacing(30)

        #vertical layout with description, redo button
        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.branches_description)
        vbox3.addWidget(self.redo_button)

        #vertical layout with removed branches, named branches, and done button
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.branches_label)
        vbox2.addWidget(self.branches_widget)
        vbox2.addWidget(self.named_branches_label)
        vbox2.addWidget(self.named_branches_widget)
        vbox2.addWidget(self.done_button)

        #horizontal layout with all the vertical boxes
        hbox = QHBoxLayout(self.main_widget)
        hbox.addLayout(vbox)
        hbox.addLayout(vbox4)
        hbox.addLayout(vbox3)
        hbox.addLayout(vbox2)  
    
    def set_branches_text(self):
        """
        Function to set the text for the branches_description QLabel with the 
        current branches description text from the plot widget.
        """
        self.branches_description.setTextFormat(Qt.RichText)
        self.branches_description.setText(self.plot_widget.branches_description_text)
    
    def update_branch_name(self):
        """
        Function to update the name of the selected branch in the named branches list.
        """
        if self.plot_widget.current_button ==2 and self.named_branches_widget.count() > 0:
            branch_text = self.named_branches_widget.item(self.named_branches_widget.count()-1).text()
            branch_number = int(branch_text.split(':')[0])
            self.named_branches[branch_number] = self.name_branch_input.text()
            self.named_branches_widget.item(self.named_branches_widget.count() - 1).setText(f"{branch_number}: {self.name_branch_input.text()}")
            self.name_branch_input.clear()

    def goback_and_close(self):
        """
        Function to close the current window and signal that user wants to return 
        to the previous stage.
        """
        self.results = {
            'continue' : False,
        }
        self.results_ready.emit(self.results)  # Emit the signal with the results
        self.close()

    def save_and_close(self):
        """
        Function to save the current state, close the window, and signal that 
        user wants to proceed to the next stage.
        """
        def str_to_array(array_str):
            return np.array(list(map(float, array_str.strip('[]').split())))
            
        # Convert the lists of strings to lists of coordinates
        self.branch_remove_list = [str_to_array(item.text()) for item in self.branches_widget.findItems("*",Qt.MatchWildcard)]
        self.branch_names = [str(item.text()) for item in self.named_branches_widget.findItems("*",Qt.MatchWildcard)]
        
        self.results = {
            'continue' : True,
            'branch_remove_list': self.branch_remove_list,
            'branch_names': self.branch_names,
        }
        self.results_ready.emit(self.results)  # Emit the signal with the results
        self.close()


def GUI2(branches : pd.DataFrame,
         nodes_df : pd.DataFrame):
    """
    Main function to run the GUI application.

    Parameters
    ----------
    branches : pd.DataFrame
        A DataFrame containing information about the branches.
    nodes_df : pd.DataFrame
        A DataFrame containing information about the nodes.

    Returns
    -------
    dict
        A dictionary containing the results after the user closes the window.
    """

    app = QApplication.instance()  # checks if QApplication already exists
    if not app:  # create QApplication if it doesnt exist 
        app = QApplication(sys.argv)
        
    window = GUI2ApplicationWindow()
    window.show()
    window.plot_widget.plot(branches, nodes_df)
    window.set_branches_text()
    
    data = {}

    def receive_results(results):
        data.update(results)
        app.quit()

    window.results_ready.connect(receive_results)  # Connect the signal to the receiving function
    app.exec_()  # Start the application event loop

    return data