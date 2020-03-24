import sys
import os
import re
import sip
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtRemoveInputHook
from pdb import set_trace
from LabledSlider import LabledSlider

import h5py

import matplotlib.pylab as plt
import matplotlib as mpl 
import matplotlib.ticker as plticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.backends.backend_qt5agg import(FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
# The window in which the figure options can be modified has some weird properties
# this is predefined by matplotlib. To get rid of those (e.g. if legend labels
# are updated, the legend design changes back) the figureoptions file is overwritten.
import figureoptions_debugging as local_figureoptions
import matplotlib.backends.backend_qt5
matplotlib.backends.backend_qt5.figureoptions = local_figureoptions
import functools
# from mpldatacursor import datacursor
from matplotlib.backend_bases import key_press_handler
import warnings
import matplotlib.dates as mdates
# import mplcursors

from collections import Counter

import numpy as np
import os
import pandas as pd
import glob
from scipy.optimize import curve_fit

# Define several natural constants
hPlanck = 6.626 * 10**(-34)
eElectron = 1.602 * 10**(-19)
cLight = 3 * 10**8
kBoltzmann = 1.38065 * 10**(-23)
TRoom = 273.15 + 21     # For debugging
# TRoom = 250 

#-------------------------------------------------------------------------------
#----------------------------- main_window CLASS ------------------------------- 
#-------------------------------------------------------------------------------
class main_window(QtWidgets.QMainWindow):
    # Constructor

    def __init__(self):
        # Initialize window and several variables

        super(main_window, self).__init__()
        self.initUI()

        # Counter needed to do the fitting of the UVVis-graphs
        self.graph_counter = 0
        # Set global folder for plotting
        self.globalPath = os.getcwd()
        # Counter to change from log to normal by pushing button in UVVis graphs
        self.isWhat = "Nothing" 
        self.isnorm = False
        self.IVcalled = False 
        # Arrays in which the device, pixel and measurements number is stored
        self.nbs = np.empty([0, 3])
        self.nbsDIV = np.empty([0, 3], dtype = "object")
        self.IVlabels = np.empty(0, dtype = "object")
        self.DIVlabels = np.empty(0, dtype = "object")
        self.EQElabels = np.empty(0, dtype = "object")
        self.performance_data = np.empty(0, dtype = "object")
        self.devNbisNb = None

        self.multipleFoldersLoaded = False 

        # Initialize the group Colors array with 12 colors for 12 different
        # groups. If more groups are created colors have to be selected
        # manually
        self.colormap = mpl.cm.get_cmap('inferno', 6).colors[:, 0:3:1]
        self.GroupColor = np.array([mpl.colors.rgb2hex(self.colormap[i, :]) 
            for i in range(np.shape(self.colormap)[0])], dtype = '|S7')
        self.IVselected = np.empty(0, dtype = "bool")
        self.DIVdata = np.empty(0)
        self._nbGroups = 0
        self.scan_nb = 1
        self.nbOfScans = 1
        self.GroupNames = np.empty(0)
        self.GroupNamesGlobal = np.empty(0)
        self.ThicknessLayer = np.empty(0)
        self.groupsAssigned = False
        # self.IVdata
        # self.IVlabels
        # Choose color map for plots
        # self.cmap = plt.get_cmap('viridis')

        self.linew = 5
        self.ax_ticks_width = 2
        self.tick_length = 5
        self.legend_pos = "best"
        self.fonts_ = 18

        self.isDual = None


    def initUI(self):
        # Initialize the user interface with all its buttons etc.

        ## MAIN WINDOW
        self.center()
        self.setWindowTitle('PV Evaluation Tool')
        self.setWindowIcon(QtGui.QIcon('icons/pv_logo.png'))
        self.show()

        # Define actions
        self.plotIVAction = QtWidgets.QAction("IV", self)
        self.plotEQEAction = QtWidgets.QAction("EQE", self)
        self.plotUVVisAction = QtWidgets.QAction("UVVis", self)
        self.plotTPVTPCAction = QtWidgets.QAction("TPV/TPC", self)
        self.mobilityMeasurementAction = QtWidgets.QAction("\u03BC", self)
        self.plotUSRAction = QtWidgets.QAction("USR", self)
        self.plotELQEAction = QtWidgets.QAction("ELQE", self)
        self.plotLogarithmicAction = QtWidgets.QAction("Log", self)
        log = QtWidgets.QAction("LOG", self)
        help = QtWidgets.QAction("?", self)

        # Define all the things needed for the mainWidget (which is a plot window)
        figureSize = (11, 10)
        # figureSize = plt.figaspect(1.1)
        self.fig = FigureCanvas(Figure(figsize = figureSize))
        self.setCentralWidget(self.fig)

        # Define axis
        self._ax = self.fig.figure.subplots()
        self.mplToolbar = NavigationToolbar(self.fig, self)
        self.mplToolbar.addAction(self.plotLogarithmicAction)

        self.addToolBar(self.mplToolbar)

        ## LOCAL VARIABLES FOR ACTIONS

        # MENU
        self.saveAction = QtWidgets.QAction(QtGui.QIcon('icons/save_as.png'), 'Save', self)
        self.changeFilePathAction = QtWidgets.QAction(QtGui.QIcon('icons/select_folder.png'), 'Change File Path', self)
        self.UVVisFittingAction = QtWidgets.QAction(QtGui.QIcon('icons/fit_graph.png'), 'Fit Onset', self)
        self.normalizeUVVisAction = QtWidgets.QAction(QtGui.QIcon('icons/normalize_UVVis.png'), 'Normalize UV-Vis Spectra', self)
        self.assignGroupAction = QtWidgets.QAction(QtGui.QIcon('icons/assign_group.png'), 'Assign Group', self)
        self.showGroupAction = QtWidgets.QAction(QtGui.QIcon('icons/show_graphs.png'), 'Show Group of IVs', self)
        self.extractMobilityAction = QtWidgets.QAction(QtGui.QIcon('icons/calculator.png'), 'Calculate Mobility', self)
        self.showOverviewAction = QtWidgets.QAction(QtGui.QIcon('icons/show_overview.png'), 'Show Overview over Performance Parameters', self)
        self.showHeroDevicesAction = QtWidgets.QAction(QtGui.QIcon('icons/hero_dev.png'), 'Show Hero Devices', self)
        self.updateLegendAction = QtWidgets.QAction(QtGui.QIcon('icons/update_legend2.png'), 'Update the Legend', self)
        self.transposeXAction = QtWidgets.QAction(QtGui.QIcon('icons/transposeX.png'), 'Fit Onset', self)
        self.TPVAction = QtWidgets.QAction(QtGui.QIcon('icons/TPV.png'), 'Plot TPV', self)
        self.TPCAction = QtWidgets.QAction(QtGui.QIcon('icons/TPC.png'), 'Plot TPC', self)
        self.intDepIVAction = QtWidgets.QAction(QtGui.QIcon('icons/intensity_dep.png'), 'Plot Intensity Dependent IV', self)

        #TOOLBAR

        ## TOOLBAR

        self.toolbar = self.addToolBar("Toolbar1")

        # GENERAL LAYOUT
        self.toolbar.setIconSize(QtCore.QSize(50,50)) # (length,hight)
        #toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon|QtCore.Qt.AlignLeading)
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolbar)

        # BUTTONS
        self.toolbar.addAction(self.changeFilePathAction)
        self.toolbar.addSeparator()

                ### DESIGN ###

        ## TOOLBAR BOTTOM
        toolbar2 = self.addToolBar("Toolbar2")
        toolbar2.setIconSize(QtCore.QSize(20,20)) # (length,hight)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar2)
        toolbar2.addAction(self.plotIVAction) 
        toolbar2.addSeparator()
        toolbar2.addAction(self.plotEQEAction)
        toolbar2.addSeparator()
        toolbar2.addAction(self.plotUVVisAction)
        toolbar2.addSeparator()
        toolbar2.addAction(self.plotTPVTPCAction)
        toolbar2.addSeparator()
        toolbar2.addAction(self.mobilityMeasurementAction)
        toolbar2.addSeparator()
        toolbar2.addAction(self.plotELQEAction)
        toolbar2.addSeparator()
        toolbar2.addAction(self.plotUSRAction)
        toolbar2.addSeparator()
        toolbar2.addAction(log)
        toolbar2.addSeparator()
        toolbar2.addAction(help)

        # Set all actions to not true at the beginning until a directory is selected
        self.plotIVAction.setEnabled(False)
        self.plotEQEAction.setEnabled(False)
        self.plotUVVisAction.setEnabled(False)
        self.plotELQEAction.setEnabled(False)
        self.plotTPVTPCAction.setEnabled(False)
        self.mobilityMeasurementAction.setEnabled(False)

        ## STATUSBAR

        self.statusBar().showMessage("Welcome: Select a Folder from which Data shall be read", 30000) 

        # STATUS TIPS FOR BUTTONS
        # plotIVAction.setStatusTip('IV Data')
        # plotEQEAction.setStatusTip('EQE Data')
        # plotUVVisAction.setStatusTip('UVVis Data')
        # plotUSRAction.setStatusTip('User Selected Folder')
        # plotTPVTPCAction.setStatusTip('Plot TPV/ TPC spectra')

        ### ACTIONS ###


        ## LINKS BETWEEN BUTTONS AND FUNCTIONS
        self.plotIVAction.triggered.connect(self.load_IV)
        self.plotEQEAction.triggered.connect(self.load_EQE)
        self.plotUVVisAction.triggered.connect(self.load_UVVIS)
        self.plotELQEAction.triggered.connect(self.load_ELQE)
        self.plotUSRAction.triggered.connect(self.plot_USR)
        self.plotLogarithmicAction.triggered.connect(self.plot_log)
        self.mobilityMeasurementAction.triggered.connect(self.mobilityMeasurement)
        self.plotTPVTPCAction.triggered.connect(self.load_TPVTPC)
        self.assignGroupAction.triggered.connect(self.assignGroup_dialog)
        self.showGroupAction.triggered.connect(self.showGroup_dialog)
        self.extractMobilityAction.triggered.connect(self.extractMobility)
        self.showOverviewAction.triggered.connect(self.showOverview_dialog)
        self.showHeroDevicesAction.triggered.connect(functools.partial(self.showHeroDevices, "group"))
        self.UVVisFittingAction.triggered.connect(self.fit_UVVis_dialog)
        self.TPVAction.triggered.connect(self.TPV_dialog)
        self.TPCAction.triggered.connect(self.TPC_dialog)
        self.transposeXAction.triggered.connect(self.transposeX_dialog)
        self.normalizeUVVisAction.triggered.connect(self.plot_UVVisNorm)
        self.updateLegendAction.triggered.connect(self.updateLegend)
        self.saveAction.triggered.connect(self.saveFile)
        self.changeFilePathAction.triggered.connect(self.changeFilePath)
        self.intDepIVAction.triggered.connect(self.plot_intDepIV)



        ## SHORTCUTS
        self.plotIVAction.setShortcut('Ctrl+I')
        self.plotEQEAction.setShortcut('Ctrl+E')
        self.plotUVVisAction.setShortcut('Ctrl+U')
        self.plotELQEAction.setShortcut('Ctrl+U')

        # filepath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Open File"))

    # -----------------------------------------------------
    # ----------------- Function Section ------------------
    # -----------------------------------------------------
    
    # -----------------------------------------------------
    # ----------- Section for Generic Functions -----------
    # -----------------------------------------------------

    def readData(self, files, names_, plotVars_, skipr_ = []):
        # Function that allows for generic reading in of data. It does read in
        # the whole file but only return the data that was specified in plotVars_
        # that is two dimensional. Like this only x and y data for easy
        # plotting is returned.

        # Read in the array with all its strings and empty lines
        try:
            file_data_list = pd.read_csv(files[0], skiprows = 0, skipfooter = 0,
                    sep = "\t", names = names_, engine = "python", skip_blank_lines = False)
            x_data = np.array(file_data_list[plotVars_[0]])
        except:
            warnings.warn("Could not load example data to figure out the file size.", category = UserWarning)
            return

        # If user did not give information about skipping rows, figure out which
        # rows do not contain numbers (real information)
        try:
            if np.size(skipr_) == 0:
                # Check how many rows have to be jumped at the beginning (header)
                skipr = 0
                for k in range(np.size(x_data)):
                    try:
                        break
                    except:
                        skipr = k + 1

                # and at the end (footer)
                skipf = 0
                for k in reversed(range(np.size(x_data))):
                    try:
                        break
                    except:
                        skipf = np.size(x_data) - k
            else:
                # User defined rows to skip
                skipr = skipr_[0]
                skipf = skipr_[1]
        except:
            warnings.warn("Could not figure out how many rows should be skipped", category = UserWarning)
        
        if skipr == np.size(x_data):
            warnings.warn("The data is not valid (e.g.separated by ,)" + 
                    "and can't be loaded", category = UserWarning)
            return

        # Declare arrays
        dataStorage = []
        graphLabels = np.empty(np.size(files), dtype = object)

        # Actual section to read in files
        idx = 0

        for filename in files:
            try:
                # Here it is checked how many rows are header/ footer so that the user
                # does not have to give this information as input.
                # Only the first file is checked so all files should have the same format
                # that are plotted together

                file_data_list = pd.read_csv(filename, skiprows = skipr, skipfooter = skipf,
                        sep = "\t", names = names_, engine = "python", skip_blank_lines = False)

                for nbPltVars in range(np.size(plotVars_)):
                    dataStorage.append(file_data_list[plotVars_[nbPltVars]])

                if np.size(dataStorage[-1]) == 0 or np.size(dataStorage[-2]) == 0:
                    warnings.warn("Couldn't load file " + str(filename) + 
                            " please check that data format matches the required format.",
                            category = UserWarning)
                    continue

                # idx/2 is always an integer by definition but this must be told to
                # python
                graphLabels[int(idx / 2)] = os.path.splitext(os.path.basename(filename))[0]

                # Increase index by two since there is always x and y data
                idx += 2
            except:
                warnings.warn("Couldn't load file " + str(filename) + 
                        " please check that data format matches the required format.",
                        category = UserWarning)

        # print(np.asarray(dataStorage))
        return np.asarray(dataStorage), graphLabels 


    def plot(self, data, labels_, xylabel_, tickInterval_,
            xlim_ = None, ylim_ = None, zeroLines_ = False, log_ = False,  
            norm_ = False, yticksoff_ = False, IV_ = False, linestyle_ = "-", marker_ = "",
            errbar_ = False, ax_ = False, minorTicks_ = True, color_ = "b", gridOn_ = True):
        # Function that does basic plotting of next to everything
        # it should be maintained as modular as possible

        # Enable several specifications by using if statements
        # Plot vertical and horizontal line to indicate zero
        if zeroLines_:
            ax_.axvline(x=0, color = "black", linewidth = self.ax_ticks_width)
            ax_.axhline(y=0, color = "black", linewidth = self.ax_ticks_width)

        # Logarithmic yscale
        if log_:
            ax_.set_yscale("log")

        if yticksoff_:
            ax_.set_yticklabels([])

        # Read in data and do plotting
        self.idx = 0

        if errbar_ == False:
            fileSz = 2
        else:
            fileSz = 3

        for dataNb in range(int(len(data) / fileSz)):
            if errbar_ == False:
                try:
                    self.line["line" + str(dataNb)], = ax_.plot(data[self.idx], data[self.idx+1],
                            linewidth = self.linew, label = labels_[dataNb], 
                            color = self.GroupColor.astype('U7')[
                                np.where(self.GroupNames.astype('U13') == labels_[dataNb])][0],
                            marker = marker_, linestyle = linestyle_)
                except:
                    self.line["line" + str(dataNb)], = ax_.plot(data[self.idx], data[self.idx+1],
                            linewidth = self.linew, label = labels_[dataNb], marker = marker_,
                            linestyle = linestyle_)
            else:
                try:
                    self.line["line" + str(dataNb)] = ax_.errorbar(data[self.idx], 
                            data[self.idx + 1], yerr = data[self.idx + 2], fmt = marker,
                            capsize = self.linew, label = labels_[dataNb],
                            c = self.GroupColor.astype('U7')[
                                np.where(self.GroupNames.astype('U13') == labels_[dataNb])][0])
                except:
                    self.line["line" + str(dataNb)] = ax_.errorbar(data[self.idx], 
                            data[self.idx + 1], yerr = data[self.idx + 2], fmt = "s", 
                            capsize = self.linew)


            # Increase index by two since there is x and y data in every iteration
            self.idx += fileSz 

        # Set up a dict mapping legend line to origline, and enable
        # picking on the legend line
        self.lines = [self.line["line" + str(i)] for i in range(int(len(data)/ fileSz))]
        self.lined = dict()
        self.lineLabel = dict()

        ax_.autoscale(enable = True, tight = True)
        ax_.set_ylim(ax_.get_ylim()[0], ax_.get_ylim()[1] * 1.02)

        if gridOn_ == True:
            ax_.grid(True)

        if xlim_ != None:
            ax_.set_xlim(xlim_)
        if ylim_ != None:
            ax_.set_ylim(ylim_)

        # Matplotlib makes it quite inconvenient to have different tick directions
        # for top and bottom (left and right) axis. This is a go around I figured out.
        # It is, however, not very pretty but it works. The bottom and left axis get
        # twin axis that have the exact same parameters but with ticks pointing in.

        # Set several aesthetic parameters of the graph (should be always the same)
        ax_.set_xlabel(xylabel_[0], fontsize = self.fonts_)
        ax_.set_ylabel(xylabel_[1], fontsize = self.fonts_)
        loc = plticker.MultipleLocator(base=tickInterval_) # this locator puts ticks at regular intervals
        ax_.xaxis.set_major_locator(loc)
        # ax__top.xaxis.set_major_locator(loc)

        # Define ticks
        if minorTicks_ == True:
            minorLocator = AutoMinorLocator()
            ax_.xaxis.set_minor_locator(minorLocator)
            ax_.tick_params(which = "minor", direction = "in", bottom = True, top = True, length = self.tick_length/4*3, width= self.ax_ticks_width)
        # ax__top.xaxis.set_minor_locator(minorLocator)

        ax_.tick_params(width = self.ax_ticks_width, length = self.tick_length)
        ax_.tick_params(which = "major", direction = "in", bottom = True, top = True, width = self.ax_ticks_width)
        ax_.tick_params(which = "major", direction = "in", left = True, right = True, width = self.ax_ticks_width)

        for axis in ['top','bottom','left','right']:
          ax_.spines[axis].set_linewidth(self.ax_ticks_width)

        # Set tick fontsizes
        for tick in ax_.xaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)
        for tick in ax_.yaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)

        # Set tight layout
        # The try and except statement is because it onces helped out :D
        try:
            self.fig.figure.tight_layout()
        except:
            locator = mdates.MinuteLocator(byminute=[0,30])
            locator.MAXTICKS = 3000
            ax_.xaxis.set_major_locator(locator)
            warnings.warn("An error occured. The maximum tick number was increased so that you can" + 
                    " detect the error!", category = UserWarning)

        # Define legend which shall be draggable and by pressing on elements they hide

        if errbar_ == False:
            leg = ax_.legend(frameon = False, fontsize = self.fonts_, loc = self.legend_pos)
            leg.get_frame().set_alpha(0.4)
            leg.set_draggable(True) # Make it draggable
            # Set the legend's lines thickness
            for legline, origline, linelabel in zip(leg.get_lines(), self.lines, ax_.get_legend_handles_labels()[0]):
                legline.set_picker(5)  # 5 pts tolerance
                self.lined[legline] = origline
                self.lineLabel[legline] = linelabel

        # Connect the pick event to the function onpick()
        self.fig.mpl_connect('pick_event', self.onpick)

        def override(**kwargs):
            # kwargs['x'] = kwargs['left'] + 0.5 * kwargs['width']
            # kwargs['y'] = kwargs['bottom'] + kwargs['height']
            labelVoc = "Voc: " + str(np.round(self.performance_data[np.where(kwargs['label'] == self.IVlabels), 1][0][0], decimals = 2))
            labelJsc = "Jsc: " + str(np.round(self.performance_data[np.where(kwargs['label'] == self.IVlabels), 2][0][0], decimals = 2))
            labelFF = "FF: " + str(np.round(self.performance_data[np.where(kwargs['label'] == self.IVlabels), 3][0][0], decimals = 2))
            labelPCE = "PCE: " + str(np.round(self.performance_data[np.where(kwargs['label'] == self.IVlabels), 4][0][0], decimals = 2)) 

            kwargs['label'] = str(kwargs['label']) + "\n " + labelVoc + "\n " + labelJsc + "\n " + labelFF + "\n " + labelPCE 
            return kwargs

        datacursor(self.lines, props_override=override, formatter='{label}'.format,
                bbox = None, draggable = True)

        # Redraw the figure
        self.fig.draw()

        # The graph counter has to be reset after every plot, otherwise there
        # might be some problems with fitting of the UV-Vis curve
        self.graph_counter = 0


    def updateToolbar(self, whichTab_):
        # This function updates the toolbar according to which tab is called

        self.toolbar.clear()

        if whichTab_ == "IV":
            self.toolbar.addAction(self.assignGroupAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.showGroupAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.showOverviewAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.showHeroDevicesAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.intDepIVAction)

        elif whichTab_ == "UVVis":
            self.toolbar.addAction(self.normalizeUVVisAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.UVVisFittingAction)

        elif whichTab_ == "TPVTPC":
            self.toolbar.addAction(self.TPVAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.TPCAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.transposeXAction)

        elif whichTab_ == "mobility":
            self.toolbar.addAction(self.assignGroupAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.showGroupAction)
            self.toolbar.addSeparator()
            self.toolbar.addAction(self.extractMobilityAction)
            self.showGroupAction.setEnabled(False)
            self.extractMobilityAction.setEnabled(False)

        self.toolbar.addSeparator()
        self.toolbar.addAction(self.updateLegendAction)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.changeFilePathAction)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.saveAction)
        self.toolbar.addSeparator()

        return


    def updateLegend(self):
        # Function that enables an updating of the legend so that only graphs
        # selected as visible are shown in the legend.


        handles_new_leg = np.empty(0)
        labels_new_leg = np.empty(0)
        for leg_elem in range(len(self._ax.get_legend_handles_labels()[0])):
            if self._ax.get_legend_handles_labels()[0][leg_elem].get_visible():
                handles_new_leg = np.append(handles_new_leg, self._ax.get_legend_handles_labels()[0][leg_elem])
                labels_new_leg = np.append(labels_new_leg, self._ax.get_legend_handles_labels()[1][leg_elem])

        # If the legends labels were updated, change their color according to
        # the colormap that was set in the IV dialog
        # The hex color has to be converted to rgb first to be able to change
        # the color
        for k in range(len(self._ax.get_legend_handles_labels()[0])):
            # print(labels_new_leg[k])
            # print(self.GroupNames)
            # print(np.where(self.GroupNames.astype('U13') == labels_new_leg[k]))
            if self._ax.get_legend_handles_labels()[1][k] in self.GroupNames.astype('U13'):
                elemNb = np.where(self.GroupNames.astype('U13') == self._ax.get_legend_handles_labels()[1][k])[0][0]
                print(elemNb)
                print(self.GroupColor.astype('U7'))
                h = self.GroupColor.astype('U7')[elemNb].lstrip('#')
                self.line["line" + str(k)].set_color(tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4)))

        fonts = 18

        # self._ax.set_legend_handles() = handles_new_leg
        # self._ax.set_legend_handles_labels() = labels_new_leg
        leg = self._ax.legend(handles_new_leg, labels_new_leg, frameon = False, fontsize = fonts)

        # Define legend which shall be draggable and by pressing on elements they hide
        leg.get_frame().set_alpha(0.4)
        leg.draggable() # Make it draggable

        # Set the legend's lines thickness
        for legline, origline in zip(leg.get_lines(), self.lines):
            legline.set_picker(5)  # 5 pts tolerance
            self.lined[legline] = origline

        self.fig.draw()

        return


    def onpick(self, event):
        # Function that allows to cut out graphs by clicking on the legend

        if event.mouseevent.button == 1:
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = self.lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            temp = None

            # pyqtRemoveInputHook()
            # set_trace()
            try:
                if self.isWhat == "IV":
                    regex = re.compile(r'\d+')
                    extracted_nbs = [int(x) for x in regex.findall(str(self.lineLabel[legline]))]
                    extracted_nbs = np.array([extracted_nbs[1], extracted_nbs[2], extracted_nbs[3]])

                    # There must be a numpy way to do this much faster but I don't know it yet
                    # so this solution works
                    for i in range(np.shape(self.nbs)[0]):
                        if np.array_equal(self.nbs[i][[0,1]], extracted_nbs[[0,1]]):
                            temp = i

                if self.isWhat == "mobility":

                    regex = re.compile(r'\d+')

                    if self.devNbisNb:
                        extracted_nbs = [int(x) for x in regex.findall(str(self.lineLabel[legline]))]
                        extracted_nbs = np.array([extracted_nbs[1], extracted_nbs[2], extracted_nbs[3]])
                    else:
                        extracted_nbs_temp = [str(x) for x in regex.findall(re.search('p(.*)', str(self.lineLabel[legline].get_label()))[0])]
                        s = str(self.lineLabel[legline].get_label())
                        result = re.search('d(.*)p' + str(extracted_nbs_temp[0]), s)[1]
                        extracted_nbs = np.array([str(result), extracted_nbs_temp[0], extracted_nbs_temp[1]]) 

                    # extracted_nbs = [int(x) for x in regex.findall(str(self.lineLabel[legline]))]
                    # extracted_nbs = np.array([extracted_nbs[1], extracted_nbs[2], extracted_nbs[3]])

                    # There must be a numpy way to do this much faster but I don't know it yet
                    # so this solution works
                    for i in range(np.shape(self.nbsDIV)[0]):
                        if np.array_equal(self.nbsDIV[i][[0,1]], extracted_nbs[[0,1]]):
                            temp = i
                    # pyqtRemoveInputHook()
                    # set_trace()

                    if temp == None:
                        warnings.warn("Error the graph could not be traced back", category = UserWarning)
                    if np.size(temp) > 1:
                        print(temp)
                        warnings.warn("Error more than one graph was selected", category = UserWarning)

            except:
                warnings.warn("Error happened trying to extract nbs of legend when picking a graph.", category = UserWarning)

            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled also change the selected item to true/ false
            if vis:
                legline.set_alpha(1.0)
                try:
                    if self.isWhat == "IV" or self.iWhat == "mobility":
                        # Now set the picked element to 
                        self.IVselected[temp] = True
                except:
                    warnings.warn("Though the IV window is selected, a graph with a label other than its device, pixel and scan number was picked.", category = UserWarning)
            else:
                legline.set_alpha(0.2)
                try:
                    if self.isWhat == "IV" or self.isWhat == "mobility":
                        # Now set the picked element to 
                        self.IVselected[temp] = False
                except:
                    warnings.warn("Though the IV window is selected, a graph with a label other than its device, pixel and scan number was picked.", category = UserWarning)
        
            self.fig.draw()

        elif event.mouseevent.button == 3:
            if self.isWhat == "IV":
                legline = event.artist
                self.line = {}
                # Clear plot (necessary to plot new stuff)
                self.fig.figure.clf()
                self.fig.draw()

                regex = re.compile(r'\d+')
                extracted_nbs = [int(x) for x in regex.findall(str(self.lineLabel[legline]))]
                extracted_nbs = np.array([extracted_nbs[1], extracted_nbs[2], extracted_nbs[3]])

                # There must be a numpy way to do this much faster but I don't know it yet
                # so this solution works
                temp = None
                for i in range(np.shape(self.nbs)[0]):
                    if np.array_equal(self.nbs[i][[0,1]], extracted_nbs[[0,1]]):
                        temp = i

                dataToPlot = np.array([self.IVdata[2 * temp][:], self.IVdata[2 * temp + 1][:], self.DIVdata[2 * temp][:], self.DIVdata[2 * temp + 1][:]])
                labels = [str(extracted_nbs), str(extracted_nbs) + "DIV"]
                xylabel = ["Voltage (V)", "Current (mA/cm$^2$)"]
                tickInterval = 0.4
                self._ax = self.fig.figure.subplots()
                self.plot(dataToPlot, labels, xylabel, tickInterval, ax_ = self._ax,
                        IV_ = True, zeroLines_ = True)

                self._ax.set_ylim(self._ax.get_ylim()[0] * 1.01, 8)
                self.fig.draw()

            if self.isWhat == "mobility":
                legline = event.artist
                regex = re.compile(r'\d+')

                # pyqtRemoveInputHook()
                # set_trace()
                # Again account for the case that the entered device name is a string
                if self.devNbisNb:
                    extracted_nbs = [int(x) for x in regex.findall(str(self.lineLabel[legline]))]
                    extracted_nbs = np.array([extracted_nbs[1], extracted_nbs[2], extracted_nbs[3]])
                else:
                    # pyqtRemoveInputHook()
                    # set_trace()
                    extracted_nbs_temp = [str(x) for x in regex.findall(re.search('p(.*)', str(self.lineLabel[legline].get_label()))[0])]
                    s = str(self.lineLabel[legline].get_label())
                    result = re.search('d(.*)p' + str(extracted_nbs_temp[0]), s)[1]
                    extracted_nbs = np.array([str(result), extracted_nbs_temp[0], extracted_nbs_temp[1]]) 

                # extracted_nbs = np.array([extracted_nbs[1], extracted_nbs[2], extracted_nbs[3]])

                # There must be a numpy way to do this much faster but I don't know it yet
                # so this solution works
                temp = None
                for i in range(np.shape(self.nbsDIV)[0]):
                    if np.array_equal(self.nbsDIV[i][[0,1]], extracted_nbs[[0,1]]):
                        temp = i

                if temp == None:
                    warnings.warn("Error the graph could not be traced back", category = UserWarning)

                try:
                    # Only plot the fit in the regime where it was done so that
                    # the user immediately sees where it was done
                    dataToPlot = np.array([self.DIVdata[2 * temp][:], self.DIVdata[2 * temp + 1][:], 
                        self.DIVdata[2 * temp][np.where(self.DIVdata[2 * temp] == self.x_min[temp])[0][0]:np.where(self.DIVdata[2 * temp] == self.x_max[temp])[0][0]],
                        self.slopeMobility[temp] * (self.DIVdata[2 * temp][np.where(self.DIVdata[2 * temp] == self.x_min[temp])[0][0]:np.where(self.DIVdata[2 * temp] 
                            == self.x_max[temp])[0][0]]- self.x_new[temp]) ** 2 + self.y_new[temp]])

                    self.line = {}
                    self.fig.figure.clf()
                    self.fig.draw()
                    labels = [str(extracted_nbs), "Quadratic Fit"]
                    xylabel = ["Voltage (V)", "Current (mA/cm$^2$)"]
                    tickInterval = 0.4
                    self._ax = self.fig.figure.subplots()
                    self.plot(dataToPlot, labels, xylabel, tickInterval, ax_ = self._ax,
                            IV_ = True, zeroLines_ = True)
                    self.fig.draw()
                except:
                    warnings.warn("There is no quadratic regime in this graph it was thus not fitted", category = UserWarning)


    def changeFilePath(self):
        # Dialog that allows the user to change the file path within the program
        # one or several folders can be selected. 

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        file_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        file_view = file_dialog.findChild(QtWidgets.QListView, "listView")

        self.paths = []
        if file_view:
            file_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QtWidgets.QTreeView)

        if f_tree_view:
            f_tree_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            self.paths = file_dialog.selectedFiles()

        if np.size(self.paths) == 1:
            self.globalPath = self.paths[0]

            # If the according directories exist in the global directory set it to true
            if os.path.isdir(self.globalPath + "/IV/") == True: 
                self.plotIVAction.setEnabled(True)
            else:
                self.plotIVAction.setEnabled(False)

            if os.path.isdir(self.globalPath + "/EQE/") == True: 
                self.plotEQEAction.setEnabled(True)
            else:
                self.plotEQEAction.setEnabled(False)

            if os.path.isdir(self.globalPath + "/UV-Vis/") == True: 
                self.plotUVVisAction.setEnabled(True)
            else:
                self.plotUVVisAction.setEnabled(False)

            if os.path.isdir(self.globalPath + "/ELQE/") == True: 
                self.plotELQEAction.setEnabled(True)
            else:
                self.plotELQEAction.setEnabled(False)

            if os.path.isdir(self.globalPath + "/TPVTPC/") == True: 
                self.plotTPVTPCAction.setEnabled(True)
            else:
                self.plotTPVTPCAction.setEnabled(False)

            if os.path.isdir(self.globalPath + "/mobility/") == True: 
                self.mobilityMeasurementAction.setEnabled(True)
            else:
                self.mobilityMeasurementAction.setEnabled(False)

        elif np.size(self.paths) > 1:
            print(self.paths)


        # self.globalPath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

        # Set all variables about the old groups to their base values again
        self.GroupNames = np.empty(0)
        self.GroupNamesGlobal = np.empty(0)
        self.GroupNbs = np.empty(0)
        self.DIVdata = np.empty(0)
        self.IVdata = np.empty(0)
        self.IVcalled = False

        if np.size(self.paths) == 1:
            self.multipleFoldersLoaded = False
            # Now check if there is a save file already
            file_name = self.globalPath + "/.evaluationData.h5"

            if os.path.isfile(file_name) == True: 
                # Read the stored data 
                f = h5py.File(file_name, "r")

                try:
                    self.GroupNames = np.array(f["GroupNames"])
                    self.GroupNbs = np.array(f["GroupNbs"])
                    self.IVselected = np.array(f["IVselected"])
                    self.GroupColor = np.array(f["GroupColor"])
                    self.scan_nb = np.array(f["scan_nb"])
                    # Voc_dev = np.array(f["Voc_dev"])
                    # Jsc_dev = np.array(f["Jsc_dev"])
                    # FF_dev = np.array(f["FF_dev"])
                    # PCE_dev = np.array(f["PCE_dev"])
                    # Voc_dev_rev = np.array(f["Voc_dev_rev"])
                    # Jsc_dev_rev = np.array(f["Jsc_dev_rev"])
                    # FF_dev_rev = np.array(f["FF_dev_rev"])
                    # PCE_dev_rev = np.array(f["PCE_dev_rev"])
                    self.GroupNamesGlobal = np.array(f["GroupNamesGlobal"])
                    print("File successfully loaded")
                except:
                    print("Couldn't load file (maybe the save file was changed)")

                f.close()

                self.showHeroDevicesAction.setEnabled(True)
                self.groupsAssigned = True

                if os.path.isdir(self.globalPath + "/IntDepIV/") == True: 
                    self.intDepIVAction.setEnabled(True)
                else:
                    self.intDepIVAction.setEnabled(False)

            else:
                self.intDepIVAction.setEnabled(False)
                self.showHeroDevicesAction.setEnabled(False)
                self.groupsAssigned = False
                self.GroupColor = np.array([mpl.colors.rgb2hex(self.colormap[i, :]) 
                    for i in range(np.shape(self.colormap)[0])], dtype = '|S7')
                self.IVselected = np.empty(0, dtype = "object")

            if self.isWhat == "IV":

                self.load_IV()

            self.assignGroupAction.setEnabled(True)
            self.showGroupAction.setEnabled(True)
            self.updateLegendAction.setEnabled(True)
            self.saveAction.setEnabled(True)

        elif np.size(self.paths) > 1:
            self.multipleFoldersLoaded = True
            GroupNamesGlobalAll = np.empty(0)
            GroupColorAll = np.empty(0)
            Voc_dev_all = np.empty(0)
            Jsc_dev_all = np.empty(0)
            FF_dev_all = np.empty(0)
            PCE_dev_all = np.empty(0)
            Voc_dev_rev_all = np.empty(0)
            Jsc_dev_rev_all = np.empty(0)
            FF_dev_rev_all = np.empty(0)
            PCE_dev_rev_all = np.empty(0)
            batch_name_all = np.empty(0)

            for i in range(np.size(self.paths)):
                # Now check if there is a save file already
                file_name = self.paths[i] + "/.evaluationData.h5"

                if os.path.isfile(file_name) == True: 
                    # Read the stored data 
                    f = h5py.File(file_name, "r")

                    try:
                        # GroupNames = np.array(f["GroupNames"])
                        GroupNamesGlobalAll = np.append(GroupNamesGlobalAll, np.array(f["GroupNamesGlobal"]))
                        GroupColorAll = np.append(GroupColorAll, np.array(f["GroupColor"]))
                        # GroupNbs = np.array(f["GroupNbs"])
                        # IVselected = np.array(f["IVselected"])
                        # scan_nb = np.array(f["scan_nb"])
                        
                        # pyqtRemoveInputHook()
                        # set_trace()
                        # batch_name_all = np.append(batch_name_all, np.repeat(f["data_set_path"], np.size((f["Voc_dev"]))))
                        Voc_dev_all = np.append(Voc_dev_all, np.array(f["Voc_dev"]))
                        Jsc_dev_all = np.append(Jsc_dev_all, f["Jsc_dev"])
                        FF_dev_all = np.append(FF_dev_all, f["FF_dev"])
                        PCE_dev_all = np.append(PCE_dev_all, f["PCE_dev"])
                        Voc_dev_rev_all = np.append(Voc_dev_rev_all, f["Voc_dev_rev"])
                        Jsc_dev_rev_all = np.append(Jsc_dev_rev_all, f["Jsc_dev_rev"])
                        FF_dev_rev_all = np.append(FF_dev_rev_all, f["FF_dev_rev"])
                        PCE_dev_rev_all = np.append(PCE_dev_rev_all, f["PCE_dev_rev"])
                        print("File of folder " + str(self.paths[i]) + " successfully loaded")
                    except:
                        print("Couldn't load file of" + str(self.paths[i]))
            
                    if os.path.isfile(file_name + "/IntDepIV") == True:
                        self.intDepIVAction.setEnabled(True)

                    f.close()

            # print(GroupNamesGlobalAll)
            # print(np.unique(GroupNamesGlobalAll.astype('U13')))
            # c = Counter([i for j in GroupNamesGlobalAll for i in j])
            # print(c)
            # print(Voc_dev_all)
            # print(GroupNamesGlobalAll[0])
            self.GroupNames = np.unique(GroupNamesGlobalAll)[np.unique(GroupNamesGlobalAll).astype('U13') != "None"]
            self.GroupNamesGlobal = np.unique(GroupNamesGlobalAll)[np.unique(GroupNamesGlobalAll).astype('U13') != "None"]
            self.batch_name_all = batch_name_all
            self.Voc_all = []
            self.Jsc_all = []
            self.FF_all = []
            self.PCE_all = []
            self.Voc_rev_all = []
            self.Jsc_rev_all = []
            self.FF_rev_all = []
            self.PCE_rev_all = []

            self.plotIVAction.setEnabled(True)

            for unGrName in np.unique(GroupNamesGlobalAll.astype('U13')):
                # self.GroupColor = GroupColorAll[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)[0]]

                if unGrName == "None":
                    continue

                self.Voc_all.append(np.concatenate(Voc_dev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))
                self.Jsc_all.append(np.concatenate(Jsc_dev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))
                self.FF_all.append(np.concatenate(FF_dev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))
                self.PCE_all.append(np.concatenate(PCE_dev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))

                self.Voc_rev_all.append(np.concatenate(Voc_dev_rev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))
                self.Jsc_rev_all.append(np.concatenate(Jsc_dev_rev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))
                self.FF_rev_all.append(np.concatenate(FF_dev_rev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))
                self.PCE_rev_all.append(np.concatenate(PCE_dev_rev_all[np.where(GroupNamesGlobalAll.astype('U13') == unGrName)], axis = None))

            self._nbGroups = np.size(self.GroupNames)
            # pyqtRemoveInputHook()
            # set_trace()

            self.isDual = True
            self.groupsAssigned = True
            self.assignGroupAction.setEnabled(True)
            self.showGroupAction.setEnabled(False)
            self.updateLegendAction.setEnabled(False)
            self.saveAction.setEnabled(False)
            self.showHeroDevicesAction.setEnabled(True)
            

    def close_dialog(self, dialog):
        # Function that closes the dialogs when exit is pressed (all the dialogs)

        dialog.close()

    def center(self):
        # position and size of main window

        # self.showFullScreen()
        qc = self.frameGeometry()
        self.resize(600, 600)
        desktopWidget = QtWidgets.QApplication.desktop()
        PCGeometry = desktopWidget.screenGeometry()
        self.resize(PCGeometry.height(), PCGeometry.height())
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qc.moveCenter(cp)
        self.move(qc.topLeft())

    def saveFile(self):
        # Here the current status of the work with the data shall be stored.
        # This shall not only allow for an easier coming back but also to compare
        # several batches with one another and to accumulate data like that.
        # The hdf5 file type was chosen, because it is fast and it is easy to
        # handle with keys. If I add something to the final file there is no
        # problem with older saved files, because one can simply retrieve old
        # stuff using the keys.

        if self.isDual == True:
            Voc_dev, Jsc_dev, FF_dev, PCE_dev, Voc_dev_rev, Jsc_dev_rev, FF_dev_rev, PCE_dev_rev, labels_ = self.sortParams("groups") 
            Voc_dev_rev = np.asarray(Voc_dev_rev, dtype = object)
            Jsc_dev_rev = np.asarray(Jsc_dev_rev, dtype = object)
            FF_dev_rev = np.asarray(FF_dev_rev, dtype = object)
            PCE_dev_rev = np.asarray(PCE_dev_rev, dtype = object)
        else:
            Voc_dev, Jsc_dev, FF_dev, PCE_dev, labels_ = self.sortParams("groups") 

        Voc_dev = np.asarray(Voc_dev, dtype = object)
        Jsc_dev = np.asarray(Jsc_dev, dtype = object)
        FF_dev = np.asarray(FF_dev, dtype = object)
        PCE_dev = np.asarray(PCE_dev, dtype = object)

        # Writing a dataset
        file_name = self.globalPath + "/.evaluationData.h5"
        f = h5py.File(file_name, "w")
        # pyqtRemoveInputHook()
        # set_trace()
        f.create_dataset("data_set_path", data = self.paths[0], dtype = h5py.special_dtype(vlen=bytes))
        f.create_dataset("scan_nb", data = self.scan_nb)
        f.create_dataset("IVselected", data = self.IVselected)
        f.create_dataset("GroupNames", data = self.GroupNames, dtype = h5py.special_dtype(vlen=bytes))
        f.create_dataset("GroupNamesGlobal", data = self.GroupNamesGlobal, dtype = h5py.special_dtype(vlen=bytes))
        f.create_dataset("GroupNbs", data = self.GroupNbs, dtype = h5py.special_dtype(vlen=np.dtype('int32')))
        f.create_dataset("GroupColor", data = self.GroupColor, dtype = h5py.special_dtype(vlen=str))
        f.create_dataset("Voc_dev", data = Voc_dev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))
        f.create_dataset("Jsc_dev", data = Jsc_dev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))
        f.create_dataset("FF_dev", data = FF_dev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))
        f.create_dataset("PCE_dev", data = PCE_dev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))

        if self.isDual == True:
            f.create_dataset("Voc_dev_rev", data = Voc_dev_rev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))
            f.create_dataset("Jsc_dev_rev", data = Jsc_dev_rev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))
            f.create_dataset("FF_dev_rev", data = FF_dev_rev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))
            f.create_dataset("PCE_dev_rev", data = PCE_dev_rev, dtype = h5py.special_dtype(vlen=np.dtype('float64')))
        f.close()

        print("File successfully saved")
    def plot_log(self):
        self._ax.set_yscale("log")
        self._ax.set_xlim(1e-6, 1.2)
        self._ax.set_ylim(1e-5, 1e3)
        # self.cb.addItem("2")
        self.fig.draw()


    # -----------------------------------------------------
    # --- Section for Functions Accessible from IV Tab ----
    # -----------------------------------------------------

    def load_IV(self):
        # Function that loads the IV data in and does the first plotting

        # try:
        # Variable that says what tab this is
        if self.multipleFoldersLoaded == True:
            self.updateToolbar("IV")

            self.isWhat = "IV"
            self.IVcalled = True 
            return

        if self.IVcalled == True:

            self.showHeroDevices("device")

        else:
            # Read in performance parameters from the PV Parameters file
            try:
                filepath = self.globalPath + "/IV/"
                filename = filepath + "PV parameters Dual.txt"
                names_ = ["Data File Name", "Voc", "Jsc", "FF", "PCE", "Vmpp", "Impp", 
                        "Voc-b", "Jsc-b", "FF-b", "PCE-b", "Vmpp-b", "Impp-b"]
                file_data_list = pd.read_csv(filename, skiprows = 1, skipfooter = 0,
                        sep = "\t", names = names_, engine = "python")
                file_data_array = np.array(file_data_list)
            except:
                try:
                    filepath = self.globalPath + "/IV/"
                    filename = filepath + "PV parameters.txt"
                    names_ = ["Data File Name", "Voc", "Jsc", "FF", "PCE", "Vmpp", "Impp"]
                    file_data_list = pd.read_csv(filename, skiprows = 1, skipfooter = 0,
                            sep = "\t", names = names_, engine = "python")
                    file_data_array = np.array(file_data_list)
                except:
                    warnings.warn("Error while reading in PV parameters Dual. Please Check File for correctness.", category = UserWarning)
                    return

            # Read in files and save the necessary numbers
            files = np.empty(0)
            filesDIV = np.empty(0)
            dev_nb = 1
            scan_nbDIV = 1
            nbs_temp = np.empty([0, 3])
            nbsDIV_temp = np.empty([0, 3])
            IVlabels_temp = np.empty(0, dtype = "object")
            DIVlabels_temp = np.empty(0, dtype = "object")
            performance_data_temp = np.empty((0, np.size(names_)), dtype = "object")
            self.performance_data = np.empty((0, np.size(names_)), dtype = "object")

            files = np.array([f for f in glob.glob(filepath + "*_d*_" 
                + "*_IV_0" + str(self.scan_nb) + ".txt")])
            filesDIV = np.array([f for f in glob.glob(filepath + "*_d*_" 
                + "*_DIV_0" + str(scan_nbDIV) + ".txt")])
    
            # Iterate over nb of files
            for nb_files in range(len(files)):
                performance_data_temp = np.append(performance_data_temp, 
                        file_data_array[np.where(files[nb_files][5:].rsplit('/', 1)[-1] == file_data_list["Data File Name"])[0]],
                        axis = 0)
                # Extract pixel
                regex = re.compile(r'\d+')
                # extracted_nbs = [int(x) for x in regex.findall(files_temp[nb_files].rsplit('/', 1)[-1])]
                # pixel_nb = extracted_nbs[2] 
                pixel_nb = [int(x) for x in regex.findall(files[nb_files].rsplit('/', 1)[-1].rsplit('_', 5)[3])][0]
                try:
                    dev_nb = int(files[nb_files].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:])
                except:
                    dev_nb = files[nb_files].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:]
                nbs_temp = np.append(nbs_temp, [[dev_nb, pixel_nb, self.scan_nb]], axis = 0)
                IVlabels_temp = np.append(IVlabels_temp, "d" + str(dev_nb) + "p" + 
                    str(pixel_nb) + "s" + str(self.scan_nb))

            for nb_files in range(len(filesDIV)):
                # extracted_nbsDIV = [int(x) for x in regex.findall(files_tempDIV[nb_files].rsplit('/', 1)[-1])]
                # pixel_nbDIV = extracted_nbsDIV[2] 
                regex = re.compile(r'\d+')
                pixel_nbDIV = [int(x) for x in regex.findall(filesDIV[nb_files].rsplit('/', 1)[-1].rsplit('_', 5)[3])][0]
                try:
                    dev_nb = int(filesDIV[nb_files].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:])
                except:
                    dev_nb = filesDIV[nb_files].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:]
                nbsDIV_temp = np.append(nbsDIV_temp, [[dev_nb, pixel_nbDIV, scan_nbDIV]], axis = 0)
                DIVlabels_temp = np.append(DIVlabels_temp, "d" + str(dev_nb) + "p" + 
                    str(pixel_nbDIV) + "s" + str(scan_nbDIV))


            # glob doesn't read in data in an order. To fix this the arrays have to 
            # be sorted
            sortingIdx = np.lexsort((nbs_temp[:,1], nbs_temp[:,0]))
            sortingIdxDIV = np.lexsort((nbsDIV_temp[:,1], nbsDIV_temp[:,0]))

            # Shortly find out how many scans have been done (rudimentary but fast method)
            self.nbOfScans =  len([f for f in glob.glob(filepath + "*_d" + str(int(nbs_temp[0,0])) + 
                    "_p" + str(int(nbs_temp[0,1])) + "*_IV_0*" + ".txt")])

            # Sort everything
            self.nbs = nbs_temp[sortingIdx]
            self.nbsDIV = nbsDIV_temp[sortingIdxDIV]

            files = files[sortingIdx]
            filesDIV = filesDIV[sortingIdxDIV]

            # pyqtRemoveInputHook()
            # set_trace()
            self.performance_data = performance_data_temp[sortingIdx]

            self.IVlabels = IVlabels_temp[sortingIdx]
            self.DIVlabels = DIVlabels_temp[sortingIdxDIV]

            if np.size(self.IVselected) == 0:
                self.IVselected = np.repeat(True, np.size(files))
            
            # Check whether or not the data is dual (forward and reverse sweep)
            try:
                if np.all(self.performance_data[:, 7] == 0):
                    self.isDual = False
                else:
                    self.isDual = True
            except:
                self.isDual = False

            # Do plotting and formatting of LIV
            names = ["LV", "LC"] # Name of columns in files to read in
            # xylabel = ["Voltage (V)", "Current (mA/cm$^2$)"]
            # tickInterval = 0.4
            plotVars = ["LV", "LC"]
            self.IVdata, labels = self.readData(files, names, plotVars)
            # self.IVlabels = self.nbs
            
            # Read in DIV
            if np.size(filesDIV) > 0:
                names = ["DV", "DC"] # Name of columns in files to read in
                plotVars = ["DV", "DC"]
                self.DIVdata, labels = self.readData(filesDIV, names, plotVars)
                self.DIVdata[1::2] = self.DIVdata[1::2] * 1000 / (4.5 / 100)

            # Convert the raw IV data to mA/cm^2
            # So every other element has to be adjusted in units
            self.IVdata[1::2] = self.IVdata[1::2] * 1000 / (4.5 / 100)
            self.showHeroDevices("device")


        # Only show a certain percentage of I>0 for IV 
        # self._ax.set_ylim(self._ax.get_ylim()[0], self._ax.get_ylim()[1] * 0.5)
        self._ax.set_ylim(self._ax.get_ylim()[0] * 1.01, 8)
        self.fig.draw()
        self.updateToolbar("IV")

        self.isWhat = "IV"
        self.IVcalled = True 


    def assignGroup_dialog(self):
        # Dialog for the group assignment. Here in the IV dialog groups can be
        # assigned to devices.

        self.manualRowCountGridLayout = 1 

        # Define dialog in which parameters should be entered
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('Group Assignement Dialog')

        # Define all the layouts and labels so that the window looks good
        verticalLayout = QtWidgets.QVBoxLayout()

        if np.size(self.paths) == 1:
            if self.isWhat == "IV":
                CBTitle = QtWidgets.QLabel("Which scan do you want to use?")
                self.cb = QtWidgets.QComboBox()

                for i in range(self.nbOfScans):
                    self.cb.addItem(str(int(i + 1)))

                self.cb.setCurrentIndex(int(self.scan_nb - 1))
                verticalLayout.addWidget(CBTitle)
                verticalLayout.addWidget(self.cb)

            if self.isWhat == "IV":
                Title = QtWidgets.QLabel("How many different groups do you want to define for devices " +
                str(np.unique(self.nbs[:, 0])) + " ?")
                verticalLayout.addWidget(Title)

            elif self.isWhat == "mobility":
                Title = QtWidgets.QLabel("How many different groups do you want to define for devices " +
                str(np.unique(self.nbsDIV[:, 0])) + " ?")
                self.nbGroupsSlider = LabeledSlider(1, int(np.size(np.unique(self.nbsDIV[:, 0]))),
                        interval = 1, orientation=QtCore.Qt.Horizontal)

            self.nbGroupsSlider = LabeledSlider(1, int(np.size(np.unique(self.nbs[:, 0]))),
                    interval = 1, orientation=QtCore.Qt.Horizontal)

        else:
            self.nbGroupsSlider = LabeledSlider(1, int(np.size(self.GroupNames)),
                    interval = 1, orientation=QtCore.Qt.Horizontal)


        # Here the self defined slider is used which has built in numbering 
        # for the ticks of the slider
        self.nbGroupsSlider.valueChanged(self.nbGroupsSliderChanged)
        self._nbGroups = self.nbGroupsSlider.value()

        if np.size(self.paths) == 1:
            verticalLayout.addWidget(self.nbGroupsSlider)

        self.gridLayoutAGD = QtWidgets.QGridLayout()
        self.gridLayoutAGD.setSpacing(10)
        self.gridLayoutAGD.addWidget(QtWidgets.QLabel("Enter Label Name (Global Name in Parenthesis)"), 1, 0)
        self.gridLayoutAGD.addWidget(QtWidgets.QLabel("Enter Devices (seperated by ,)"), 1, 1)

        self.GroupNamesLineEdit = np.empty(0, dtype = "object")
        self.GroupNamesLineEdit = np.append(self.GroupNamesLineEdit, QtWidgets.QLineEdit())
        self.gridLayoutAGD.addWidget(self.GroupNamesLineEdit[0], 2, 0)

        if np.size(self.paths) == 1:
            self.GroupNbsLineEdit = np.empty(0, dtype = "object")
            self.GroupNbsLineEdit = np.append(self.GroupNbsLineEdit, QtWidgets.QLineEdit())
            self.gridLayoutAGD.addWidget(self.GroupNbsLineEdit[0], 2, 1)

        self.GroupColorsButton = np.empty(0, dtype = "object")
        self.GroupColorsButton = np.append(self.GroupColorsButton, QtWidgets.QPushButton(""))
        self.GroupColorsButton[0].clicked.connect(functools.partial(self.changeColor, 0)) 
        self.GroupColorsButton[0].setStyleSheet("background-color: " + str(self.GroupColor.astype('U7')[0]))
        self.gridLayoutAGD.addWidget(self.GroupColorsButton[0], 2, 2)

        if self.isWhat == "mobility":
            self.gridLayoutAGD.addWidget(QtWidgets.QLabel("Enter Layer Thickness (nm)"), 1, 3)
            self.GroupThickness = np.empty(0, dtype = "object")
            self.GroupThickness = np.append(self.GroupThickness, QtWidgets.QLineEdit())
            self.gridLayoutAGD.addWidget(self.GroupThickness[0], 2, 3)

        horizontalLayout6 = QtWidgets.QHBoxLayout()
        self.fitButton = QtWidgets.QPushButton("Name Groups")
        self.fitButton.clicked.connect(functools.partial(self.assignGroup, dialog))

        self.exitButton = QtWidgets.QPushButton("Close")
        self.exitButton.clicked.connect(functools.partial(self.close_dialog, dialog))
        horizontalLayout6.addWidget(self.exitButton)
        horizontalLayout6.addWidget(self.fitButton)

        verticalLayout.addLayout(self.gridLayoutAGD)
        verticalLayout.addLayout(horizontalLayout6)

        dialog.setLayout(verticalLayout)

        # If there were already some Group Names defined. Load them into the
        # Line Edits
        if np.size(self.GroupNames) > 0:
            self.nbGroupsSlider.setValue(np.size(self.GroupNames))
            self.nbGroupsSliderChanged()
            # pyqtRemoveInputHook()
            # set_trace()

            for groupNb in range(self._nbGroups):
                try:
                    # if self.GroupNames[groupNb] == self.GroupNamesGlobal[groupNb]:
                        # self.GroupNamesLineEdit[groupNb].setText(self.GroupNames.astype('U13')[groupNb])
                    # else:
                    self.GroupNamesLineEdit[groupNb].setText(self.GroupNames.astype('U13')[groupNb] + " ("
                            + self.GroupNamesGlobal.astype('U13')[groupNb] + ")")
                except:
                    self.GroupNamesLineEdit[groupNb].setText(self.GroupNames.astype('U13')[groupNb])

                if np.size(self.paths) == 1:
                    self.GroupNbsLineEdit[groupNb].setText(','.join(map(str, self.GroupNbs[groupNb])))

        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.statusBar().showMessage("Please Assign Group Names to Devices") 
        dialog.show()
        dialog.exec_()

    def changeColor(self, groupNb):
        # Change the color of the group by clicking on the button

        color = QtWidgets.QColorDialog.getColor()
        
        # Change the color of the push button according to the selection
        self.GroupColorsButton[groupNb].setStyleSheet("background-color: " + str(color.name()))

        # Add the color to the list
        # print(color.name())
        if groupNb < np.size(self.GroupColor):
            # print(groupNb)
            self.GroupColor[groupNb] = color.name()
        elif groupNb > np.size(self.GroupColor):
            self.GroupColor = np.append(self.GroupColor, color.name())

    def nbGroupsSliderChanged(self):
        # If the slider in assignGroup_dialog is moved the window has to be
        # updated according to the number of groups that shall be named.

        # Read out the Slider value
        self._nbGroups = self.nbGroupsSlider.value() 

        # Compare if the slider value is greater or smaller than the shown no
        # of QLineEdits. Add or delete elements accordingly.
        if self.manualRowCountGridLayout < self._nbGroups:
            while self.manualRowCountGridLayout < self._nbGroups:
                self.GroupNamesLineEdit = np.append(self.GroupNamesLineEdit, QtWidgets.QLineEdit())
                if np.size(self.paths) == 1:
                    self.GroupNbsLineEdit = np.append(self.GroupNbsLineEdit, QtWidgets.QLineEdit())
                self.GroupColorsButton = np.append(self.GroupColorsButton, QtWidgets.QPushButton(""))
                self.GroupColorsButton[-1].clicked.connect(functools.partial(
                    self.changeColor, self.manualRowCountGridLayout))

                if self.isWhat == "mobility":
                    self.GroupThickness = np.append(self.GroupThickness, QtWidgets.QLineEdit())

                # print(self.manualRowCountGridLayout)
                # print(np.size(self.GroupColor))
                if self.manualRowCountGridLayout < np.size(self.GroupColor):
                    self.GroupColorsButton[-1].setStyleSheet("background-color: " + 
                            str(self.GroupColor.astype('U7')[self.manualRowCountGridLayout]))

                self.gridLayoutAGD.addWidget(self.GroupNamesLineEdit[-1], 
                        np.size(self.GroupNamesLineEdit) + 1, 0)
                if np.size(self.paths) == 1:
                    self.gridLayoutAGD.addWidget(self.GroupNbsLineEdit[-1], 
                            np.size(self.GroupNamesLineEdit) + 1, 1)
                self.gridLayoutAGD.addWidget(self.GroupColorsButton[-1],
                        np.size(self.GroupNamesLineEdit) + 1, 2)
                if self.isWhat == "mobility":
                    self.gridLayoutAGD.addWidget(self.GroupThickness[-1],
                            np.size(self.GroupNamesLineEdit) + 1, 3)
                self.manualRowCountGridLayout += 1

        elif self.manualRowCountGridLayout > self._nbGroups:
            while self.manualRowCountGridLayout > self._nbGroups:
                # All the following commands have to be applied to "properly"
                # delete widgets in PyQt. 
                self.gridLayoutAGD.removeWidget(self.GroupNamesLineEdit[-1])
                if np.size(self.paths) == 1:
                    self.gridLayoutAGD.removeWidget(self.GroupNbsLineEdit[-1])
                self.gridLayoutAGD.removeWidget(self.GroupColorsButton[-1])
                self.GroupNamesLineEdit[-1].setParent(None)
                self.GroupNamesLineEdit[-1] = None
                if np.size(self.paths) == 1:
                    self.GroupNbsLineEdit[-1].setParent(None)
                    self.GroupNbsLineEdit[-1] = None
                self.GroupColorsButton[-1].setParent(None)
                self.GroupColorsButton[-1] = None

                if self.isWhat == "mobility":
                    self.gridLayoutAGD.removeWidget(self.GroupThickness[-1])
                    self.GroupThickness[-1].setParent(None)
                    self.GroupThickness[-1] = None

                # Delete element from numpy array as well
                self.GroupNamesLineEdit = np.delete(self.GroupNamesLineEdit, 
                        np.size(self.GroupNamesLineEdit) - 1)
                if np.size(self.paths) == 1:
                    self.GroupNbsLineEdit = np.delete(self.GroupNbsLineEdit, 
                            np.size(self.GroupNbsLineEdit) - 1)
                self.GroupColorsButton = np.delete(self.GroupColorsButton, 
                        np.size(self.GroupColorsButton) - 1)

                if self.isWhat == "mobility":
                    self.GroupThickness = np.delete(self.GroupThickness, 
                            np.size(self.GroupThickness) - 1)

                self.manualRowCountGridLayout -= 1

    def assignGroup(self, dialog):
        # Function that actually does the group assignment according to what was
        # typed in by the user. (Also checks if all entries are valid)


        # Define the variables that shall contain the information about the groups
        self.GroupNames = np.empty(np.size(self.GroupNamesLineEdit), dtype = "object")
        self.GroupNamesGlobal = np.empty(np.size(self.GroupNamesLineEdit), dtype = "object")

        if np.size(self.paths) == 1:
            self.GroupNbs = np.empty(np.size(self.GroupNbsLineEdit), dtype = "object")
        self.ThicknessLayer = np.empty(np.size(self.ThicknessLayer), dtype = "object")
        # self.GroupColor = np.empty(np.size(self.GroupNbsLineEdit), dtype = "object")

        for Group in range(np.size(self.GroupNamesLineEdit)):
            # Now check if the user input was valid or not
            if np.size(self.paths) == 1:
                if self.GroupNbsLineEdit[Group].text() == "":
                    warnings.warn("Please do enter device/s for your group. Try again!", category = UserWarning)
                    return
                if self.GroupNbsLineEdit[Group].text()[-1] == ",":
                    warnings.warn("Please do not end your numbers with a comma. Try again!", category = UserWarning)
                    return 
                try:
                    mylist = np.array([int(x) for x in self.GroupNbsLineEdit[Group].text().split(',')])
                except:
                    mylist = np.array([str(x) for x in self.GroupNbsLineEdit[Group].text().split(',')])
                if self.isWhat == "IV":
                    if np.all(np.in1d(mylist, np.unique(self.nbs[:, 0]))) == False:
                        warnings.warn("Please only enter valid device numbers.", category = UserWarning)
                        return
                elif self.isWhat == "mobility":
                    if np.all(np.in1d(mylist, np.unique(self.nbsDIV[:, 0]))) == False:
                        warnings.warn("Please only enter valid device numbers.", category = UserWarning)
                        return

            if self.GroupNamesLineEdit[Group].text() == "":
                warnings.warn("Please do enter a name for your group. Try again!", category = UserWarning)
                return


            # Here I want to distinguish between the global name and the local
            # name. The difference is only if I read in several folders and the
            # local name in the group was different to the one I want to use
            # e.g. on a paper and to accumulate data.
            s = self.GroupNamesLineEdit[Group].text()
            self.GroupNames[Group] = s.split('(')[0].rstrip() 
            
            if self.isWhat == "mobility":
                self.ThicknessLayer[Group] = float(self.GroupThickness[Group].text())

            if "(" in s:
                self.GroupNamesGlobal[Group] = s[s.find("(")+1:s.find(")")] 
            else:
                self.GroupNamesGlobal[Group] = "None" 
            
            if np.size(self.paths) == 1:
                self.GroupNbs[Group] = mylist
            # self.GroupColor[Group] = self.cmap(float(Group) / np.size(self.GroupNamesLineEdit))


        self.statusBar().showMessage("Group Names assigned to Devices", 5000) 
        self.showHeroDevicesAction.setEnabled(True)

        # If the directory used to save intensity dependent IV exists the 
        # intensity dependent IV action can be enabled
        if os.path.isdir(self.globalPath + "/IntDepIV/") == True: 
            self.intDepIVAction.setEnabled(True)

        # print(self.GroupColor)
        self.groupsAssigned = True

        if np.size(self.paths) == 1:
            if self.isWhat == "IV":
                if self.scan_nb != int(self.cb.currentText()):
                    # Set selected scan number
                    self.scan_nb = int(self.cb.currentText())
                    self.IVcalled = "False"
                    self.IVselected = np.empty(0, dtype = "object")
                    self.load_IV()
            elif self.isWhat == "mobility":
                self.showGroupAction.setEnabled(True)
                self.extractMobilityAction.setEnabled(True)

        dialog.close()


    def showGroup_dialog(self):
        # Dialog that shall enable to plot the IVs of a group or of a device 

        # Define dialog in which parameters should be entered
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('Show Group Dialog')

        # Define all the layouts and labels so that the window looks good
        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.addWidget(QtWidgets.QLabel("IV Curves of which group or device shall be plotted?"))
        verticalLayout.addWidget(QtWidgets.QLabel("Light IV"))

        # In a grid layout put buttons for all the valid devices
        self.gridLayoutSGDDevices = QtWidgets.QGridLayout()
        self.PBdevices = np.empty(0, dtype = "object")

        # Read out the Slider value
        # Read out number of devices
        nb_devices = np.size(np.unique(self.nbs[:, 0]))
        nb_devicesDIV = np.size(np.unique(self.nbsDIV[:, 0]))
        count_row = 0 # have a line break every max_row push buttons
        max_row = 6

        # Generate the same number of push buttons than nb of devices
        for nbPB in range(nb_devices):
            self.PBdevices = np.append(self.PBdevices, 
                    QtWidgets.QPushButton(str(int(np.unique(self.nbs[:, 0])[nbPB]))))
            self.PBdevices[nbPB].clicked.connect(functools.partial(self.showGroup,
                np.unique(self.nbs[:, 0])[nbPB], "light"))
            self.gridLayoutSGDDevices.addWidget(self.PBdevices[-1], 
                    int(count_row / max_row), (np.size(self.PBdevices) - 1) % max_row)
            count_row += 1

        verticalLayout.addLayout(self.gridLayoutSGDDevices)

        if np.size(self.DIVdata) > 0:
            verticalLayout.addWidget(QtWidgets.QLabel("Dark IV"))
            self.gridLayoutSGDDevicesDark = QtWidgets.QGridLayout()
            self.PBdevicesDark = np.empty(0, dtype = "object")

            count_row = 0 # have a line break every max_row push buttons

            # Generate the same number of push buttons than nb of devices for dark IV
            for nbPB in range(nb_devicesDIV):
                try:
                    self.PBdevicesDark = np.append(self.PBdevicesDark, 
                            QtWidgets.QPushButton(str(int(np.unique(self.nbsDIV[:, 0])[nbPB]))))
                except:
                    self.PBdevicesDark = np.append(self.PBdevicesDark, 
                            QtWidgets.QPushButton(str(np.unique(self.nbsDIV[:, 0])[nbPB])))

                self.PBdevicesDark[nbPB].clicked.connect(functools.partial(self.showGroup,
                    np.unique(self.nbsDIV[:, 0])[nbPB], "dark"))
                self.gridLayoutSGDDevicesDark.addWidget(self.PBdevicesDark[-1], 
                        int(count_row / max_row), (np.size(self.PBdevicesDark) - 1) % max_row)
                count_row += 1

            verticalLayout.addLayout(self.gridLayoutSGDDevicesDark)

        if np.size(self.GroupNames) > 0 and self.isWhat == "IV":
            # Generate the same number of push buttons than nb of devices
            # In a grid layout put buttons for all the valid groups
            self.labelGroup = QtWidgets.QLabel("Groups:")
            verticalLayout.addWidget(self.labelGroup)
            self.gridLayoutSGDGroups = QtWidgets.QGridLayout()
            self.PBgroups = np.empty(0, dtype = "object")

            nb_devices = np.size(np.unique(self.nbs[:, 0]))
            count_row = 0 # have a line break every max_row push buttons

            for nbPB in range(np.size(self.GroupNames)):
                self.PBgroups = np.append(self.PBgroups, 
                        QtWidgets.QPushButton(str(self.GroupNames.astype('U13')[nbPB])))
                self.PBgroups[nbPB].clicked.connect(functools.partial(self.showGroup,
                    self.GroupNbs[nbPB], "light"))
                self.gridLayoutSGDGroups.addWidget(self.PBgroups[-1], 
                        int(count_row / max_row), (np.size(self.PBgroups) - 1) % max_row)
                count_row += 1

            verticalLayout.addLayout(self.gridLayoutSGDGroups)

        # Add an exit button to the dialog
        self.exitButton = QtWidgets.QPushButton("Close")
        self.exitButton.clicked.connect(functools.partial(self.close_dialog, dialog))
        verticalLayout.addWidget(self.exitButton)

        dialog.setLayout(verticalLayout)

        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.show()
        dialog.exec_()

    def showGroup(self, DeviceNumber, darkOrLight):
        print(self.IVselected)
        # Actual function that shows the group of IV curves according to which
        # button was pressed.

        # Clear plot (necessary to plot new stuff)
        self.fig.figure.clf()
        self.fig.draw()

        # Define the necessary parameters for plotting IV data
        xylabel = ["Voltage (V)", "Current (mA/cm$^2$)"]
        tickInterval = 0.4

        # From the read in data only extract the chosen device/ group
        PixelsOfDevice = np.empty(0, dtype = int)
        PixelsOfDeviceIV = np.empty(0, dtype = int)
        DataOfDevice = np.empty(0, dtype = int)

        # If a group is selected go through all the devices
        if np.size(DeviceNumber) > 1:
            for devnb in DeviceNumber:
                if darkOrLight == "light":
                    tempPixelsOfDeviceIV = np.where(self.nbs[:, 0] == devnb)[0]
                    tempPixelsOfDevice = np.where(self.nbs[:, 0] == devnb)[0]
                elif darkOrLight == "dark":
                    tempPixelsOfDeviceIV = np.where(self.nbs[:, 0] == devnb)[0]
                    tempPixelsOfDevice = np.where(self.nbsDIV[:, 0] == devnb)[0]

                PixelsOfDevice = np.append(PixelsOfDevice, tempPixelsOfDevice) 
                PixelsOfDeviceIV = np.append(PixelsOfDeviceIV, tempPixelsOfDeviceIV)
                DataOfDevice = np.append(DataOfDevice, np.append(tempPixelsOfDevice + tempPixelsOfDevice[0],
                    tempPixelsOfDevice + np.size(tempPixelsOfDevice) + tempPixelsOfDevice[0]))
        # If a single device is selected only plot that one
        elif np.size(DeviceNumber) == 1:
            if darkOrLight == "light":
                tempPixelsOfDeviceIV = np.where(self.nbs[:, 0] == DeviceNumber)[0]
                tempPixelsOfDevice = np.where(self.nbs[:, 0] == DeviceNumber)[0]
            elif darkOrLight == "dark":
                tempPixelsOfDeviceIV = np.where(self.nbs[:, 0] == DeviceNumber)[0]
                tempPixelsOfDevice = np.where(self.nbsDIV[:, 0] == DeviceNumber)[0]

            PixelsOfDevice = np.append(PixelsOfDevice, tempPixelsOfDevice) 
            PixelsOfDeviceIV = np.append(PixelsOfDeviceIV, tempPixelsOfDeviceIV) 
            DataOfDevice = np.append(DataOfDevice, np.append(tempPixelsOfDevice + tempPixelsOfDevice[0],
                tempPixelsOfDevice + np.size(tempPixelsOfDevice) + tempPixelsOfDevice[0]))
        else:
            warnings.warn("Error: No device is selected", category = UserWarning)

        # Obtain the data that shall be plotted from the already read in IV data
        if darkOrLight == "light":
            dataToPlot = self.IVdata[DataOfDevice]
            labels = self.IVlabels[PixelsOfDevice]
        elif darkOrLight == "dark":
            dataToPlot = self.DIVdata[DataOfDevice]
            labels = self.DIVlabels[PixelsOfDevice]

        # Plot the selected data with the correct labels
        self._ax = self.fig.figure.subplots()
        self.plot(dataToPlot, labels, xylabel, tickInterval, ax_ = self._ax,
                IV_ = True, zeroLines_ = True)

        # Do not show the non-selected pixel (must also be before fig.draw())
        all_handles = self._ax.get_legend().get_lines()
        i = 0

        if self.isWhat == "IV":
            for logical in self.IVselected[PixelsOfDeviceIV]:
                if logical == False:
                    # self.IVselected[]
                    legline = all_handles[i]
                    origline = self.lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    legline.set_alpha(0.2)
                i += 1
            
            self._ax.set_ylim(self._ax.get_ylim()[0] * 1.01, 8)

        elif self.isWhat == "mobility":
            for logical in self.IVselected[PixelsOfDevice]:
                if logical == False:
                    # self.IVselected[]
                    legline = all_handles[i]
                    origline = self.lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    legline.set_alpha(0.2)
                i += 1

        # Only show a certain percentage of I>0 for IV 
        # self._ax.set_ylim(self._ax.get_ylim()[0], self._ax.get_ylim()[1] * 0.5)
        self.fig.draw()

    def showOverview_dialog(self):
        # Dialog that enables to show an overview over all selected IV curves.
        # Overview in this case means that Voc, Jsc, FF and PCE are plotted
        # seperately in a box plot.

        # Define dialog in which parameters should be entered
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('Show Overview of Performance Parameters')

        # Define all the layouts and labels so that the window looks good
        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.addWidget(QtWidgets.QLabel("Show Overview of Forward Data:"))

        # In a grid layout put buttons for all the valid devices
        self.horizontalLayoutSOD = QtWidgets.QHBoxLayout()

        if self.multipleFoldersLoaded == False:
            self.buttonOverviewDevices = QtWidgets.QPushButton("Devices")
            self.buttonOverviewDevices.clicked.connect(functools.partial(self.showOverview, "devices", "forward"))
            self.horizontalLayoutSOD.addWidget(self.buttonOverviewDevices)

        self.buttonOverviewGroup = QtWidgets.QPushButton("Group")
        self.buttonOverviewGroupAvg = QtWidgets.QPushButton("Avg Group")

        self.buttonOverviewGroup.clicked.connect(functools.partial(self.showOverview, "groups", "forward"))
        self.buttonOverviewGroupAvg.clicked.connect(functools.partial(self.showOverview, "groups", "forward", avg = True))

        if self.groupsAssigned == True:
            self.buttonOverviewGroup.setEnabled(True)
            self.buttonOverviewGroupAvg.setEnabled(True)
        elif self.groupsAssigned == False:
            self.buttonOverviewGroup.setEnabled(False)
            self.buttonOverviewGroupAvg.setEnabled(False)

        self.horizontalLayoutSOD.addWidget(self.buttonOverviewGroup)
        self.horizontalLayoutSOD.addWidget(self.buttonOverviewGroupAvg)

        verticalLayout.addLayout(self.horizontalLayoutSOD)

        # If there is dual data it must be possible to plot it seperately
        if self.isDual == True:
            verticalLayout.addWidget(QtWidgets.QLabel("Show Overview of Reverse Data:"))
            self.horizontalLayoutSODRev = QtWidgets.QHBoxLayout()

            if self.multipleFoldersLoaded == False:
                self.buttonOverviewDevicesRev = QtWidgets.QPushButton("Devices")
                self.buttonOverviewDevicesRev.clicked.connect(functools.partial(self.showOverview, "devices", "reverse"))
                self.horizontalLayoutSODRev.addWidget(self.buttonOverviewDevicesRev)

            self.buttonOverviewGroupRev = QtWidgets.QPushButton("Group")
            self.buttonOverviewGroupRevAvg = QtWidgets.QPushButton("Avg Group")
            self.buttonOverviewGroupRev.clicked.connect(functools.partial(self.showOverview, "groups", "reverse"))
            self.buttonOverviewGroupRevAvg.clicked.connect(functools.partial(self.showOverview, "groups", "reverse", avg = True))

            self.horizontalLayoutSODRev.addWidget(self.buttonOverviewGroupRev)
            self.horizontalLayoutSODRev.addWidget(self.buttonOverviewGroupRevAvg)
            verticalLayout.addLayout(self.horizontalLayoutSODRev)

            if self.groupsAssigned == True:
                self.buttonOverviewGroupRev.setEnabled(True)
            elif self.groupsAssigned == False:
                self.buttonOverviewGroupRev.setEnabled(False)


        # Add an exit button to the dialog
        self.exitButton = QtWidgets.QPushButton("Close")
        self.exitButton.clicked.connect(functools.partial(self.close_dialog, dialog))
        verticalLayout.addWidget(self.exitButton)

        dialog.setLayout(verticalLayout)

        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.show()
        dialog.exec_()

    def showOverview(self, groupOrDevice, forwardOrReverse, avg = False):
        # Function that does prepare everything for plotting the overview in a 
        # box plot

        if self.multipleFoldersLoaded == False:
            if self.isDual == True:
                Voc_dev, Jsc_dev, FF_dev, PCE_dev, Voc_dev_rev, Jsc_dev_rev, FF_dev_rev, PCE_dev_rev, labels_ = self.sortParams(groupOrDevice) 
            else:
                Voc_dev, Jsc_dev, FF_dev, PCE_dev, labels_ = self.sortParams(groupOrDevice) 
        else:
            Voc_dev = self.Voc_all
            Jsc_dev = self.Jsc_all
            FF_dev = self.FF_all
            PCE_dev = self.PCE_all
            if self.isDual == True:
                Voc_dev_rev = self.Voc_rev_all
                Jsc_dev_rev = self.Jsc_rev_all
                FF_dev_rev = self.FF_rev_all
                PCE_dev_rev = self.PCE_rev_all

            labels_ = self.GroupNames.astype('U13')


        if forwardOrReverse == "forward":
            self.plotOverview(Voc_dev, Jsc_dev, FF_dev, PCE_dev, groupOrDevice, forwardOrReverse, labels_, avg)
        elif forwardOrReverse == "reverse":
            self.plotOverview(Voc_dev_rev, Jsc_dev_rev, FF_dev_rev, PCE_dev_rev, groupOrDevice, forwardOrReverse, labels_, avg)
    
    def sortParams(self, groupOrDevice):
        # This function has to be seperated from showOverview since it shall
        # also be called in the saveFile section. Like this the user does not
        # have to use showOverview before the file is saved.

        # Forward sweep
        Voc_dev = []
        Jsc_dev = []
        FF_dev = []
        PCE_dev = []

        if self.isDual == True:
            # Reverse sweep
            Voc_dev_rev = []
            Jsc_dev_rev = []
            FF_dev_rev = []
            PCE_dev_rev = []

        # Labels
        labels_ = np.empty(0, dtype = str)

        if groupOrDevice == "devices":
            for i in np.unique(self.nbs[:, 0]):
                # Forward sweep
                Voc_dev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 1])
                Jsc_dev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 2])
                FF_dev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 3])
                PCE_dev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 4])

                if self.isDual == True:
                    # Reverse sweep
                    Voc_dev_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 7])
                    Jsc_dev_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 8])
                    FF_dev_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 9])
                    PCE_dev_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 10])

                labels_ = np.append(labels_, str(int(i))) 

        elif groupOrDevice == "groups":
            for k in range(np.size(self.GroupNames)):
                temp_Voc = []
                temp_Jsc = []
                temp_FF = []
                temp_PCE = []

                if self.isDual == True:
                    temp_Voc_rev = []
                    temp_Jsc_rev = []
                    temp_FF_rev = []
                    temp_PCE_rev = []

                for i in np.unique(self.GroupNbs[k]):
                    temp_Voc.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 1])
                    temp_Jsc.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 2])
                    temp_FF.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 3])
                    temp_PCE.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 4])

                    if self.isDual == True:
                        # Reverse sweep
                        temp_Voc_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 7])
                        temp_Jsc_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 8])
                        temp_FF_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 9])
                        temp_PCE_rev.append(self.performance_data[np.logical_and(self.nbs[:, 0] == i, self.IVselected), 10])

                temp_Voc = np.concatenate(temp_Voc)
                temp_Jsc = np.concatenate(temp_Jsc)
                temp_FF = np.concatenate(temp_FF)
                temp_PCE = np.concatenate(temp_PCE)

                if self.isDual == True:
                    temp_Voc_rev = np.concatenate(temp_Voc_rev)
                    temp_Jsc_rev = np.concatenate(temp_Jsc_rev)
                    temp_FF_rev = np.concatenate(temp_FF_rev)
                    temp_PCE_rev = np.concatenate(temp_PCE_rev)

                # forward
                Voc_dev.append(np.asarray(temp_Voc, dtype = "float64"))
                Jsc_dev.append(np.asarray(temp_Jsc, dtype = "float64"))
                FF_dev.append(np.asarray(temp_FF, dtype = "float64"))
                PCE_dev.append(np.asarray(temp_PCE, dtype = "float64"))

                # reverse
                if self.isDual == True:
                    Voc_dev_rev.append(np.asarray(temp_Voc_rev, dtype = "float64"))
                    Jsc_dev_rev.append(np.asarray(temp_Jsc_rev, dtype = "float64"))
                    FF_dev_rev.append(np.asarray(temp_FF_rev, dtype = "float64"))
                    PCE_dev_rev.append(np.asarray(temp_PCE_rev, dtype = "float64"))

            labels_ = self.GroupNames.astype('U13')

        else:
            warnings.warn("Please enter groups or devices in the sortParams function and nothing else.", category = UserWarning)


        if self.isDual == True:
            return Voc_dev, Jsc_dev, FF_dev, PCE_dev, Voc_dev_rev, Jsc_dev_rev, FF_dev_rev, PCE_dev_rev, labels_
        else:
            return Voc_dev, Jsc_dev, FF_dev, PCE_dev, labels_

    def plotOverview(self, Voc_dev, Jsc_dev, FF_dev, PCE_dev, groupOrDevice, forwardOrReverse, labels_, avg):
        # Function that actually does the plotting of the overview

        # For the overview the performance parameters for all the cells must be plotted
        self.line = {}
        # Clear plot (necessary to plot new stuff)
        self.fig.figure.clf()
        self.fig.draw()

        # Generate four subplots
        self._ax = self.fig.figure.subplots(2, 2)
        self._ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self._ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        self._ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        self._ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # Voc in avg plot
        if avg == True:
            avgVoc = [np.average(subarray) for subarray in Voc_dev]
            stdVoc = [np.std(subarray) for subarray in Voc_dev]
            avgJsc = [np.average(subarray) for subarray in Jsc_dev]
            stdJsc = [np.std(subarray) for subarray in Jsc_dev]
            avgFF = [np.average(subarray) for subarray in FF_dev]
            stdFF = [np.std(subarray) for subarray in FF_dev]
            avgPCE = [np.average(subarray) for subarray in PCE_dev]
            stdPCE = [np.std(subarray) for subarray in PCE_dev]

            # # Do plotting and formatting
            xlim = (-0.5, np.size(self.GroupNames) - 0.5)
            ylimVoc = (np.min(avgVoc) - np.max(stdVoc) - 0.01, np.max(avgVoc) + np.max(stdVoc) + 0.01)
            ylimJsc = (np.min(avgJsc) - np.max(stdJsc) - 0.2, np.max(avgJsc) + np.max(stdJsc) + 0.2)
            ylimFF = (np.min(avgFF) - np.max(stdFF) - 0.5, np.max(avgFF) + np.max(stdFF) + 0.5)
            ylimPCE = (np.min(avgPCE) - np.max(stdPCE) - 0.2, np.max(avgPCE) + np.max(stdPCE) + 0.2)
            xylabel = ["", "Voc (V)"]
            tickInterval = 1 
            labels = self.GroupNames

            self.plot([self.GroupNames, avgVoc, stdVoc], labels, xylabel, tickInterval, 
                    ax_ = self._ax[0,0], linestyle_ = "", marker_ = "s", xlim_ = xlim, 
                    ylim_ = ylimVoc, errbar_ = True, minorTicks_ = False)
            self.plot([self.GroupNames, avgJsc, stdJsc], labels, xylabel, tickInterval, 
                    ax_ = self._ax[0,1], linestyle_ = "", marker_ = "s", xlim_ = xlim, 
                    ylim_ = ylimJsc, errbar_ = True, minorTicks_ = False)
            self.plot([self.GroupNames, avgFF, stdFF], labels, xylabel, tickInterval, 
                    ax_ = self._ax[1,0], linestyle_ = "", marker_ = "s", xlim_ = xlim, 
                    ylim_ = ylimFF, errbar_ = True, minorTicks_ = False)
            self.plot([self.GroupNames, avgPCE, stdPCE], labels, xylabel, tickInterval, 
                    ax_ = self._ax[1,1], linestyle_ = "", marker_ = "s", xlim_ = xlim, 
                    ylim_ = ylimPCE, errbar_ = True, minorTicks_ = False)

        else:

            # Voc in boxplot
            self._ax[0,0].boxplot(Voc_dev, labels = labels_, showfliers = False)
            self._ax[0,1].boxplot(Jsc_dev, labels = labels_, showfliers = False)
            self._ax[1,0].boxplot(FF_dev, labels = labels_, showfliers = False)
            self._ax[1,1].boxplot(PCE_dev, labels = labels_, showfliers = False)
            
            # Have every single data point plotted as well with a bit of spread in
            # x direction
            if forwardOrReverse == "forward":
                color = "b"
            elif forwardOrReverse == "reverse":
                color = "r"

            if groupOrDevice == "devices":
                for i in range(np.size(np.unique(self.nbs[:, 0]))):
                    # Voc scatter plot
                    y_Voc = Voc_dev[i]
                    x_Voc = np.random.normal(1+i, 0.04, size=len(y_Voc))
                    self._ax[0,0].plot(x_Voc, y_Voc, color + '.')

                    # Jsc scatter plot
                    y_Jsc = Jsc_dev[i]
                    x_Jsc = np.random.normal(1+i, 0.04, size=len(y_Jsc))
                    self._ax[0,1].plot(x_Jsc, y_Jsc, color + '.')

                    # FF scatter plot
                    y_FF = FF_dev[i]
                    x_FF = np.random.normal(1+i, 0.04, size=len(y_FF))
                    self._ax[1,0].plot(x_FF, y_FF, color + '.')

                    # PCE scatter plot
                    y_PCE = PCE_dev[i]
                    x_PCE = np.random.normal(1+i, 0.04, size=len(y_PCE))
                    self._ax[1,1].plot(x_PCE, y_PCE, color + '.')

            elif groupOrDevice == "groups":
                for i in range(np.size(self.GroupNames)):
                    # Voc scatter plot
                    y_Voc = Voc_dev[i]
                    x_Voc = np.random.normal(1+i, 0.04, size=len(y_Voc))
                    self._ax[0,0].plot(x_Voc, y_Voc, color + '.')

                    # Jsc scatter plot
                    y_Jsc = Jsc_dev[i]
                    x_Jsc = np.random.normal(1+i, 0.04, size=len(y_Jsc))
                    self._ax[0,1].plot(x_Jsc, y_Jsc, color + '.')

                    # FF scatter plot
                    y_FF = FF_dev[i]
                    x_FF = np.random.normal(1+i, 0.04, size=len(y_FF))
                    self._ax[1,0].plot(x_FF, y_FF, color + '.')

                    # PCE scatter plot
                    y_PCE = PCE_dev[i]
                    x_PCE = np.random.normal(1+i, 0.04, size=len(y_PCE))
                    self._ax[1,1].plot(x_PCE, y_PCE, color + '.')

                    print(self.GroupNames.astype('U13')[i])
                    print("Avg. Voc: " + str(np.round(np.average(Voc_dev[i]), 4)) + 
                            "+-" + str(np.round(np.std(Voc_dev[i]), 4)) +
                        ", Avg. Jsc: " + str(np.round(np.average(Jsc_dev[i]), 4)) +
                        "+-" + str(np.round(np.std(Jsc_dev[i]), 4)) +
                        ", Avg. FF: " + str(np.round(np.average(FF_dev[i]), 4)) + 
                        "+-" + str(np.round(np.std(FF_dev[i]), 4)) +
                        ", Avg. PCE: " + str(np.round(np.average(PCE_dev[i]), 4)) + 
                        "+-" + str(np.round(np.std(PCE_dev[i]), 4)))


        # grid
        self._ax[0,0].grid(True)
        self._ax[0,1].grid(True)
        self._ax[1,0].grid(True)
        self._ax[1,1].grid(True)

        # set ylabel (no xlabel)
        self._ax[0,0].set_ylabel("V$_{oc}$ [V]", fontsize = self.fonts_)
        self._ax[0,1].set_ylabel("J$_{SC}$ [mA/cm$^2$]", fontsize = self.fonts_)
        self._ax[1,0].set_ylabel("FF [%]", fontsize = self.fonts_)
        self._ax[1,1].set_ylabel("PCE [%]", fontsize = self.fonts_)

        # set axis linewidth 
        for axis in ['top','bottom','left','right']:
          self._ax[0,0].spines[axis].set_linewidth(self.ax_ticks_width)
          self._ax[0,1].spines[axis].set_linewidth(self.ax_ticks_width)
          self._ax[1,0].spines[axis].set_linewidth(self.ax_ticks_width)
          self._ax[1,1].spines[axis].set_linewidth(self.ax_ticks_width)

        # Set tick fontsizes
        for tick in self._ax[0,0].xaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)
        for tick in self._ax[0,0].yaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)

        # Set tick fontsizes
        for tick in self._ax[0,1].xaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)
        for tick in self._ax[0,1].yaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)

        # Set tick fontsizes
        for tick in self._ax[1,0].xaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)
        for tick in self._ax[1,0].yaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)

        # Set tick fontsizes
        for tick in self._ax[1,1].xaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)
        for tick in self._ax[1,1].yaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)

        # set tight layout so that nothing does overlap
        self.fig.figure.tight_layout()
        self.fig.draw()

    def showHeroDevices(self, groupOrDevice):
        # Function that only shows the hero devices of the groups (when IV is
        # pressed for the first time it shows the hero devices of each cell
        # since no group assignment was done at this point.)

        # First figure out where the hero devices are
        # dataToPlot = np.empty(0)
        dataToPlot = [] 
        labels = np.empty(0)

        if groupOrDevice == "group" and np.size(self.GroupNames) == 0:
            warnings.warn("Use the assign group dialog before plotting the Hero Devices of each Group!", category = UserWarning)
            return

        # Plot hero pixel of every group
        if groupOrDevice == "group":
            for groupNb in range(np.size(self.GroupNames)):
                dataToPlot.append(self.IVdata[2 * np.where(self.performance_data[:, 4] == np.max(self.performance_data[
                    np.where(np.in1d(self.nbs[:, 0], self.GroupNbs[groupNb][:]))][:, 4]))[0][0]])
                dataToPlot.append(self.IVdata[2 * np.where(self.performance_data[:, 4] == np.max(self.performance_data[
                    np.where(np.in1d(self.nbs[:, 0], self.GroupNbs[groupNb][:]))][:, 4]))[0][0] + 1])

                print(self.GroupNames.astype('U13')[groupNb] + "'s hero pixel is " + str(self.nbs[np.where(self.performance_data[:, 4] == np.max(self.performance_data[
                    np.where(np.in1d(self.nbs[:, 0], self.GroupNbs[groupNb][:]))][:, 4]))[0][0], :]))
            labels = self.GroupNames.astype('U13')
            self.statusBar().showMessage("Hero Pixels of all different Groups", 10000) 

        # Plot hero pixel of every device
        elif groupOrDevice == "device":
            for devNb in np.unique(self.nbs[:, 0]):

                dataToPlot.append(self.IVdata[2 * np.where(self.performance_data[:, 4] == np.max(self.performance_data[
                    np.where(self.nbs[:, 0] == devNb)][:, 4]))[0][0]])
                dataToPlot.append(self.IVdata[2 * np.where(self.performance_data[:, 4] == np.max(self.performance_data[
                    np.where(self.nbs[:, 0] == devNb)][:, 4]))[0][0] + 1])
                labels = np.append(labels, self.IVlabels[np.where(self.performance_data[:, 4] == np.max(self.performance_data[
                    np.where(self.nbs[:, 0] == devNb)][:, 4]))[0][0]])
                # print(self.nbs[devNb, 0])
            self.statusBar().showMessage("Hero Pixels of all different Devices", 10000) 

        # Clear plot (necessary to plot new stuff)
        self.line = {}
        self.fig.figure.clf()
        self.fig.draw()

        xylabel = ["Voltage (V)", "Current (mA/cm$^2$)"]
        tickInterval = 0.4
        self._ax = self.fig.figure.subplots()

        self.plot(dataToPlot, labels, xylabel, tickInterval, ax_ = self._ax,
                IV_ = True, zeroLines_ = True)

        # Only show a certain percentage of I>0 for IV 
        # self._ax.set_ylim(self._ax.get_ylim()[0], self._ax.get_ylim()[1] * 0.5)
        self._ax.set_ylim(self._ax.get_ylim()[0] * 1.01, 8)
        self.fig.draw()

    def plot_intDepIV(self):
        # Function that does plotting and calculation of intensity dependent
        # IV plot. It is only active if a folder named /IntDepIV/ exists in the
        # global parent folder and a group assignment has been done.

        # Read in data from PVParameters file and load in an array
        filepath = self.globalPath + "/IntDepIV/"
        filename = filepath + "PV parameters Dual.txt"
        names_ = ["Data File Name", "Voc", "Jsc", "FF", "PCE", "Vmpp", "Impp", 
                "Voc-b", "Jsc-b", "FF-b", "PCE-b", "Vmpp-b", "Impp-b"]
        file_data_list = pd.read_csv(filename, skiprows = 1, skipfooter = 0,
                sep = "\t", names = names_, engine = "python")
        file_data_array = np.array(file_data_list)

        # Declare arrays needed in a bit
        infZeroElems = np.empty(0, dtype = int)
        elems_to_delete = np.empty(0, dtype = int)
        nbs_temp = np.empty([0, 4], dtype = int)

        # From file name extract the important numbers
        # Detect unlogical parameters (infinity, zero etc)
        for perfNb in range(np.shape(file_data_array)[0]):
            try:
                regex = re.compile(r'\d+')
                extracted_nbs = [int(x) for x in regex.findall(file_data_array[perfNb, 0].rsplit('/', 1)[-1])]
                # print(extracted_nbs)
                nbs_temp = np.append(nbs_temp, [[extracted_nbs[1], extracted_nbs[2],
                    extracted_nbs[3], extracted_nbs[4]]], axis = 0)

                if np.any(file_data_array[perfNb, :] == 0) or np.any(file_data_array[perfNb, :] == float("inf")):
                    file_data_array[perfNb, file_data_array[perfNb, :] == float("inf")] = 0
                    file_data_array[perfNb, 0] = 0
                    infZeroElems = np.append(infZeroElems, perfNb)
            except:
                warnings.warn("An error occured while trying to extract nbs from filename " + 
                        str(file_data_array[perfNb, 0]) + " please check the filenames in the " +
                        "PV Parameters file.", category = UserWarning)

        # Detect all elements belonging to the same pixel that was unlogic
        for elem2Del in range(np.shape(infZeroElems)[0]):
            elems_to_delete = np.append(elems_to_delete, np.where(np.logical_and(nbs_temp[:, 1] == nbs_temp[infZeroElems[elem2Del], 1],
                nbs_temp[:, 2] == nbs_temp[infZeroElems[elem2Del], 2], 
                nbs_temp[:, 3] == nbs_temp[infZeroElems[elem2Del], 3])))

        # Really delete the unlogical pixels and all belonging measurements
        file_data_array = np.delete(file_data_array, np.unique(elems_to_delete), axis = 0)
        print(file_data_array)
        nbs_temp = np.delete(nbs_temp, np.unique(elems_to_delete), axis = 0)

        # Now sort the elements to enable plotting according to their group and 
        # according to the used intensity mask
        Voc_dev = []

        self.line = {}
        self.fig.figure.clf()
        self.fig.draw()

        self._ax = self.fig.figure.subplots()
        tickInterval_ = 10

        for k in range(np.size(self.GroupNames)):
            store_plot = []
            store_plot_err = []
            for m in np.unique(nbs_temp[:, 0]):
                temp_Voc = []
                temp_FF = []
                temp_Jsc = []
                temp_PCE = []

                for i in np.unique(self.GroupNbs[k]):
                    temp_Voc = np.append(temp_Voc, file_data_array[np.logical_and(nbs_temp[:, 1] == i,
                        nbs_temp[:, 0] == m), 1])

                    # Those three are only stored temporarily to detect odd
                    # results
                    temp_Jsc = np.append(temp_Jsc, file_data_array[np.logical_and(nbs_temp[:, 1] == i,
                        nbs_temp[:, 0] == m), 2])
                    temp_FF = np.append(temp_FF, file_data_array[np.logical_and(nbs_temp[:, 1] == i,
                        nbs_temp[:, 0] == m), 3])
                    temp_PCE = np.append(temp_PCE, file_data_array[np.logical_and(nbs_temp[:, 1] == i,
                        nbs_temp[:, 0] == m), 4])

                if np.size(temp_Voc) > 0:
                    # Only keep values that are within one std deviation error
                    # because the other ones are most probably odd.
                    # This is checked for all four parameters (Voc, Jsc, FF and
                    # PCE). Hopefully the results are good enough so that not
                    # all IV curves have to be checked to do the plotting.
                    # Only keep values that are higher than a certain value
                    # threshold = 0.8
                    # temp_Jsc = temp_Jsc[temp_Voc > threshold]
                    # temp_FF = temp_FF[temp_Voc > threshold]
                    # temp_PCE = temp_PCE[temp_Voc > threshold]
                    # temp_Voc = temp_Voc[temp_Voc > threshold]

                    temp_Voc = temp_Voc[np.logical_and(
                        np.logical_and(np.logical_and(temp_Voc >= np.mean(temp_Voc) - np.std(temp_Voc),
                        temp_Voc <= np.mean(temp_Voc) + np.std(temp_Voc)), 
                        np.logical_and(temp_Jsc >= np.mean(temp_Jsc) - np.std(temp_Jsc),
                        temp_Jsc <= np.mean(temp_Jsc) + np.std(temp_Jsc))),
                        np.logical_and(np.logical_and(temp_FF >= np.mean(temp_FF) - np.std(temp_FF),
                        temp_FF <= np.mean(temp_FF) + np.std(temp_FF)),
                        np.logical_and(temp_PCE >= np.mean(temp_PCE) - np.std(temp_PCE),
                        temp_PCE <= np.mean(temp_PCE) + np.std(temp_PCE))))]


                    if np.size(temp_Voc) == 1:
                        store_plot = np.append(store_plot, temp_Voc)
                        store_plot_err = np.append(store_plot_err, np.std(temp_Voc))
                    elif np.size(temp_Voc) == 0:
                        print(i)
                        warnings.warn("Not possible to plot this intensity dependent IV", category = UserWarning)
                    else:
                        store_plot = np.append(store_plot, np.mean(temp_Voc))
                        store_plot_err = np.append(store_plot_err, np.std(temp_Voc))
                
            # Plot directly here
            if np.size(store_plot) > 0:
                # Convert optical density to power

                x = 100 * 10.**(-np.unique(nbs_temp[:, 0]) / 10)
                x = x.astype(float)
                store_plot = store_plot.astype(float)
                # x = 100 * np.exp(-np.unique(nbs_temp[:, 0]) / 10)
                # pyqtRemoveInputHook()
                # set_trace()

                try:
                    self._ax.errorbar(x, store_plot, yerr = store_plot_err,
                            fmt = "s", capsize = 10, label = self.GroupNames.astype('U13')[k], c = self.GroupColor.astype('U7')[k])
                except:
                    print(i)
                    warnings.warn("Not possible to plot this intensity dependent IV", category = UserWarning)

                # The unit for the fitting is not important (To convert to SI
                # units one would have to multiply by x by 10) since by
                # subtracting two logarithms the unit just drops out.
                try:
                    popt, pcov = curve_fit(lambda t,a,b: a * np.log(t) + b, x, store_plot, maxfev = 10000)
                    # fit_p, cov = np.polyfit(np.log(x), store_plot, 1, cov = True)

                    # self._ax.plot(x, fit_p[0] * np.log(x) + fit_p[1], "-", c = self.GroupColor.astype('U7')[k],
                            # linewidth = self.linew)
                    self._ax.plot(x, popt[0] * np.log(x) + popt[1], "-", c = self.GroupColor.astype('U7')[k],
                            linewidth = self.linew)

                    # Cov is the covariance matrix with the variances on the diagonals
                    print(self.GroupNames.astype('U13')[k])

                    # Divide by kBT/q as it is convention
                    print("Slope: (" + str(np.round(popt[0] / (kBoltzmann * TRoom) * eElectron, 2)) +
                            " +- " + str(np.round(pcov[0][0] / (kBoltzmann * TRoom) * eElectron, 2)) + ") kBT/q")
                    print("Offset: " + str(popt[1]))
                except:
                    warnings.warn("Fit for " + str(self.GroupNames.astype('U13')[k]) + " could not be done for some reason", category = UserWarning)

        self._ax.set_xscale("log")
        self._ax.autoscale(enable = True, tight = True)
        self._ax.set_ylim(self._ax.get_ylim()[0], self._ax.get_ylim()[1] * 1.02)
        self._ax.set_xlim(self._ax.get_xlim()[0] * 0.8, self._ax.get_xlim()[1] * 1.2)
        self._ax.grid(True)
        self._ax.set_ylabel("V$_{OC}$ (V)", fontsize = self.fonts_)
        self._ax.set_xlabel("Light Intensity (mW/cm$^2$)", fontsize = self.fonts_)
        # loc = plticker.MultipleLocator(base=tickInterval_) # this locator puts ticks at regular intervals
        # self._ax.xaxis.set_major_locator(loc)

        # Define ticks
        # minorLocator = AutoMinorLocator()
        # self._ax.xaxis.set_minor_locator(minorLocator)
        # self._ax_top.xaxis.set_minor_locator(minorLocator)
        self._ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        # self._ax.get_xaxis().get_major_formatter().set_useOffset = False
        # self._ax.ticklabel_format(useOffset=False, style='plain')
        self._ax.xaxis.set_major_formatter(plticker.ScalarFormatter())
        
        self._ax.tick_params(width = self.ax_ticks_width, length = self.tick_length)
        self._ax.tick_params(which = "minor", direction = "in", bottom = True, top = True, length = self.tick_length/4*3, width= self.ax_ticks_width)
        self._ax.tick_params(which = "major", direction = "in", bottom = True, top = True, width = self.ax_ticks_width)
        self._ax.tick_params(which = "major", direction = "in", left = True, right = True, width = self.ax_ticks_width)

        for axis in ['top','bottom','left','right']:
          self._ax.spines[axis].set_linewidth(self.ax_ticks_width)

        # Make room for x label
        self.fig.figure.subplots_adjust(left = 0.20, bottom = 0.20)

        # Use a data cursor in the matplotlib graph
        # datacursor(self.lines)

        # Set tick fontsizes
        for tick in self._ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)
        for tick in self._ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(self.fonts_)

        # Set tight layout
        self.fig.figure.tight_layout()

        # Define legend which shall be draggable and by pressing on elements they hide
        # get handles
        handles, labels = self._ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        # use them in the legend
        leg = self._ax.legend(handles, labels, frameon = False, fontsize = self.fonts_, loc = self.legend_pos)
        leg.get_frame().set_alpha(0.4)
        leg.set_draggable(True) # Make it draggable


        self.fig.draw()

    # -----------------------------------------------------
    # --- Section for Functions Accessible from EQE Tab ---
    # -----------------------------------------------------

    def load_EQE(self):
        # Function that loads the EQE data in and does the first plotting

        self.isWhat = "EQE"
        filepath = self.globalPath + "/EQE/"
        files = np.empty(0)
        scan_nb = 1
        dev_nb = 1

        files = np.array([f for f in glob.glob(filepath + "*_d*_" 
            + "*_EQE_0" + str(scan_nb) + ".txt")])

        for nb_files in range(len(files)):
            regex = re.compile(r'\d+')
            extracted_nbs = [int(x) for x in regex.findall(files[nb_files])]

            pixel_nb = [int(x) for x in regex.findall(files[nb_files].rsplit('/', 1)[-1].rsplit('_', 5)[3])][0]
            # pixel_nb = extracted_nbs[2] 
            try:
                dev_nb = int(files[nb_files].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:])
            except:
                dev_nb = files[nb_files].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:]

            self.EQElabels = np.append(self.EQElabels, "d" + str(dev_nb) + "p" + 
                    str(pixel_nb) + "s" + str(scan_nb))

        self.isWhat = "EQE"
        # filepath = "./EQE/"
        # files = [f for f in glob.glob(filepath + "*.txt")]
        self.line = {}
        # Clear plot (necessary to plot new stuff)
        self.fig.figure.clf()
        self.fig.draw()

        # Do plotting and formatting
        xlim = (350, 800)
        ylim = (0, 100)
        names = ["wavelength", "device_current", "incident_light_int", "EQE"] # Name of columns in files to read in
        xylabel = ["Wavelength (nm)", "EQE (%)"]
        tickInterval = 100
        plotVars = ["wavelength", "EQE"]
        dataToPlot, labels = self.readData(files, names, plotVars)
        self._ax = self.fig.figure.subplots()
        self.plot(dataToPlot, self.EQElabels, xylabel, tickInterval, ylim_ = ylim, ax_ = self._ax)
        self.updateToolbar("EQE")

    # -----------------------------------------------------
    # -- Section for Functions Accessible from UVVIS Tab --
    # -----------------------------------------------------

    def load_UVVIS(self):
        # Function that loads the UVVIS data in and does the first plotting
        try:
            # Catch the error that this is not true

            if self.isWhat == "UVVIS" and self.isnorm == False:
                xlim = self._ax.get_xlim()
                ylim = self._ax.get_ylim()
            elif self.isWhat == "UVVIS" and self.isnorm == True:
                xlim = self._ax.get_xlim()
                ylim = None
            else:
                xlim = None
                ylim = None

            filepath = self.globalPath + "/UV-Vis/"
            files = [f for f in glob.glob(filepath + "*.txt")]
            self.line = {}
            # Clear plot (necessary to plot new stuff)
            self.fig.figure.clf()
            self.fig.draw()

            # Do plotting and formatting
            names = ["wavelength", "absorption"] # Name of columns in files to read in
            xylabel = ["Wavelength (nm)", "Absorption (a.u.)"]
            tickInterval = 100
            plotVars = ["wavelength", "absorption"]
            dataToPlot, labels = self.readData(files, names, plotVars)
            self._ax = self.fig.figure.subplots()
            self.plot(dataToPlot, labels, xylabel, tickInterval, ax_ = self._ax,
                    xlim_ = xlim, ylim_ = ylim, yticksoff_ = True)
            self.updateToolbar("UVVis")
            self.isWhat = "UVVIS"

        except:
            warnings.warn("UV-VIS data can't be plotted. Most probably there is no valid UV-VIS data.", category = UserWarning)


    def plot_UVVisNorm(self):
        # Function that plots the UVVIS data normalized 
        try:
            if self.isnorm == False:
                filepath = self.globalPath + "/UV-Vis/"
                files = [f for f in glob.glob(filepath + "*.txt")]
                self.line = {}

                # Do plotting and formatting
                xlim = self._ax.get_xlim()
                ylim = (0, 1.1) 
                names = ["wavelength", "absorption"] # Name of columns in files to read in
                xylabel = ["Wavelength (nm)", "Normalized Absorption (a.u.)"]
                tickInterval = 100
                plotVars = ["wavelength", "absorption"]

                # Clear plot (necessary to plot new stuff)
                self.fig.figure.clf()
                self.fig.draw()
                dataToPlot, labels = self.readData(files, names, plotVars)

                # Do normalization of the data
                # Only do the normalization in the chosen limits
                # (Rather lengthy expressions but this was the best, fastest I could come up with)
                xDataLims = np.reshape(dataToPlot[0::2, :][np.logical_and(dataToPlot[0::2, :] >= xlim[0], dataToPlot[0::2, :] <= xlim[1])],
                        (int(dataToPlot.shape[0] / 2), int(np.size(dataToPlot[0::2, :][np.logical_and(dataToPlot[0::2, :] >= xlim[0], dataToPlot[0::2, :] <= xlim[1])])
                            / dataToPlot.shape[0] * 2)))
                yDataLims = np.reshape(dataToPlot[1::2, :][np.logical_and(dataToPlot[0::2, :] >= xlim[0], dataToPlot[0::2, :] <= xlim[1])],
                        (int(dataToPlot.shape[0] / 2), int(np.size(dataToPlot[0::2, :][np.logical_and(dataToPlot[0::2, :] >= xlim[0], dataToPlot[0::2, :] <= xlim[1])])
                            / dataToPlot.shape[0] * 2)))


                # Normalize data (to difference between minimum and maximum) of each subarray
                for nb in range(yDataLims.shape[0]):
                    yDataLims[nb, :] = (yDataLims[nb, :] - np.min(yDataLims[nb, :])) / np.max(yDataLims[nb, :] - np.min(yDataLims[nb, :]))
                
                # Save in the right format to give it to the plot function
                dataNorm = np.empty((int(dataToPlot.shape[0]), xDataLims.shape[1]))
                dataNorm[0::2, :] = xDataLims 
                dataNorm[1::2, :] = yDataLims 
                self._ax = self.fig.figure.subplots()
                
                # Plot the data
                self.plot(dataNorm, labels, xylabel, tickInterval, ax_ = self._ax,
                    xlim_ = xlim, ylim_ = ylim, yticksoff_ = True)

                self.isnorm = True 

            # If it is already a normalized plot, do plot as before
            else:
                self.load_UVVIS()
                self.isnorm = False

        except:
            warnings.warn("Some error happened. Normalization is not possible.", category = UserWarning)


    def fit_UVVis_dialog(self):
        # Dialog to fit the UVVis onset. Here the boundaries of the left and
        # right fit can be inserted. On the long term this can be automatized
        # as well so that the dialog becomes unnecessary.

        # Define dialog in which parameters should be entered
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('Fit Dialog')

        # Define all the layouts and labels so that the window looks good
        verticalLayout = QtWidgets.QVBoxLayout()

        horizontalLayout1 = QtWidgets.QHBoxLayout()
        Title = QtWidgets.QLabel("Enter the limits for the two fits of the onset")
        horizontalLayout1.addWidget(Title)

        horizontalLayout2 = QtWidgets.QHBoxLayout()
        fit1 = QtWidgets.QLabel("Left Fit:")

        horizontalLayout3 = QtWidgets.QHBoxLayout()
        self.fit1LowerLim = QtWidgets.QLineEdit()
        self.fit1UpperLim = QtWidgets.QLineEdit()
        Fit1LowerLabel = QtWidgets.QLabel("Lower Boundary Fit 1")
        Fit1UpperLabel = QtWidgets.QLabel("Upper Boundary Fit 1")
        nmLabel1 = QtWidgets.QLabel("nm")
        nmLabel2 = QtWidgets.QLabel("nm")
        horizontalLayout3.addWidget(Fit1LowerLabel)
        horizontalLayout3.addWidget(self.fit1LowerLim)
        horizontalLayout3.addWidget(nmLabel1)
        horizontalLayout3.addWidget(Fit1UpperLabel)
        horizontalLayout3.addWidget(self.fit1UpperLim)
        horizontalLayout3.addWidget(nmLabel2)

        horizontalLayout4 = QtWidgets.QHBoxLayout()
        fit1 = QtWidgets.QLabel("Right Fit:")

        horizontalLayout5 = QtWidgets.QHBoxLayout()
        self.fit2LowerLim = QtWidgets.QLineEdit()
        self.fit2UpperLim = QtWidgets.QLineEdit()
        nmLabel3 = QtWidgets.QLabel("nm")
        nmLabel4 = QtWidgets.QLabel("nm")
        Fit2LowerLabel = QtWidgets.QLabel("Lower Boundary Fit 2")
        Fit2UpperLabel = QtWidgets.QLabel("Upper Boundary Fit 2")
        horizontalLayout5.addWidget(Fit2LowerLabel)
        horizontalLayout5.addWidget(self.fit2LowerLim)
        horizontalLayout5.addWidget(nmLabel3)
        horizontalLayout5.addWidget(Fit2UpperLabel)
        horizontalLayout5.addWidget(self.fit2UpperLim)
        horizontalLayout5.addWidget(nmLabel4)

        horizontalLayout6 = QtWidgets.QHBoxLayout()
        self.fitButton = QtWidgets.QPushButton("Fit")
        self.fitButton.clicked.connect(self.fit_UVVis)

        self.exitButton = QtWidgets.QPushButton("Close")
        self.exitButton.clicked.connect(functools.partial(self.close_dialog, dialog))
        horizontalLayout6.addWidget(self.exitButton)
        horizontalLayout6.addWidget(self.fitButton)

        verticalLayout.addLayout(horizontalLayout1)
        verticalLayout.addLayout(horizontalLayout2)
        verticalLayout.addLayout(horizontalLayout3)
        verticalLayout.addLayout(horizontalLayout4)
        verticalLayout.addLayout(horizontalLayout5)
        verticalLayout.addLayout(horizontalLayout6)

        dialog.setLayout(verticalLayout)

        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.show()
        dialog.exec_()
    
    def fit_UVVis(self):
        # Function that does the actual fitting according to the parameters entered
        # in the dialog

        # Read out the fitting boundaries
        try:
            fit1_boundaries = (int(self.fit1LowerLim.text()), int(self.fit1UpperLim.text()))
            fit2_boundaries = (int(self.fit2LowerLim.text()), int(self.fit2UpperLim.text()))
            fit_plot_sub = 60
        except:
            warnings.warn("Please enter fitting boundaries", category = UserWarning)

        if self.graph_counter > 0:
            for i in range(2 * self.graph_counter):
                self._ax.lines[len(self._ax.lines) - 1].remove()
            self.graph_counter = 0

        # Get xy data of all visible graphs (only the ones not made invisible by clicking on the legend)
        for linenb in range(len(self._ax.lines)):

            if self._ax.lines[linenb].get_visible():
                self.graph_counter += 1
                wavelength = self._ax.lines[linenb].get_xydata()[:, 0]
                absorption = self._ax.lines[linenb].get_xydata()[:, 1]
                band_gap_wl = self.calcEg(wavelength, absorption, fit1_boundaries, fit2_boundaries)
                print("Estimated bandgap: " + str(np.round(band_gap_wl, 1)) + "nm = " + str(np.round(hPlanck / eElectron * cLight / (band_gap_wl * 10**(-9)), 2)) + "eV")

                # Left fit
                fit1 = self._ax.plot(wavelength[(wavelength > fit1_boundaries[0] - fit_plot_sub) & (wavelength < fit1_boundaries[1] + fit_plot_sub)], 10**(wavelength[(wavelength > fit1_boundaries[0] - fit_plot_sub) & (wavelength < fit1_boundaries[1] + fit_plot_sub)] * self.fitlin(wavelength, absorption, fit1_boundaries)[0] + self.fitlin(wavelength, absorption, fit1_boundaries)[1]), label = "fit1", linewidth = self.linew)

                # Right fit
                fit2 = self._ax.plot(wavelength[(wavelength > fit2_boundaries[0] - fit_plot_sub) & (wavelength < fit2_boundaries[1] + fit_plot_sub)], 10**(wavelength[(wavelength > fit2_boundaries[0] - fit_plot_sub) & (wavelength < fit2_boundaries[1] + fit_plot_sub)] * self.fitlin(wavelength, absorption, fit2_boundaries)[0] + self.fitlin(wavelength, absorption, fit2_boundaries)[1]), label = "fit2", linewidth = self.linew)

        self.fig.draw()

    def fitlin(self, l, dat, ran):
        # Function that does the fitting of the UV Vis curves.

        x = l[(l>ran[0]) & (l<ran[1])]
        y = dat[(l>ran[0]) & (l<ran[1])]
        # slope, intercept, r_value, p_value, std_err = linregress(x, y)
        popt, pcov = curve_fit(lambda t,a,b: np.power(10, (a + b * t)),  x,  y, p0 = [0.01, 0.01], maxfev = 10000)

        intercept = popt[0]
        slope = popt[1]

        return slope, intercept

    def calcEg(self, l, A, st_range, bs_range):
        # Function that calculates the bandgap energy from the fit parameters

        slope_f1, intercept_f1 = self.fitlin(np.array(l), np.array(A), st_range)
        slope_f2, intercept_f2 = self.fitlin(np.array(l), np.array(A), bs_range)

        # Intersection of the two lines
        return (intercept_f2 - intercept_f1) / (slope_f1 - slope_f2)

    def TPV_dialog(self):
        # Here the TPV data can be plotted. This shall later be a dialog. Up to
        # now it plots only all the TPV data

        # Clear plot (necessary to plot new stuff)
        self.line = {}
        self.fig.figure.clf()
        self.fig.draw()

        xlim = [-0.5, 5]
        tickInterval = 0.5 
        names = ["time", "A", "B", "AvA"] # Name of columns in files to read in
        xylabel = ["Time ($\mu$s)", "Normalized Voltage (a.u.)"]
        plotVars = ["time", "AvA"]
        self._ax = self.fig.figure.subplots()

        # Turn around the array (for standard architecture)
        self.plot(self.TPVdata, self.TPVlabels, xylabel, tickInterval, ax_ = self._ax, xlim_ = xlim, yticksoff_ = True)

    def TPC_dialog(self):
        # Here the TPC data can be plotted. This shall later be a dialog. Up to
        # now it plots only all the TPC data

        # Clear plot (necessary to plot new stuff)
        self.line = {}
        self.fig.figure.clf()
        self.fig.draw()

        xlim = [-0.5, 5]
        tickInterval = 0.5 
        names = ["time", "A", "B", "AvA"] # Name of columns in files to read in
        xylabel = ["Time ($\mu$s)", "Normalized Current (a.u.)"]
        plotVars = ["time", "AvA"]
        self._ax = self.fig.figure.subplots()

        # Turn around the array (for standard architecture)
        self.plot(self.TPCdata, self.TPClabels, xylabel, tickInterval, ax_ = self._ax, xlim_ = xlim, yticksoff_ = True)

    def transposeX_dialog(self):
        # The TPV/ TPC data is automatically set to zero. The algorithm that
        # does this is however not perfect. In the future the user shall be
        # able to adjust this also further by hand.

        # Define dialog in which parameters should be entered
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('transpose Time Dialog')

        # Define all the layouts and labels so that the window looks good
        verticalLayout = QtWidgets.QVBoxLayout()

        Title = QtWidgets.QLabel("Where do you want to set time to zero?")

        # Here the self defined slider is used which has built in numbering 
        # for the ticks of the slider
        self.transpose = QtWidgets.QLineEdit()

        horizontalLayout6 = QtWidgets.QHBoxLayout()
        self.transposeButton = QtWidgets.QPushButton("transpose Time")
        self.transposeButton.clicked.connect(self.transposeX)

        self.exitButton = QtWidgets.QPushButton("Close")
        self.exitButton.clicked.connect(functools.partial(self.close_dialog, dialog))
        horizontalLayout6.addWidget(self.exitButton)
        horizontalLayout6.addWidget(self.transposeButton)

        verticalLayout.addWidget(Title)
        verticalLayout.addWidget(self.transpose)
        verticalLayout.addLayout(horizontalLayout6)

        dialog.setLayout(verticalLayout)

        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.exec_()

    def transposeX(self):
        return 0 

    def plot_USR(self):
        # Function that allows to plot from a user defined folder whatever he or
        # she likes. The file however has to be valid. Now it can only read in
        # IV data because the columns are defined that way. In the future this
        # shall be changed by asking the user which columns he or she wants to 
        # plot

        # try:
        self.isWhat = "USR"
        # This window shall enable the user to select a folder
        # of data that shall be plotted (in the given style).
        # try:
        filepath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Open File"))

        # glob has problems with some file path names (no special characters)
        # to work around this one can change in the directory using os.chdir
        current_path = os.getcwd()  # get current working directory
        os.chdir(filepath)
        files = [f for f in glob.glob("*.txt")]

        self.line = {}
        # Clear plot (necessary to plot new stuff)
        self.fig.figure.clf()
        self.fig.draw()
        # self._ax = self.fig.figure.subplots()

        os.chdir(current_path)      # go back to current working directory

    def load_TPVTPC(self):
        # Load in the TPV/TPC data adjust it and do the first plotting

        self.isWhat = "TPVTPC"
        # This window shall enable the user to select a folder
        # of data that shall be plotted (in the given style).
        filepath = self.globalPath + "/TPVTPC/"
        files = [f for f in glob.glob(filepath + "*.txt")]
        self.line = {}
        print(files)

        # Clear plot (necessary to plot new stuff)
        self.fig.figure.clf()
        self.fig.draw()
        files_TPV = np.empty(0, dtype = object)
        type_TPV = np.empty(0, dtype = object)
        nbs_TPV = np.empty([0, 2])
        labels_TPV = np.empty(0, dtype = object)

        files_TPC = np.empty(0, dtype = object)
        type_TPC = np.empty(0, dtype = object)
        nbs_TPC = np.empty([0, 2])
        labels_TPC = np.empty(0, dtype = object)

        for fileNb in range(np.size(files)):
            regex = re.compile(r'\d+')
            extracted_codes = files[fileNb].rsplit('/', 1)[-1].split('_', 2)
            extracted_nbs = [int(x) for x in regex.findall(extracted_codes[-1])]

            if extracted_codes[0] == "TPV":
                files_TPV = np.append(files_TPV, files[fileNb])
                type_TPV = np.append(type_TPV, extracted_codes[1])
                nbs_TPV = np.append(nbs_TPV, [[extracted_nbs[0], extracted_nbs[1]]], axis = 0)

            elif extracted_codes[0] == "TPC":
                files_TPC = np.append(files_TPC, files[fileNb])
                type_TPC = np.append(type_TPC, extracted_codes[1])
                nbs_TPC = np.append(nbs_TPC, [[extracted_nbs[0], extracted_nbs[1]]], axis = 0)

        # Sort everything to obtain sorted data
        sortingIdxTPV = np.lexsort((nbs_TPV[:, 1], nbs_TPV[:, 0], type_TPV))
        sortingIdxTPC = np.lexsort((nbs_TPC[:, 1], nbs_TPC[:, 0], type_TPC))

        files_TPV = files_TPV[sortingIdxTPV]
        type_TPV = type_TPV[sortingIdxTPV]
        nbs_TPV = nbs_TPV[sortingIdxTPV]

        files_TPC = files_TPC[sortingIdxTPC]
        type_TPC = type_TPC[sortingIdxTPC]
        nbs_TPC = nbs_TPC[sortingIdxTPC]

        # Do plotting and formatting
        names = ["time", "A", "B", "AvA"] # Name of columns in files to read in
        xylabel = ["Time ($\mu$s)", "Normalized Voltage (a.u.)"]
        plotVars = ["time", "AvA"]
        dataToPlot, labels = self.readData(files_TPV, names, plotVars)

        # Turn around the array (for standard architecture)
        dataToPlot[1::2, :] = np.negative(dataToPlot[1::2, :])

        # Let's find out where the voltage starts to decrease. 
        # The minimum value of the gradient should be the first one where the real
        # decay starts, since we expect an exponential decay.
        # To get rid of fluctuations, the mean is taken. (The last few values
        # of the array must be cut.)
        # meanX = np.mean(dataToPlot[0, 0:int(np.size(dataToPlot[0, :]) / meanOver) * meanOver].reshape(-1, meanOver), axis=1)
        for k in range(int(np.shape(dataToPlot)[0] / 2)):
            meanOver = 20 
            meanY = np.mean(dataToPlot[1 + 2 * k, 0:int(np.size(dataToPlot[1 + 2 * k, :]) 
                / meanOver) * meanOver].reshape(-1, meanOver), axis=1)
            meanOverNorm = 100
            meanYNorm = np.mean(dataToPlot[1 + 2 * k, 0:int(np.size(dataToPlot[1 + 2 * k, :]) 
                / meanOverNorm) * meanOverNorm].reshape(-1, meanOverNorm), axis = 1)

            # Do move the data close to zero
            dataToPlot[2 * k, :] = dataToPlot[2 * k, :] - dataToPlot[2 * k, meanOver 
                    * (np.where(np.min(np.gradient(meanY)) == np.gradient(meanY))[0] - 1)]
            
            # Now do normalize the data in the region next to the point of interest
            dataToPlot[2 * k + 1, :] = ((dataToPlot[2 * k + 1, :] - np.min(meanYNorm)) 
                / (np.max(meanYNorm) - np.min(meanYNorm)))

        self.TPVdata = dataToPlot
        self.TPVlabels = labels

        dataToPlot, labels = self.readData(files_TPC, names, plotVars)

        dataToPlot[1::2, :] = np.negative(dataToPlot[1::2, :])

        for k in range(int(np.shape(dataToPlot)[0] / 2)):
            meanOver = 10 
            meanY = np.mean(dataToPlot[1 + 2 * k, 0:int(np.size(dataToPlot[1 + 2 * k, :]) 
                / meanOver) * meanOver].reshape(-1, meanOver), axis=1)
            meanOverNorm = 200
            meanYNorm = np.mean(dataToPlot[1 + 2 * k, 0:int(np.size(dataToPlot[1 + 2 * k, :]) 
                / meanOverNorm) * meanOverNorm].reshape(-1, meanOverNorm), axis = 1)

            # Do move the data close to zero
            dataToPlot[2 * k, :] = dataToPlot[2 * k, :] - dataToPlot[2 * k, meanOver 
                    * (np.where(np.min(np.gradient(meanY)) == np.gradient(meanY))[0] - 2)]
            
            # Now do normalize the data in the region next to the point of interest
            dataToPlot[2 * k + 1, :] = ((dataToPlot[2 * k + 1, :] - np.min(meanYNorm)) 
                / (np.max(meanYNorm) - np.min(meanYNorm)))

        self.TPCdata = dataToPlot
        self.TPClabels = labels

        xlim = [-0.5, 5]
        tickInterval = 0.5 
        self._ax = self.fig.figure.subplots()
        self.plot(dataToPlot, labels, xylabel, tickInterval, ax_ = self._ax, xlim_ = xlim, yticksoff_ = True)
        self.updateToolbar("TPVTPC")
        # dataToPlot, labels = self.readData(files, skipr, names, plotVars)
    # except:
        # warnings.warn("Something went wrong", category = UserWarning)

    def mobilityMeasurement(self):

        # Function that loads the mobility measurements 

        self.isWhat = "mobility"
        nbsDIV_temp = np.empty([0, 3])
        DIVlabels_temp = np.empty(0, dtype = "object")
        filesDIV = np.empty(0)
        filepath = self.globalPath + "/mobility/"
        scan_nbDIV = 1
        dev_nb = 1

        # Now try if user decided to enter a number or something else in the device
        # field of the file name
        filesDIV = np.array([f for f in glob.glob(filepath + "*_d*_" 
            + "*_DIV_0" + str(scan_nbDIV) + ".txt")])

        temp_dvNb = filesDIV[0].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:]
        if temp_dvNb.isdigit():
            self.devNbisNb = True
        else:
            self.devNbisNb = False

        # Iterate over nb of files
        for nb_files in range(len(filesDIV)):
            # extracted_nbsDIV = [int(x) for x in regex.findall(files_tempDIV[nb_files].rsplit('/', 1)[-1])]
            # pixel_nbDIV = extracted_nbsDIV[2] 
            regex = re.compile(r'\d+')
            pixel_nbDIV = [int(x) for x in regex.findall(filesDIV[nb_files].rsplit('/', 1)[-1].rsplit('_', 5)[3])][0]

            # else:
            dev_nb = filesDIV[nb_files].rsplit('/', 1)[-1].rsplit('_', 4)[1][1:]

            nbsDIV_temp = np.append(nbsDIV_temp, [[dev_nb, pixel_nbDIV, int(scan_nbDIV)]], axis = 0)
            DIVlabels_temp = np.append(DIVlabels_temp, "d" + str(dev_nb) + "p" + 
                    str(pixel_nbDIV) + "s" + str(scan_nbDIV))

        # glob doesn't read in data in an order. To fix this the arrays have to 
        # be sorted
        sortingIdxDIV = np.lexsort((nbsDIV_temp[:,1], nbsDIV_temp[:,0]))

        self.nbsDIV = nbsDIV_temp[sortingIdxDIV]

        # pyqtRemoveInputHook()
        # set_trace()
        if self.devNbisNb:
            self.nbsDIV = self.nbsDIV.astype(int)

        # pyqtRemoveInputHook()
        # set_trace()

        filesDIV = filesDIV[sortingIdxDIV]
        self.DIVlabels = DIVlabels_temp[sortingIdxDIV]

        if np.size(self.IVselected) == 0:
            self.IVselected = np.repeat(True, np.size(filesDIV))

        self.line = {}
        self.fig.figure.clf()
        self.fig.draw()

        # Plot the selected data with the correct labels

        # Do plotting and formatting
        xylabel = ["Voltage (V)", "Current (mA/cm$^2$)"]
        tickInterval = 0.4
        names = ["DV", "DC"] # Name of columns in files to read in
        plotVars = ["DV", "DC"]
        self.DIVdata, labels = self.readData(filesDIV, names, plotVars)
        self.DIVdata[1::2] = self.DIVdata[1::2] * 1000 / (4.5 / 100)
        self._ax = self.fig.figure.subplots()

        self.plot(self.DIVdata, self.DIVlabels, xylabel, tickInterval, ax_ = self._ax,
                IV_ = True, zeroLines_ = True)
        self.updateToolbar("mobility")

    def extractMobility(self):
        # Function to extract mobility from selected IV curves

        tot_selected = np.empty(np.size(self.IVselected) * 2, dtype = bool)
        tot_selected[0::2] = self.IVselected
        tot_selected[1::2] = self.IVselected
        nbs_selected = self.nbsDIV[self.IVselected]
        # self.DIVdata[tot_selected]
        self.slopeMobility = np.zeros(np.shape(self.nbsDIV)[0])
        self.x_new = np.zeros(np.shape(self.nbsDIV)[0])
        self.y_new = np.zeros(np.shape(self.nbsDIV)[0])
        self.x_min = np.zeros(np.shape(self.nbsDIV)[0])
        self.x_max = np.zeros(np.shape(self.nbsDIV)[0])

        # print(int(np.shape(self.DIVdata[tot_selected])[0] / 2))
        # Go over all DIV data and sort by their group
        nb_curves = np.zeros(self._nbGroups)
        mobility_sum = np.zeros(self._nbGroups)
        for m in range(self._nbGroups):
            for k in self.GroupNbs[m]:
                for i in range(np.shape(self.nbsDIV)[0]):
                    if self.IVselected[i] == True and k == self.nbsDIV[i,0]:
                        self.slopeMobility[i], self.x_new[i], self.y_new[i], self.x_min[i], self.x_max[i] = self.fitMobility(self.DIVdata[(2*i):(2*i+2)])
                        mobility_sum[m] = mobility_sum[m] + (self.slopeMobility[i] 
                                * (8 * (self.ThicknessLayer[m] * 10**(-9))**3) 
                                / (9 * 3 * 8.854 * 10**(-12)) * 100000)
                        # mobility_sum[m], self.slopeMobility[i] = [mobility_sum[m], 0] + self.fitMobility(self.DIVdata[(2*i):(2*i+2)])
                        if self.slopeMobility[i] != 0:
                            nb_curves[m] += 1
                        
        mobility_group = mobility_sum / nb_curves

        for i in range(np.size(mobility_group)):
            print("Group " + str(self.GroupNames[i]) + ": " + str(np.round(mobility_group[i], 8)) + " cm2/Vs")


    def fitMobility(self, dat):
        # Function that does the fitting for the DIV curves to extract the mobility 

        # Only select data to the maximum of the y data
        xdata_temp = dat[0]
        ydata_temp = dat[1]

        # pyqtRemoveInputHook()
        # set_trace()
        try:
            xdata = xdata_temp[np.logical_and(np.gradient(np.log10(ydata_temp), np.log10(xdata_temp))> 1.6, 
                np.gradient(np.log10(ydata_temp), np.log10(xdata_temp)) < 2.5)]
            ydata = ydata_temp[np.logical_and(np.gradient(np.log10(ydata_temp), np.log10(xdata_temp)) > 1.6,
                np.gradient(np.log10(ydata_temp), np.log10(xdata_temp)) < 2.5)]

            # slope, intercept, r_value, p_value, std_err = linregress(x, y)
            # y times 10 to convert mA/cm^2 in A/m^2
            popt, pcov = curve_fit(lambda t,a,b,c: a * (t - b) ** 2 + c, xdata, ydata, maxfev = 10000)

            print(np.min(xdata))
            print(np.max(xdata))

            return popt[0], popt[1], popt[2], np.min(xdata), np.max(xdata)
        except:
            warnings.warn("Could not fit the curve probably it is nowhere quadratic", category = UserWarning)
            return 0,0,0,0,0

    def load_ELQE(self):

        self.isWhat = "ELQE"
        filepath = self.globalPath + "/ELQE/"
        files = np.empty(0)
        scan_nb = 1
        dev_nb = 1
        filesELQE = np.array([f for f in glob.glob(filepath + "*_LIVdata.txt")])


        self.line = {}
        # Clear plot (necessary to plot new stuff)
        self.fig.figure.clf()
        self.fig.draw()
        xlim = (0, 1000)
        ylim = (0, 1)
        xylabel = ["Current (mA/cm$^2$)", "ELQE (%)"]
        tickInterval = 100
        names = ["V", "C", "OP", "CandA", "Candm", "EQE" ] # Name of columns in files to read in
        plotVars = ["C", "EQE"]
        dataToPlot, labels = self.readData(filesELQE, names, plotVars)
        self._ax = self.fig.figure.subplots()
        self.plot(dataToPlot, labels, xylabel, tickInterval, ax_ = self._ax, log_ = True, 
                xlim_ = xlim, ylim_ = ylim)
        self._ax.yaxis.set_major_formatter(plticker.ScalarFormatter())
        self.updateToolbar("ELQE")


    def closeEvent(self, event):
        # If no group names have been defined the dialog can be killed by
        # pressing the x on top
        if np.size(self.GroupNames) == 0:
            event.accept()
            return

        # Otherwise ask if the user wants to save before closing. The last
        # argument sets the standard button for pressing enter
        quit_msg = "Do you want to save the status of your program before closing?"
        reply = QtWidgets.QMessageBox.question(self, 'Message', 
                        quit_msg, QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Close,
                        QtWidgets.QMessageBox.Save)

        if reply == QtWidgets.QMessageBox.Close:
            event.accept()
        else:
            self.saveFile()
            event.accept()
        
#-------------------------------------------------------------------------------
#---------------------------------- MAIN ---------------------------------------
#-------------------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = main_window()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()