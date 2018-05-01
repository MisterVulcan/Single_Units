
import numpy as np
import pandas as pd
import matlab.engine
import os
from pathlib import Path
import json
from datetime import datetime
from tridesclous import DataIO, CatalogueConstructor, Peeler

from oct2py import Oct2Py

import matplotlib.pyplot as plt
import imageio

ac = None
mc = None

def getVT_Data(fileName, oc=mc):
    """
    Utility function that retrieves Cineplex tracking data from a legacy .plx file.
    Currently this is done using the Plexon Matlab SDK as a pure python implementation
    does not exist. As such, the directory containing the SDK .m files
    needs to be on the matlab search path and you need to have compiled the SDK mex file
    and installed the Python Matlab Engine in your python environment. The Plexon SDK auto-
    generates .p files when you compile, so it won't work in Octave. Sorry ;-)

    Inputs:
        fileName:   name of the .plx file to be opened, can be a relative or an absolute path.

        oc:         matlab engine instance object, by default initialized to None. This is only
                    so you don't have to re-initialize a matlab engine instance every time you
                    use this function. 
    Outputs:
        cc:         numpy array containing columns (TS, X_CRD, Y_CRD) for each tracking element.
                    As of this writing (2018.3.12) we only use a single tracking LED, so it isn't
                    necessary to create more than one output array. Each row is a single sample.
    """

    # Initialize or reset the matlab engine
    if not oc:
        oc = matlab.engine.start_matlab()
    else:
        # Clears any existing matlab vars and sets current working directory, in case we're using
        # relative paths
        oc.clear(nargout=0)
        oc.cd(str(os.getcwd()), nargout=0)

    # Retrieve ts event from .plx file. By default, vt data is contained in Event Channel 257, 'Strobed'
    [n, ts, sv] = oc.plx_event_ts(fileName, matlab.double([257]), nargout=3)

    # Transcode the vt data from the event channel, 3rd output of the transcode function
    c = oc.plx_vt_interpret(ts, sv, nargout=4)[3]

    # Convert vt data to a numpy array
    cc = np.asarray(c, dtype=np.float)

    return cc

def getPLX_DateTime(fileName, oc=mc):
    """
    Use the PLX Matlab SDK to parse the datetime string for a legacy .plx file. Returns a python
    datetime.datetime object. WARNING: this doesn't work unless you modify the SDK sourcecode, see
    line comments below.

    Inputs:
        fileName:   Self explanatory, can be a relative or an absolute path

        oc:         matlab engine instance object, same as in getVT_Data
    
    Outputs:
        dT:         Python datetime.datetime object, parsed from the DateTime string returned from
                    plx_information matlab function in the Plexon Matlab SDK.
    """

    if not oc:
        oc = matlab.engine.start_matlab()
    else:
        oc.clear(nargout=0)
        oc.cd(str(os.getcwd()), nargout=0)

    # DateTime output argument for the plx_information function is the last of 13 produced
    dT_str = oc.plx_information(fileName, nargout=13)[-1]

    # Use strptime function to parse the DateTime string into a datetime.datetime function, format
    # string is 'mm/dd/YYYY HH:MM:SS'. Note that I had to modify the `PlexMethods.cpp` file, 
    # located in `Matlab Offline Files SDK/mexPlex/` (line 1458) for this to actually work
    dT = datetime.strptime(dT_str, '%m/%d/%Y %H:%M:%S')

    return dT

def getContChannels(fileName, oc=ac):
    """
    Utility function that retrieves all continuous channel recordings from a .plx file, using a
    3rd-party library derived from the Plexon Matlab SDK (THIS NEEDS TO BE FIXED). Currently (2018.3.12)
    the library is compiled under Octave instead of Matlab, so it uses an oct2py ipython kernel to evaluate
    functions in Octave and results to the python environment. Continuous channel data is written as a binary
    data file ('.dat'), usable in most cluster sorting software (e.g. klusta, tridesclous)

    Inputs:
        fileName:   Same scheme as in getVT_Data function

        oc:         Ditto, except it's an Oct2Py instance instead of Matlab Python Engine & uses a different
                    library
    Outputs:
        None

    """

    if not oc:
        oc = Oct2Py()
    else:
        oc.restart()

    oc.eval("a = readPLXFileC('" + fileName + "', 'continuous');", verbose=False)
    oc.eval("b = {a.ContinuousChannels(:).Values};", verbose=False)
    oc.eval("c = cat(2, b{:});")

    c = oc.pull("c")
    d = np.reshape(c, c.size)

    d.tofile(fileName[:-4] + ".dat")

    # oc.exit()

def getSortedTimes(dirName, chanGroup):

    dataio = DataIO(dirname=dirName)
    dataio.load_catalogue(chan_grp=chanGroup)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    sample_rate = dataio.sample_rate # Just initialize sample rate, will set later

    unitTimes = np.empty(dataio.nb_segment, dtype=object)

    for j in range(dataio.nb_segment):
        
        idd = {}
        times = {}

        try:
            # List of all cluster labels
            cluster_ids = np.array([i for i in catalogueconstructor.cluster_labels])
            # List of all detected peaks by cluster ID     
            clusters = np.array([i[1] for i in dataio.get_spikes(j)])

            spike_times = np.array([i[0] for i in dataio.get_spikes(j)])

        except:
            cluster_ids  = np.array([])
            clusters = np.array([])
            spike_times = np.array([])

        for i in cluster_ids:
            idd[i] = np.argwhere(clusters == i)

        for i in cluster_ids:
            times[i] = spike_times[idd[i]]/sample_rate  

        mx = np.max([times[i].size for i in times.keys()])

        for i in times.keys():
            times[i].resize(mx + 1, 1)

        timesArray = np.array([times[i] for i in times.keys()])

        timesArray = np.roll(timesArray, 1)
        timesArray[:, 0, :] = np.array(list(times.keys())).reshape(timesArray.shape[0], 1)

        timesArray = np.transpose(timesArray)

        unitTimes[j] = timesArray[0]
    
    return unitTimes

def spikeBins(vtTS, unitTS, ignBad=True):
    """
    Preps unit spike data for location binning by first sorting each detected spike into a time bin
    and otherwise cleaning up

    Inputs:
        vtTS:       Array of vectors containing time stamps and coordinates for Cineplex (or other)
                    object tracking data. Schema for this input should be:

                    vtTS[0]     vtTS[1]     vtTS[2]     vtTS[3]     vtTS[...]
                    Timestamp   LED1_X      LED1_Y      LED2_X      etc...

                    Shortly, one vector for timestamps and 2 per LED (X,Y) in that order. Needs at
                    least 1 tracking LED (duh), if more than one used the mean (X,Y) coordinates will
                    be used for each timestamp
        
        unitTS:     Array of vectors containing time stamps for each spike for a given unit, each
                    column vector corresponds to a unit

        ignBad:     Bool flag, determines whether to drop spikes that occur at cartesian coord. (0,0),
                    which is where the tracker automatically goes (ours does, anyway) when it loses its
                    lock on the tracked object
    Outputs:
        unitCube:   A 3D array divided into 3 slices - slice 0 contains the spike times for
                    each unit (columnwise), slice 1 contains the corresponding X coordinate, slice
                    2 contains Y coordinate
    """
    # Initialize unitCube array
    unitCube = np.zeros([3, unitTS.shape[0], unitTS.shape[1]])

    # Slice 0 contains unit spike times
    unitCube[0] = unitTS

    # Use the numpy digitize function to bin spike times based on the time stamps from the tracking
    # data, uses binary search to sort so it should be pretty efficient in most cases
    indices = np.digitize(unitTS, vtTS[:,0])

    # Correct for border cases in the indices array to avoid indexing errors. This is rare, but it happens
    indices[indices >= len(vtTS[:,0])] = len(vtTS[:,0]) - 1

    # Copy the X, Y coordinates corresponding to each time bin that each spike belongs to
    unitCube[1] = vtTS[indices,1]
    unitCube[2] = vtTS[indices,2]

    # NaN-out filler zeros in spike time slice and corresponding X,Y coordinates
    unitCube[0][unitCube[0] == 0] = np.nan
    unitCube[1][np.isnan(unitCube[0])] = np.nan
    unitCube[2][np.isnan(unitCube[0])] = np.nan

    # If we want to filter out bad tracking data
    if ignBad:
        # NaN-out bad values where tracking failed, coordinate tuple is (0,0), as well as
        # corresponding spike times in slice 0
        unitCube[1][unitCube[1] == 0] = np.nan
        unitCube[2][unitCube[2] == 0] = np.nan
        unitCube[0][np.isnan(unitCube[1])] = np.nan

        # NaN-out bad values in the tracking array
        vtTS[vtTS[:,1] == 0] = np.nan
    
    return unitCube

def plc_Analysis(vtTS, unitCube, xBin=40, yBin=36, xDim=1024, yDim=768, ignBad=True, normed=True):
    """
        Conducts a place cell analysis - binned spike counts per unit, mean firing rate per bin -
        for each given unit with the provided tracking data. This function uses numpy's built-in 
        2D histogram functions, probably won't be used if using plot.ly API
        
        Inputs:
            vtTS:       Array of vectors containing time stamps and coordinates for Cineplex (or other)
                        object tracking data. Schema for this input should be:

                        vtTS[0]     vtTS[1]     vtTS[2]     vtTS[3]     vtTS[...]
                        Timestamp   LED1_X      LED1_Y      LED2_X      etc...

                        Shortly, one vector for timestamps and 2 per LED (X,Y) in that order. Needs at
                        least 1 tracking LED (duh), if more than one used the mean (X,Y) coordinates will
                        be used for each timestamp

            unitCube:   A 3D array divided into 3 slices - slice 0 contains the binned spike times for
                        each unit (columnwise), slice 1 contains the corresponding X coordinate, slice
                        2 contains Y coordinate
            
            minTBin:    (float) Minimum amount of time, in seconds, that the tracked object must have stayed
                        in a bin for a spike detected there to count. Currently this only applies to the
                        TOTAL elapsed time spent in a bin

            xBins:      Number of horizontal bins to divide the map into. If xDim isn't evenly divisible by xBins,
                        will round up to closest integer

            yBin:       Ditto, vertical bins

            xDim:       Horizontal dimension of the map used for tracking

            yDim:       Ditto, vertical dimension

            ignKern:    Kernel size, in pixels, to filter bad location data. If set to 0, ignoreBad feature won't
                        be used. This checks to see if, for a given (X,Y) coordinate at a given timestep, the current
                        coordinate set isn't too different from the chronologically previous set. Main purpose is
                        to get rid of garbage tracking data
            
        Outputs:
            scMap:      (nUnit, xBin, yBin)-sized array containing the number of spike counts for each bin
                        for each unit, where nUnit is the number of units provided
            
            frMap:      (nUnit, xBin, yBin)-sized array containing the mean firing rate for each unit within each bin
    """
    # Number of units to be binned
    numUnits = unitCube.shape[2]

    # Time interval between tracking samples varies (a really small amount, but it's there), so use
    #  mean interval to calculate time spent in each bin
    timeInt = vtTS[-1,0] / vtTS.shape[0]

    # scMap is a 3d array with n slices, each of size (xBin, yBin), contains spike counts for each bin
    scMap = np.zeros([numUnits, xBin, yBin])

    # tsMap is a 2d array of size (xBin, yBin), will contain the time spent in each bin
    tsMap = np.zeros([xBin, yBin])

    # frMap is a 3d array with n slices, each of size (xBin, yBin), will contain the mean firing rate for each bin
    frMap = np.zeros([numUnits, xBin, yBin])

    # Time spent in each bin is just the # of time stamps sampled in each bin ...

    # Only use the value's that aren't nan'd out.
    tsMap = np.histogram2d(vtTS[:,1][~np.isnan(vtTS[:,1])], 
                                vtTS[:,2][~np.isnan(vtTS[:,2])],
                                bins=[xBin, yBin], range=[[0, xDim], [0, yDim]],
                                normed=normed)[0]

    # Flip the resulting histogram vertically to swap order of y-axis
    tsMap = np.flipud(tsMap)

    # Nan-out any bins where no time was spent. Numpy handles division by nan's, otherwise we'd have to
    # deal with 0-division
    tsMap[tsMap == 0] = np.nan

    # Multiply the count per bin by avg. sample interval to get time spent in each bin
    tsMap = tsMap * timeInt

    # Iterate over every unit
    for i in range(numUnits):
        # Spikes per bin
        scMap[i] = np.histogram2d(unitCube[1,:,i][~np.isnan(unitCube[1,:,i])],
                                    unitCube[2,:,i][~np.isnan(unitCube[2,:,i])],
                                     bins=[xBin, yBin], range=[[0, xDim], [0, yDim]],
                                     normed=normed)[0]

        # Flip 
        scMap[i] = np.flipud(scMap[i])

    # Firing rate bin map is # of spikes per bin divided by amount of time spent in each bin
    frMap = scMap / tsMap

    # Get rid of nan values so frMap looks nice when graphed
    frMap[np.isnan(frMap)] = 0

    return np.array([scMap, frMap])

def batchPLC_Analysis(experimentDay, tdcName='tdc_', chanGroup=0, xBin=40, yBin=36, ignBad=True,
                        normed=False, exportCSV=True, exportImage=False):
    """
        Inputs:
            experimentDay:  string value of the path for the test subject name and the day of the recording.
                            e.g.: 'T_XX/MM-DD' is the path for subject T_XX on MM-DD
    """

    # Get the absolute path for the directory containing all recordings for one day
    parentDir = os.path.normpath(os.path.join(os.getcwd(), experimentDay))

    # NEEDS PYTHON >3.5 TO WORK. Creates a Path object for the experiment directory
    p = Path(parentDir)

    tdcDir = str(p.joinpath(tdcName))
    tdcConfig = str(p.joinpath(tdcName, 'info.json'))

    print("Importing unit spike time data from tdc catalogue...")
    # Get the spike times for each unit for each recording in the catalogue
    tdcData = getSortedTimes(tdcDir, chanGroup)

    unitNames = tdcData[0][0]

    print("done")

    with open(tdcConfig) as json_data:
        datFiles = json.load(json_data)['datasource_kargs']['filenames']

    # This is just a list of the plx files corresponding to .dat files used in tdc
    plxFiles = [datFiles[i][:-4] + '.plx' for i in range(len(datFiles))]

    # segNames = [os.path.splitext(os.path.basename(i))[0] for i in plxFiles]

    print("Parsing plexon recording file information...")

    # plxTS is a list of DateTime objects for each recording, parsed from the .plx files
    # Takes a while because we have to use the Plexon Matlab SDK to retrieve metadata
    plxTS = [getPLX_DateTime(i) for i in plxFiles]

    # plxList is a list of tuples, 1st element is absolute filepath, 2nd element is corresponding
    # array of unit spike times, 3rd element is corresponding DateTime object
    plxList = [(plxFiles[i], tdcData[i], plxTS[i]) for i in range(len(plxFiles))]

    # Sort plxList in place chronologically, using the datetime elements
    plxList.sort(key = lambda dt: dt[2])

    print("done")

    print("Importing cineplex tracking data...")

    # vt is an object array that will contain all of the cineplex tracking data from each recording
    vt = np.empty(len(plxFiles), dtype=object)

    # Extract object tracking data from each recording 
    for i in range(len(plxList)):
        vt[i] = getVT_Data(plxList[i][0])

    print("done")

    store_ = np.zeros((len(plxList), 2, tdcData[0].shape[1], xBin, yBin))
    # array for plotly has to be an object array because element arrays are not likely to have the
    # same shape without padding
    store_Plotly = np.empty(len(plxList), dtype=object)

    print("Calculating per unit heat maps...")
    for i in range(len(plxList)):

        print("segment:", i)

        # Get a unitCube for the current recording, leave out the unit header row in tdcData
        dataCube = spikeBins(vt[i], plxList[i][1][1:])

        store_[i] = plc_Analysis(vt[i], dataCube, normed=normed)

        # These routines only apply if we want to format data for use in Plotly
        if exportCSV:
            store_Plotly[i] = dataCube.T

            catShape = (store_Plotly[i].shape[:-1], 2)
            catShape = catShape[0] + (catShape[1],)
            catArray = np.empty(catShape, dtype=None)
            catArray[:] = np.nan

            store_Plotly[i] = np.concatenate((store_Plotly[i], catArray), axis=2)

            store_Plotly[i][:,:,4][~np.isnan(store_Plotly[i][:,:,0])] = i

            for j in range(len(unitNames)):
                store_Plotly[i][j,:,3][~np.isnan(store_Plotly[i][j, :, 0])] = unitNames[j]

    if exportImage:
        img_store_ = np.zeros((len(tdcData), 2, tdcData[0].shape[1], xBin, yBin, 4))
        cmap = plt.cm.viridis

        for i in range(len(tdcData)):
            for j in range(len(unitNames)):
                normSC = plt.Normalize(vmin=store_[i, 0, j].min(), vmax=store_[i, 0, j].max())
                normFR = plt.Normalize(vmin=store_[i, 1, j].min(), vmax=store_[i, 1, j].max())

                img_store_[i, 0, j] = cmap(normSC(store_[i, 0, j]))
                img_store_[i, 1, j] = cmap(normFR(store_[i, 1, j]))

        kargs = {'duration' : 0.5}
        for i in range(len(unitNames)):
            imageio.mimwrite(str(p.joinpath('SC_' + str(unitNames[i]) + '.gif')),
             img_store_[:, 0, i], 'GIF', **kargs)
            imageio.mimwrite(str(p.joinpath('FR_' + str(unitNames[i]) + '.gif')),
             img_store_[:, 1, i], 'GIF', **kargs)

    if exportCSV:
        # We're going to store all our data as a pandas dataframe, first need to break down datacubes

        # Stack all spike timestamps by unit, then by recording
        store_Plotly = np.concatenate(np.concatenate(store_Plotly, axis=1), axis=0)
        # Get rid of any nan'd out values
        store_Plotly = store_Plotly[~np.any(np.isnan(store_Plotly), axis=1)]
        # dtList will be a column vector of DateTime objects for each valid spike 
        dtCol = np.copy(store_Plotly[:, -1])
        dtCol = dtCol.astype(np.int)
        dtCol = np.array([plxList[i][-1] for i in dtCol])

        # Create initial dataframe from store_Plotly array, leave out last (segNo.) column
        dfUnits = pd.DataFrame(store_Plotly[:,:-1], columns=['Timestamp', 'X', 'Y', 'Unit'])

        # (X,Y) and Unit # don't need to be floats, can change here to make dataframe leaner
        dfUnits['X'] = dfUnits['X'].astype('int16')
        dfUnits['Y'] = dfUnits['Y'].astype('int16')
        dfUnits['Unit'] = dfUnits['Unit'].astype('int16')
        # Add the datetime column
        dfUnits['DateTime'] = dtCol

        # That's it! Now return for plotting!
        return dfUnits

    else:
        return store_, unitNames

def batchPLX_Import(experimentDay):

    # Get the absolute path for the directory containing all recordings for one day
    parentDir = os.path.normpath(os.path.join(os.getcwd(), experimentDay))

    # NEEDS PYTHON >3.5 TO WORK. Creates a Path object for the experiment directory
    p = Path(parentDir)

    # Get a list of every recording file in the directory, hardcoded for .plx files right now
    plxFiles = [str(i) for i in sorted(p.glob("**/*.plx"))]

    for i in range(len(plxFiles)):
        getContChannels(plxFiles[i])

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], interpolation='bilinear')
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional