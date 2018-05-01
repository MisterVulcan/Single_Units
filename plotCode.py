# utilFuncs module contains all of our helpful functions for converting data and sorting tdc
# analysis results
import utilFuncs

import numpy as np
import pandas as pd

# We're going to make everything offline because we're too poor to afford cloud hosting
import plotly.offline as offline
import plotly.graph_objs as go

def spikeScatter(dfUnits, histType='sc', groupBy='Unit', xBin=40, yBin=36, xDim=1024, yDim=768,
                    tsInt=.03125):
    """
    Creates a plot.ly graph from the provided pandas Dataframe, showing firing location for each spike
    as a scatter point. Optionally creates a 2D histogram with spike scatter plot as an overlay. Note
    that histogram is calculated for all scatter points in the dataframe and does not distinguish by
    group

    Inputs:
        dfUnits:    A pandas dataframe with the following columns:
                    Timestamp       X       Y       Unit        DateTime
        
        histType:   String parameter for controlling histogram type, with following options:
                    'sc' for spike count histogram, 'fr' for avg. firing rate histogram,
                    '' or None for nothing

        groupBy:    String parameter for controlling how scatter points are grouped (color code):
                    'Unit' for unit name, 'DateTime' for recording datetime. For now whichever is
                    used for color coding, the other will be in the hovertext for each point
        
        tsInt:      Float # representing the sampling interval used by cineplex tracker, translates
                    to 


    """

    # Get unit names and segment DateTimes, will be used for grouping
    segDates = pd.unique(dfUnits['DateTime'])

    if groupBy == 'Unit':
        nameList = pd.unique(dfUnits['Unit'])
        # hoverList = pd.unique(dfUnits['DateTime'])
    elif groupBy == 'DateTime':
        nameList = pd.unique(dfUnits['DateTime'])
        # hoverList = pd.unique(dfUnits['Unit'])

    data = []

    # Loop over every group being plotted
    for i in range(len(nameList)):
        # EXPERIMENTAL Using ScatterGL to leverage WebGL rendering, should be useful for when dealing
        # with large numbers of spikes
        trace = go.Scattergl(
            x = dfUnits[dfUnits[groupBy]==nameList[i]]['X'],
            y = dfUnits[dfUnits[groupBy]==nameList[i]]['Y'],
            name = nameList[i],
            mode = 'markers',
        )
        data.append(trace)

    histTrace = go.Histogram2d(

        x = dfUnits['X'],
        autobinx=False,
        xbins=dict(start=0, end=xDim, size=xDim/xBin),

        y = dfUnits['Y'],
        autobiny=False,
        ybins=dict(start=0, end=yDim, size=yDim/yBin),

        zsmooth = '',
        histnorm = '',
        histfunc = '',

        colorbar=dict(x=1.15),
        colorscale='Viridis',
    )
    data.append(histTrace)

    fig = go.Figure(data = data)

    offline.plot(fig)


