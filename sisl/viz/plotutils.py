import numpy as np
import itertools
import multiprocessing as mp
import tqdm

import pickle

from .plot import Plot

def calculateGap(bands):
    '''
    Calculates the gap of a set of bands
    
    Arguments
    ----------
    bands: 2d array
        A 2 dimensional array containing the bands for which the gap needs to be calculated
    
    Returns
    ----------
    gap: float
        The value of the gap in the same units that the bands array was in.
    gapLimitsLoc: array of int
        Contains the locations of the gap limits in the bands array provided.
            - First row: highest occupied (bottom limit of gap)
            - Second row: lowest unoccupied (top limit of gap)
    '''
    
    #Initialize the gap limits array
    gapLimits = np.array([-10**3, 10**3], dtype=float)
    gapLimitsLoc = np.array([[0,0], [0,0]], dtype=int)
    
    #
    isAboveFermi = bands > 0
    
    for i, cond in enumerate([~isAboveFermi, isAboveFermi]):
        
        gapLimitsLoc[i, :] = np.argwhere(abs(bands) == np.min(abs(bands[cond])) )[0]
        
        iK, iWF = gapLimitsLoc[i]

        gapLimits[i] = bands[iK, iWF]
    
    gap = np.diff(gapLimits)[0]
    
    return gap, gapLimitsLoc

def calculateSpinGaps(bands):
    '''
    Gets the gap for each spin separately making use of the calculateGap() function.

    Arguments
    -----------
    bands: 3d array
        A 3 dimensional array containing the bands for both spins. The first dimension should be the spin component.
        Tip: You can use np.rollaxis() if they are not originally provided in this way.
    
    Returns
    -----------
    gaps: list of float
        A list with the two gaps (one for each spin)
    gapLimitsLocs: list
        A list containing the two gapLimitsLoc returned by calculateGap().
    '''
    
    gaps = [0,0]
    gapLimitsLocs = [0,0]
    
    for i, spinBands in enumerate(bands):
        
        gaps[i], gapLimitsLocs[i] = calculateGap(spinBands)
        
    return gaps, gapLimitsLocs

def sortOrbitals(orbitals):
    '''
    Function that sorts a list of orbital names scientifically. (1s -> 2s -> etc)

    Arguments
    ---------
    orbitals: list of str
        the list of orbitals to sort
    
    Return
    --------
    sortedOrbs: list
    '''

    def sortKey(x):

        l = "spdfghi"

        return x[0], l.index(x[1]) 

    return sorted(orbitals, key = sortKey)

def initSinglePlot(argsTuple):
    '''
    Initialize a single plot. This function is meant to be used in multiprocessing, when multiple plots need to be initialized
    '''

    PlotClass, args, kwargs = argsTuple

    return PlotClass(**kwargs)._getPickleable()

def initMultiplePlots(PlotClass, argsList = None, kwargsList = None, nPlots = None):
    '''
    Makes use of the multiprocessing module to initialize multiple plots in the fastest way possible

    Arguments
    ----------
    PlotClass: child class of sisl.viz.Plot or array of child classes of sisl.viz.Plot
        If a single class is passed, it will generate all plots with this class.
        If an array of classes is passed, it will iterate over the array to generate the plots.
    argsList
    '''

    #Prepare the arguments to be passed to the initSinglePlot function
    toZip = [PlotClass, argsList, kwargsList]
    for i, arg in enumerate(toZip):
        if not isinstance(arg, (list, tuple, np.ndarray)):
            toZip[i] = itertools.repeat(arg)
        else:
            nPlots = len(arg)
    
    #Create a pool with the appropiate number of processes
    pool = mp.Pool( processes = min(nPlots, mp.cpu_count() - 1) )
    #Define the plots array to store all the plots that we initialize
    plots = [None]*nPlots

    #Initialize the pool iterator and the progress bar that controls it
    progress = tqdm.tqdm(pool.imap(initSinglePlot, zip(*toZip) ), total=nPlots)
    progress.set_description("Reading the files needed in {} processes".format(pool._processes))

    #Run the processes and store each result in the plots array
    for i, res in enumerate(progress):
        plots[i] = res

    return plots

def load(path):

    with open(path, 'rb') as handle:
        plt = pickle.load(handle)
    
    return plt