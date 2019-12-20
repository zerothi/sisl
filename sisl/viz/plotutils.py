import numpy as np
import itertools
from pathos.pools import ProcessPool as Pool
import tqdm

import os
import glob
import pickle
from copy import deepcopy

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

def copyParams(params, only = [], exclude = []):
    '''
    Function that returns a copy of the provided plot parameters.

    Arguments
    ----------
    params: tuple
        The parameters that have to be copied. This will come presumably from the "_parameters" variable of some plot class.
    only: array-like
        Use this if you only want a certain set of parameters. Pass the wanted keys as a list.
    exclude: array-like
        Use this if there are some parameters that you don't want. Pass the unwanted keys as a list.
        This argument will not be used if "only" is present.
    
    Returns
    ----------
    copiedParams: tuple
        The params that the user asked for. They are not linked to the input params, so they can be modified independently.
    ''' 

    if only:
        return tuple( param for param in deepcopy(params) if param["key"] in only)
    else:
        return tuple( param for param in deepcopy(params) if param["key"] not in exclude)

def copyDict(dictInst, only = [], exclude = []):
    '''
    Function that returns a copy of a dict. This function is thought to be used for the settings dictionary, for example.

    Arguments
    ----------
    dictInst: dict
        The dictionary that needs to be copied.
    only: array-like
        Use this if you only want a certain set of values. Pass the wanted keys as a list.
    exclude: array-like
        Use this if there are some values that you don't want. Pass the unwanted keys as a list.
        This argument will not be used if "only" is present.
    
    Returns
    ----------
    copiedDict: dict
        The dictionary that the user asked for. It is not linked to the input dict, so it can be modified independently.
    ''' 

    if only:
        return {k: v for k,v  in deepcopy(dictInst).iteritems() if k in only}
    else:
        return {k: v for k,v  in deepcopy(dictInst).iteritems() if k not in exclude}

def load(path):
    '''
    Loads a previously saved python object using pickle. To be used for plots, sessions, etc...

    Arguments
    ----------
    path: str
        The path to the saved object.
    
    Returns
    ----------
    loadedObj: object
        The object that was saved.
    '''

    with open(path, 'rb') as handle:
        loadedObj = pickle.load(handle)
    
    return loadedObj

#-------------------------------------
#            Filesystem
#-------------------------------------

def findFiles(rootDir, searchString, depth = [0,0], sort = True, sortFn = None):
    '''
    Function that finds files (or directories) according to some conditions.

    Arguments
    -----------
    rootDir: str
        Path of the directory from which the search will start.
    searchString: str
        This is the string that will be passed to glob.glob() to find files or directories. 
        It works mostly like bash, so you can use wildcards, for example.
    depth: array-like of length 2
        It will specify the limits of the search. 
        For example, depth = [1,3] will make the function search for the searchString from 1 to 3 directories deep from rootDir.
        (0 depth means to look for files in the rootDir)
    sort: optional, boolean
        Whether the returned list of paths should be sorted.
    sortFn: optional, function
        The function that has to be used for sorting the paths. Only meaningful if sort is True.

    Returns
    -----------
    paths: list
        A list with all the paths found for the given conditions and sorted according to the provided arguments.
    '''

    files = []
    for depth in range(depth[0],depth[1] + 1):
        newFiles = glob.glob(os.path.join(rootDir,"*/"*depth, searchString))
        if newFiles:
            files += [os.path.abspath(path) for path in newFiles]

    if sort:
        return sorted(files, key = sortFn)
    else:       
        return files

#-------------------------------------
#         Multiprocessing
#-------------------------------------

def _applyMethod(argsTuple):
    '''
    Apply a method to an object. This function is meant for multiprocessing.
    '''

    method, obj, args, kwargs = argsTuple

    if args == None:
        args = []

    method(obj, *args, **kwargs)

    return obj._getPickleable()

def _initSinglePlot(argsTuple):
    '''
    Initialize a single plot. This function is meant to be used in multiprocessing, when multiple plots need to be initialized
    '''

    PlotClass, args, kwargs = argsTuple

    return PlotClass(**kwargs)._getPickleable()

def runMultiple(func, *args, argsList = None, kwargsList = None, messageFn = None):
    '''

    Makes use of the pathos.multiprocessing module to run a function simultanously multiple times.
    This is meant mainly to update multiple plots at the same time, which can accelerate significantly the process of visualizing data.

    All arguments passed to the function, except func, can be passed as specified in the arguments section of this documentation
    or as a list containing multiple instances of them.
    If a list is passed, each time the function needs to be run it will take the next item of the list.
    If a single item is passed instead, this item will be repeated for each function run.
    However, at least one argument must be a list, so that the number of times that the function has to be ran is defined.

    Arguments
    ----------
    func: function
        The function to be executed. It has to be prepared to recieve the arguments as they are provided to it (zipped).

        See the applyMethod() function as an example.
    *args:
        Contains all the arguments that are specific to the individual function that we want to run.
        See each function separately to understand what you need to pass (you may not need this parameter).
    argsList: array-like
        An array of arguments that have to be passed to the executed function.

        Can also be a list of arrays (see this function's description).

        WARNING: Currently it only works properly for a list of arrays. Didn't fix this because the lack of interest
        of argsList on Plot's methods (everything is passed as keyword arguments).

    kwargsList: dict
        A dictionary with the keyword arguments that have to be passed to the executed function.
        
        If the executed function is a Plot's method, these can be the settings, for example.

        Can also be a list of dicts (see this function's description).

    messageFn: function
        Function that recieves the number of tasks and nodes and needs to return a string to display as a description of the progress bar.

    Returns
    ----------
    results: list
        A list with all the returned values or objects from each function execution.
        This list is ordered, so results[0] is the result of executing the function with argsList[0] and kwargsList[0].  
    '''

    #Prepare the arguments to be passed to the initSinglePlot function
    toZip = [*args, argsList, kwargsList]
    for i, arg in enumerate(toZip):
        if not isinstance(arg, (list, tuple, np.ndarray)):
            toZip[i] = itertools.repeat(arg)
        else:
            nTasks = len(arg)

    #Create a pool with the appropiate number of processes
    pool = Pool( min(nTasks, os.cpu_count() - 1) )
    #Define the plots array to store all the plots that we initialize
    results = [None]*nTasks

    #Initialize the pool iterator and the progress bar that controls it
    progress = tqdm.tqdm(pool.imap(func, zip(*toZip) ), total = nTasks)

    #Set a description for the progress bar
    if not callable(messageFn):
        message = "Updating {} plots in {} processes".format(nTasks, pool.nodes)
    else:
        message = messageFn(nTasks, pool.nodes)

    progress.set_description(message)

    #Run the processes and store each result in the plots array
    for i, res in enumerate(progress):
        results[i] = res
    
    pool.close()
    pool.join()
    pool.clear()
    
    return results

def initMultiplePlots(PlotClass, argsList = None, kwargsList = None):
    '''
    Initializes a set of plots in multiple processes simultanously making use of runMultiple()

    All arguments passed to the function, can be passed as specified in the arguments section of this documentation
    or as a list containing multiple instances of them.
    If a list is passed, each time the function needs to be run it will take the next item of the list.
    If a single item is passed instead, this item will be repeated for each function run.
    However, at least one argument must be a list, so that the number of times that the function has to be ran is defined.

    Arguments
    ----------
    PlotClass: child class of sisl.viz.Plot
        The plot class that must be initialized

        Can also be a list of classes (see this function's description).
    argsList: array-like
        An array of arguments that have to be passed to the executed function.

        Can also be a list of arrays (see this function's description).

        WARNING: Currently it only works properly for a list of arrays. Didn't fix this because the lack of interest
        of argsList on Plot's methods (everything is passed as keyword arguments).
    kwargsList: dict
        A dictionary with the keyword arguments that have to be passed to the executed function.
        
        If the executed function is a Plot's method, these can be the settings, for example.

        Can also be a list of dicts (see this function's description).

    Returns
    ----------
    plots: list
        A list with all the initialized plots.
        This list is ordered, so plots[0] is the plot initialized with argsList[0] and kwargsList[0].  
    '''

    return runMultiple(_initSinglePlot, PlotClass, argsList = argsList, kwargsList = kwargsList)

def applyMethodOnMultipleObjs(method, objs, argsList = None, kwargsList = None):
    
    '''
    Applies a given method to the objects provided on multiple processes simultanously making use of the runMultiple() function.

    This is useful in principle for any kind of object and any method, but has been tested only on plots.

    All arguments passed to the function, except method, can be passed as specified in the arguments section of this documentation
    or as a list containing multiple instances of them.
    If a list is passed, each time the function needs to be run it will take the next item of the list.
    If a single item is passed instead, this item will be repeated for each function run.
    However, at least one argument must be a list, so that the number of times that the function has to be ran is defined.

    Arguments
    ----------
    method: func
        The method to be executed.
    objs: object
        The object to which we need to apply the method (e.g. a plot)

        Can also be a list of objects (see this function's description).
    argsList: array-like
        An array of arguments that have to be passed to the executed function.

        Can also be a list of arrays (see this function's description).

        WARNING: Currently it only works properly for a list of arrays. Didn't fix this because the lack of interest
        of argsList on Plot's methods (everything is passed as keyword arguments).
    kwargsList: dict
        A dictionary with the keyword arguments that have to be passed to the executed function.
        
        If the executed function is a Plot's method, these can be the settings, for example.

        Can also be a list of dicts (see this function's description).

    Returns
    ----------
    plots: list
        A list with all the initialized plots.
        This list is ordered, so plots[0] is the plot initialized with argsList[0] and kwargsList[0].  
    '''

    return runMultiple(_applyMethod, method, objs, argsList = argsList, kwargsList = kwargsList)

def repeatIfChilds(method):
    '''
    Decorator that will force a method to be run on all the plot's childPlots in case there are any.
    '''
    
    def applyToAllPlots(obj, *args, **kwargs):
        
        if hasattr(obj, "childPlots"):

            kwargsList = kwargs.get("kwargsList", kwargs)
            
            obj.childPlots = applyMethodOnMultipleObjs(method, obj.childPlots, kwargsList = kwargsList)
                
            obj.updateSettings(onlyOnParent = True, updateFig = False, **kwargs).getFigure()
        
        else:
        
            return method(obj, *args, **kwargs)
    
    return applyToAllPlots    