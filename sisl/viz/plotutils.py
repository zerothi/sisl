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

    PlotClass, *args = argsTuple

    kwargs = { key: args[i+1] for i, key in enumerate(args) if i%2 == 0 }

    plot = PlotClass(**kwargs)

    return plot._getDictToSave()

def load(path):

    with open(path, 'rb') as handle:
        saved = pickle.load(handle)
    
    PlotClass = list(filter(lambda cls: cls.__name__ == saved['additionalInfo']['className'], Plot.__subclasses__()))[0]
    
    plt = PlotClass()
    
    for key, val in saved.items():
        if key != "additionalInfo":
            setattr(plt, key, val)
    
    return plt