from sisl.viz import Plot, Configurable

#Plot to be checked
from sisl.viz import PDOSPlot as PlotToTest

def errorMessage(message, helpMessage=None, links=None):

    errorString = "\nError:\n\t{}".format(message.replace("\n", "\n\t"))

    helpString = "\nWhat to do:\n\t{}".format(helpMessage.replace("\n", "\n\t")) if helpMessage else ""

    if isinstance(links, str):
        links = [links]

    linkString = "\nSee links for more info:\n\t{}".format("\n\t".join(links)) if links else ""
    
    return "{0}{1}{2}{3}{0}\n".format("\n"+"-"*40, errorString, helpString, linkString)

def checkType(varname, variable, shouldType = object, customErrMessage = False):

    errMessage = customErrMessage(varname, shouldType.__name__,type(variable).__name__ ) if customErrMessage else errorMessage("'{}' should be a {}, it is now a {}".format(varname, shouldType.__name__, type(variable).__name__))

    assert isinstance(variable, shouldType), errMessage

def classVariableCheckType(*args, **kwargs):

    checkType(*args,
        customErrMessage=lambda varname, correctType, currentType: errorMessage("The class variable '{}' should be a {}, it is now a {}".format(varname, correctType, currentType)),
        **kwargs,
    )


def test_verybasics():
    assert PlotToTest, errorMessage("Nothing was provided to be tested")
    assert isinstance(PlotToTest, type), errorMessage("You didn't provide a class, this is a plot class tester",
        "Please provide the class that you have defined for your analysis.\n A class should be defined as 'class MyPlot: ...'",
        "https://www.w3schools.com/python/python_classes.asp")

#------------------------------------
#       Test class variables
#------------------------------------
classDict = vars(PlotToTest)
def test_parameters():

    assert "_parameters" in classDict, errorMessage("This plot class does not have a _parameters variable.\nAre you sure you don't want to allow the user to tune your plot?",
        "Please define a class variable called _parameters that contains all the parameters that the user can tweak about your plot.")
    _parameters = PlotToTest._parameters
    classVariableCheckType("_parameters", _parameters, tuple)
    wrongTypes = [type(param).__name__ for param in _parameters if not isinstance(param, dict) ]
    assert len(wrongTypes) == 0, errorMessage("All parameters inside the _parameters variable should be dictionaries.\nWrong types found: {}".format(",".join(wrongTypes) ))

    def checkDict(dictToCheck, keysAndTypes):

        for key, correctType in keysAndTypes.items():
            assert key in dictToCheck, errorMessage("The parameter {} does not have a '{}' attribute ".format(param, key),
                "Please set a '{}' attribute for this parameter. {}".format(key, "It should be of type: {}".format(correctType.__name__) if correctType else "")
            )
            if correctType:
                checkType(key, dictToCheck[key], correctType)

    for param in _parameters:

        checkDict(param, {"key": str, "name": str, "default": None})

    
    keys = [param["key"] for param in _parameters]
    duplicateKeys = set([key for key in keys if keys.count(key) > 1])
    assert len(duplicateKeys) == 0, errorMessage("The following keys were found duplicate in '_parameters' ({})".format(",".join(duplicateKeys)),
        "Please provide a different 'key' attribute for each parameter")

def test_plotType():
    assert "_plotType" in classDict, errorMessage("This plot class does not have a _plotType variable defining the name of the plot")
    _plotType = PlotToTest._plotType
    classVariableCheckType("_plotType", _plotType, str)
    existantPlotTypes = [PlotClass._plotType for PlotClass in Plot.__subclasses__() if PlotClass != PlotToTest]
    assert _plotType not in existantPlotTypes, errorMessage("Plot type '{}' already exists".format(_plotType),
        "Please give your class a _plotType attribute that is not in this list: {}".format(",".join(existantPlotTypes))
    )

#------------------------------------
#       Test class inheritance
#------------------------------------
def test_init():
    assert isinstance(PlotToTest(), PlotToTest), errorMessage("This plot class can't be initialized without parameters",
        "Please check that default settings values are enough to initialize a plot")

def test_parents():

    assert Plot in type.mro(PlotToTest), errorMessage("This plot class does not inherit from Plot", 
        "Define your class as 'class {}(Plot):' ".format(PlotToTest.__name__),
        "https://www.w3schools.com/python/python_inheritance.asp")
    assert Configurable in type.mro(PlotToTest), errorMessage("This plot class does not inherit from Configurable",
        "Contact sisl's developers because something is messed up there probably. Most certainly not your fault :)")
    
    classHasInit = callable( getattr(PlotToTest, "__init__", None) )
    assert isinstance( getattr(PlotToTest(),"settings", None), dict), errorMessage("Plot settings are not getting initialized correctly",
     "Please make sure that you are calling super().__init__(**kwargs) in your __init__() method" if classHasInit else "Contact sisl's developers :)")

#------------------------------------
#            Test methods
#------------------------------------


