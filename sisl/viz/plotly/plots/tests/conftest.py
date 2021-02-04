import importlib
import pytest
import inspect
from pathlib import Path

import sisl


class MultipleTesterCreator(type):
    """
    Metaclass to setup multiple tests seamlessly.

    Otherwise, we would need to use decorators.

    See Also
    -----------
    `PlotTester`: the class that uses this metaclass and serves
    as a base to create tests that can be very easily and consistently
    iterated over.
    """

    def __new__(cls, name, bases, attrs):

        # Try to get the run_for argument, which indicates the parameters
        # that the user wants to run the test for
        params = attrs.pop("run_for", None)

        # Generate the new class
        cls = type.__new__(cls, name, bases, attrs)

        # If there were no parameters, just return the class
        if params is None:
            return cls

        # Otherwise, we need to create the right environment for the tests to run
        # That is, we need to define the pytest fixtures that will be parametrized
        # and set them in the scope of the class (so that the class detects them)

        # For this, we first try to find the file where the class is being defined
        frame = inspect.currentframe()
        while True:
            frame = frame.f_back

            if Path(frame.f_code.co_filename).stem.startswith("test_"):
                break

        scope = frame.f_globals

        # Once we have it, we proceed to apply the fixtures and returned the class
        # properly setup for tests :)
        return setup_multiple_tests(cls, params, scope)


@pytest.mark.usefixtures("only_if_plot_initialized")
class PlotTester(metaclass=MultipleTesterCreator):
    """
    Helps defining tests that will be run with multiple parameters.

    To use it, one only needs to define a child class with the attribute 
    `_required_attrs` that will indicate what the test needs as an input to run.
    These attributes will be available at first level during the test.

    Then, you also need to define the `run_for` attribute so that it knows which parameters
    to use each time. `run_for` is a dictionary because in that way we can use the keys as
    the ID for each test so that we can more easily understand what is failing.

    One "hidden" required attribute is `init_func`, which is the function to initialize the plot
    that will be tested. The first thing any class that inherits from PlotTester will do is to
    try to run that function and store the result in self.plot. If the plot can not be initialized
    for some reason, PlotTester won't bother to run the rest of the tests so that you can more
    easily spot the problem. 

    One can also provide the "plot_file" attribute instead of "init_func" if all they want to do
    is to get the file from sisl files and and plot it.

    Examples
    ----------

    from sisl.viz.plotly.plots.tests.helpers import PlotTester
    from functools import partial

    class TestMyPlot(PlotTester):

        # Indicate that you expect "expected_result" to be provided
        # for each run of the test
        _required_attrs = ["expected_result"]

        # Define each run
        run_for = {
            "first_test": {
                "init_func": H.plot,
                "expected_result": 4
            },

            "with_precision": {
                "init_func": partial(H.plot, precision="double"),
                "expected_result": 4.35,
            }
        }

        # Apply your tests to the initialized plot under self.plot
        def test_result(self):

            assert self.plot.result = self.expected_result

    """

    _attrs = {}

    def __setattr__(self, key, val):
        self.__class__._attrs[key] = val

    def __getattr__(self, key):
        if key in self._attrs:
            return self.__class__._attrs[key]

        raise AttributeError(key)

    def test_plot_initialization(self, sisl_files):
        """
        We are going to try to initialize the plot.

        If plot_file is provided, we will retrieve it from sisl_files and plot
        the default plot. Otherwise, an init_func must be provided. The init_func
        receives sisl_files, just in case you want to plot a file with a plot method
        other than the default.
        """
        filename = getattr(self, "plot_file", None)

        if filename is None:
            plot = self.init_func(sisl_files=sisl_files, _debug=True)
        else:
            sile = sisl.get_sile(sisl_files(filename))
            plot = sile.plot()

        # We have to set the plot as a class attribute because for some reason pytest doesn't like to
        # set an instance attribute
        self.plot = plot


@pytest.fixture
def only_if_plot_initialized(request):
    """
    Skips all tests of a class if the plot failed to be initialized.
    """

    if request.function.__name__ != "test_plot_initialization":
        # loop modules that viz.plotly depends on
        importables = []
        non_importables = []
        for modname in ("plotly", "skimage", "pandas", "xarray"):
            try:
                importlib.import_module(modname)
                importables.append(modname)
            except ImportError:
                non_importables.append(modname)

        if importables:
            msg = ", ".join(importables) + " is/are importable"
            if non_importables:
                msg += "; " + ", ".join(non_importables) + " is/are not importable"
        elif non_importables:
            msg = ", ".join(non_importables) + " is/are not importable"

        if "plot" not in request.cls._attrs:
            pytest.skip(f"plot was not initialized ({msg})")


def setup_multiple_tests(cls, params, global_scope):
    """
    Sets up a test class to be ran multiple times with different parameters

    Parameters
    ------------
    cls: 
        the class that we want to set up
    params: dict
        a dictionary where the keys are the names of each test run and
        the values are the parameters to use. See the examples section of `PlotTester`
        for an example of it.
    global_scope: globals()
        the global scope of the class that we are setting up. This is required
        because we need to introduce the pytest fixtures in this scope, otherwise
        the class will have no idea that they exist and it will result in an error.
    """

    # Define the fixture name, which needs to be dynamic just in case
    # there is more than one class to set up in the same file.
    fixture_name = f"setup_{cls.__name__}"

    # Create the pytest fixture that we are going to use to run the same test
    # multiple times
    @pytest.fixture(scope="class", params=params.items(), ids=lambda item: item[0])
    def setup_plot_tester(request):

        key, plot_attributes = request.param

        # Set up the tester attributes
        for attr in [*getattr(request.cls, "_required_attrs", [])]:
            if attr not in plot_attributes:
                pytest.fail(f"You are missing the '{attr}' required attribute")
        request.cls._attrs = plot_attributes

        yield # run the test

        # Clean the tester attributes (this is after the test has run, because
        # we are using yield)
        request.cls.attrs = {}

    # Introduce the fixtures that will be used to the scope of the class
    # so that they are available to it
    global_scope[fixture_name] = setup_plot_tester
    global_scope["only_if_plot_initialized"] = only_if_plot_initialized

    # Finally, return the class after indicating the fixtures that we are going to use.
    # Note that "only_if_plot_initialized" is already indicated in PlotTester, that's why
    # we don't need to indicate it here.
    return pytest.mark.usefixtures(fixture_name)(cls)
