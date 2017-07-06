"""
Sub-package to easily make algorithmic decisions based on different routines

Several functions exists here to most efficiently choose the most performant 
routine.

The `Selector` will run through the different routines and decide, based on 
all the calls which is the best one.

Basically the `Selector` will only be a powerful tool if a given routine is
called numerous times.

The following example will show how the `TimeSelector` may be used
to automatically call the fastest of 3 routines::

  >>> def func1():
  ...    print('Func - 1')
  >>> def func2():
  ...    print('Func - 2')
  ...    time.sleep(1)
  >>> def func3():
  ...    print('Func - 3')
  ...    time.sleep(1)
  >>> selector = TimeSelector([func1, func2, func3])
  >>> selector()
  Func - 1
  >>> selector()
  Func - 2
  >>> selector()
  Func - 3
  >>> selector() # will now only call the fastest of the 3
  Func - 1

In certain cases one may wish to limit the search for a selected routine
by only searching until the performance of the *next* called routine drops.
This is called an *ordered* selector because it tries them in order, and 
once one is slower than the former tested ones, it will not test any further.
For the above same functions we may do::

  >>> selector = TimeSelector([func1, func2, func3], ordered=True)
  >>> selector()
  Func - 1
  >>> selector()
  Func - 2
  >>> selector()
  Func - 1
  >>> selector()
  Func - 1
"""
from __future__ import print_function, division

import warnings
import time
import numpy as np

from ._help import _str as str, _range as range, _zip as zip

__all__ = ['Selector', 'TimeSelector']


class Selector(object):
    """ Base class for implementing a selector of class routines

    This class should contain a list of routines and may then be used
    to always return the best performant routine.

    This is done on a per-class basis where this class should initially 
    determine which routine is the best performing one and then always return
    that one.

    Attributes
    ----------
    routines : list of func
       this is a list of functions that will be selected from.
    ordered : bool
      If False a simple selection of the most performant one will be chosen.
      If True, it will check the routines in order and once *one* of the
      routines is less performant it will choose from the setof runned
      routines.
    """

    __slots__ = ['_routines', '_performances', '_best', '_ordered']

    def __init__(self, routines=None, ordered=False):

        # Copy the routines to the list
        if routines is None:
            self._routines = []
        else:
            self._routines = routines

        self._ordered = ordered

        # Create a list of performance identifiers
        self._performances = [None] * len(self.routines)

        # Initialize no best routine
        self._best = None

    @property
    def routines(self):
        return self._routines

    @property
    def performances(self):
        return self._performances

    @property
    def best(self):
        if self._best is None:
            return None
        return self._best

    @property
    def ordered(self):
        return self._ordered

    def prepend(self, routine):
        """ Prepends a new routine to the selector

        Parameters
        ----------
        routine : func
           the new routine to be tested in the selector
        """
        self._routines.insert(0, routine)
        self._performances.insert(0, None)
        # Ensure that the best routine has not been chosen
        self._best = None

    def append(self, routine):
        """ Prepends a new routine to the selector

        Parameters
        ----------
        routine : func
           the new routine to be tested in the selector
        """
        self._routines.append(routine)
        self._performances.append(None)
        # Ensure that the best routine has not been chosen
        if self.ordered:
            if not self.performances[-1] is None:
                # Reset the best chosen routine.
                # This is because if the last routine has been checked it means
                # that we may possibly use the new routine.
                # If the last routine have not been checked, then
                # definitely the appended routine would not be chosen
                self._best = None
        else:
            self._best = None

    def reset(self):
        """ Reset the performance table to redo the performance checks """
        self._performances = [None] * len(self._performances)
        self._best = None

    def select_best(self, routine=None):
        """ Update the `best` routine, if applicable

        Update the selector to choose the best method.
        If not all routines have been carried through, then
        no best routine will be selected (unless `self.ordered` is True).

        By passing a routine as an argument that given routine
        will by default be the chosen best algorithm.

        Parameters
        ----------
        routine : func or str
           If `None` is passed (the default) it will select the best 
           default routine based on the stored performances.
           If, however, not all performance values has been created
           no routine will be selected.

           If passing a `func` that function will be chosen as the
           best method
        """
        if routine is None:

            # Try and select the routine based on the internal runned
            # performance specifiers
            selected, perf = -1, 0.
            for i, v in enumerate(self.performances):
                if v is None:
                    # Quick return if we are not done
                    return

                if v > perf:
                    perf = v
                    selected = i

                elif self.ordered and selected >= 0:
                    # We have an ordered selector
                    # I.e. if the performance is decreasing,
                    # we simply choose the last one.
                    break

            self._best = self.routines[selected]
            return

        # Default to None
        self._best = None

        if isinstance(routine, str):
            # Select the best routine, based on the name
            for r in self.routines:
                if r.__name__ == routine:
                    self._best = r
                    break
            if self.best is None:
                warnings.warn((self.__class__.__name__ + ' selection of '
                               'optimal routine is not in the list of available '
                               'routines. Will not select a routine.'), UserWarning)
        else:
            self._best = routine

    def next(self):
        """ Choose the next routine that requires performance analysis

        Returns
        -------
        int, func :
           a tuple with the `int` specifying the routine index. `func` is the
           routine that is to be runned.
        """
        for i, v in enumerate(self.performances):
            if v is None:
                # Quick return the routine to test next time
                return i, self.routines[i]

        return -1, self._best

    def __call__(self, *args, **kwargs):
        """ Call the function that optimizes the run-time the most 

        The first argument *must* be an object (`self`) while all remaining
        arguments are transferred to the routine calls
        """

        best = self.best
        if not best is None:
            return best(*args, **kwargs)

        # Figure out if we have the performance for all the routines
        idx, routine = self.next()
        if idx < 0:
            return routine(*args, **kwargs)

        # Start the performance profile
        start = self.start()

        # Run the routine
        returns = routine(*args, **kwargs)

        # Update the internal data in the performance list
        self._performances[idx] = self.stop(start)

        # Update the best method, if possible
        self.select_best()

        return returns

    def start(self):
        """ Start the performance profiler

        This routine should return an initial state value.
        The difference between `stop() - start()` should yield a 
        performance identifier which may be used to control the 
        used algorithm.

        A large performance identifier results in the use of the routine.
        """
        raise NotImplementedError

    def stop(self, start):
        """ Stop the performance profiler

        This routine should return an initial state value.
        The difference between `stop() - start()` should yield a 
        performance identifier which may be used to control the 
        used algorithm.

        A large performance identifier results in the use of the routine.

        Parameters
        ----------
        start : float
            the output of the `start()` routine to convert to actual performance
            identifier
        """
        raise NotImplementedError


class TimeSelector(Selector):
    """ Routine performance selector based on timings for the routines """

    def start(self):
        """ Start the timing routine """
        return time.time()

    def stop(self, start):
        """ Stop the timing routine """
        return 1. / (time.time() - start)
