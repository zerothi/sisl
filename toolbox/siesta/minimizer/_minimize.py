from hashlib import sha256
from abc import abstractmethod
from pathlib import Path
from numbers import Real
import warnings
import logging

import numpy as np
from scipy.optimize import minimize, dual_annealing

from sisl._dispatcher import AbstractDispatch
from sisl._dispatcher import ClassDispatcher
from sisl.io import tableSile
from sisl.utils import PropertyDict


_log = logging.getLogger("sisl_toolbox.siesta.pseudo")


def _convert_optimize_result(minimizer, result):
    """ Convert optimize result to conform to the scaling procedure performed """
    # reverse optimized value
    # and also store the normalized values (to match the gradients etc)
    if minimizer.norm[0] in ("none", "identity"):
        # We don't need to do anything
        # We haven't scaled anything
        return result
    result.x_norm = result.x
    result.x = minimizer.reverse_normalize(result.x)
    if hasattr(result, "jac"):
        # transform the jacobian
        # The jacobian is dM / dx with dx possibly being scaled
        # So here we change multiply by  dx / dv
        result.jac_norm = result.jac.copy()
        result.jac /= minimizer.reverse_normalize(np.ones(len(minimizer)),
                                                  with_offset=False)
    return result


class BaseMinimize:

    # Basic minimizer basically used for figuring out whether
    # to use a local or global minimization strategy

    def __init__(self, variables=(), out="minimize.dat", norm='identity'):
        # ensure we have an ordered dict, for one reason or the other
        self.variables = []
        if variables is not None:
            for v in variables:
                self.add_variable(v)
        self.reset(out, norm)

    def reset(self, out=None, norm=None):
        """ Reset data table to be able to restart """
        # While this *could* be a named-tuple, we would not be able
        # to override the attribute, hence we use a property dict
        # same effect.
        self.data = PropertyDict(x=[], y=[], hash=[])

        # log
        log = ""

        if not out is None:
            log += f" out={str(out)}"
            self.out = Path(out)

        if not norm is None:
            log += f" norm={str(norm)}"
            if isinstance(norm, str):
                self.norm = (norm, 1.)
            elif isinstance(norm, Real):
                self.norm = ("l2", norm)
            else:
                self.norm = norm
        _log.info(f"{self.__class__.__name__} resetting{log}")

    def normalize(self, variables, with_offset=True):
        if isinstance(variables, str):
            # this means we grab the variable name from the attributes
            # of each variable
            out = np.empty(len(self.variables))
            for i, v in enumerate(self.variables):
                out[i] = v.normalize(v.attrs[variables], self.norm, with_offset=with_offset)
        else:
            out = np.empty_like(variables)
            for i, v in enumerate(variables):
                out[i] = self.variables[i].normalize(v, self.norm, with_offset=with_offset)
        return out

    def normalize_bounds(self):
        return [v.normalize(v.bounds, self.norm) for v in self.variables]

    def reverse_normalize(self, variables, with_offset=True):
        # ensures numpy array
        out = np.empty_like(variables)
        for i, v in enumerate(variables):
            out[i] = self.variables[i].reverse_normalize(v, self.norm, with_offset=with_offset)
        return out

    def __getitem__(self, key):
        return self.variables[key]

    @staticmethod
    def get_hash(data):
        return sha256(data.view(np.uint8)).hexdigest()

    def add_variable(self, variable):
        if self.variables.count(variable.name) != 0:
            raise ValueError(f"Multiple variables with same name {variable.name}")
        self.variables.append(variable)

    @property
    def names(self):
        return [v.name for v in self.variables]

    @property
    def values(self):
        return np.array([v.value for v in self.variables], np.float64)

    def update(self, variables):
        """ Update internal variables for the values """
        for var, v in zip(self.variables, variables):
            var.update(v)

    def dict_values(self):
        """ Get all vaules in a dictionary table """
        return {v.name: v.value for v in self.variables}

    # Define a dispatcher for converting Minimize data to some specific data
    #  BaseMinimize().to.skopt() will convert to an skopt.OptimizationResult structure
    to = ClassDispatcher("to",
                         obj_getattr=lambda obj, key:
                         (_ for _ in ()).throw(
                             AttributeError((f"{obj}.to does not implement '{key}' "
                                             f"dispatcher, are you using it incorrectly?"))
                         )
    )

    def __enter__(self):
        """ Open the file and fill with stuff """
        _log.debug(f"__enter__ {self.__class__.__name__}")

        # check if the file exists
        if self.out.exists():
            # read in previous data
            # this will be "[variables, runs]"
            data, header = tableSile(self.out).read_data(ret_header=True)
        else:
            data = np.array([])

        # check if the file exists
        if self.out.exists() and data.size > 0:
            nvars = data.shape[0] - 1
            if nvars != len(self):
                raise ValueError(f"Found old file {self.out} which contains previous data for another number of parameters, please delete or move file")

            # now parse header
            *header, _ = header[1:].split()
            idx = []
            for name in self.names:
                # find index in header
                for i, head in enumerate(header):
                    if head == name:
                        idx.append(i)
                        break

            if nvars != len(idx):
                print(header)
                print(self.names)
                print(idx)
                raise ValueError(f"Found old file {self.out} which contains previous data with some variables being renamed, please correct header or move file")

            # add functional value, no pivot
            idx.append(len(self))

            # re-arrange data (in case user swapped order of variables)
            data = np.ascontiguousarray(data[idx].T)
            x, y = data[:, :-1], data[:, -1]
            # We populate with hashes without the functional
            # That would mean we can't compare hashes between input arguments
            # only make the first index a list (x.tolist() makes everything a list)
            self.data.x = [xi for xi in x]
            self.data.y = [yi for yi in y]
            self.data.hash = list(map(self.get_hash, self.data.x))

        # Re-open file (overwriting it)

        # First output a few things in this file
        comment = f"Created by sisl '{self.__class__.__name__}'."
        header = self.names + ["metric"]
        if len(self.data.x) == 0:
            self._fh = tableSile(self.out, 'w').__enter__()
            self._fh.write_data(comment=comment, header=header)
        else:
            comment += f" The first {len(self.data)} lines contains prior content."
            data = np.column_stack((self.data.x, self.data.y))
            self._fh = tableSile(self.out, 'w').__enter__()
            self._fh.write_data(data.T, comment=comment, header=header, fmt='20.17e')
            self._fh.flush()

        return self

    def __exit__(self, *args, **kwargs):
        """ Exit routine """
        self._fh.__exit__(*args, **kwargs)
        # clean-up
        del self._fh

    def __len__(self):
        return len(self.variables)

    @abstractmethod
    def __call__(self, variables, *args):
        """ Actual running code that takes `variables` conforming to the order of initial setup.

        It will return the functional of the minimize method

        Parameters
        ----------
        variables : array-like
          variables to be minimized according to the metric `self.metric`
        """

    def _minimize_func(self, norm_variables, *args):
        """ Minimization function passed to the minimization method

        This is a wrapper which does 3 things:

        1. Convert input values from normalized to regular values
        2. Update internal variables with the value currently being
           runned.
        3. Check if the values have already been calculated, if so
           return the metric directly from the stored table.
        4. Else, calculate the metric using the ``self.__call__``
        5. Append values to the data and hash it.

        Parameters
        ----------
        norm_variables : array_like
           normed variables to be minimized
        *args :
           arguments passed directly to the ``self.__call__`` method
        """
        _log.debug(f"{self.__class__.__name__}._minimize_func")
        # Update internal set of variables
        variables = self.reverse_normalize(norm_variables)
        self.update(variables)

        # First get the hash of the current variables
        current_hash = self.get_hash(variables)

        try:
            idx = self.data.hash.index(current_hash)
            # immediately return functional value that is hashed
            _log.info(f"{self.__class__.__name__}._minimize_func, using prior hashed calculation {idx}")

            return self.data.y[idx]
        except ValueError:
            # in case the hash is not found
            pass

        # Else we have to call minimize
        metric = np.array(self(variables, *args))
        # add the data to the output file and hash it
        self._fh.write_data(variables.reshape(-1, 1), metric.reshape(-1, 1), fmt='20.17e')
        self._fh.flush()
        self.data.x.append(variables)
        self.data.y.append(metric)
        self.data.hash.append(current_hash)

        return metric

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Run the minimize model """


class LocalMinimize(BaseMinimize):

    def run(self, *args, **kwargs):
        # Run minimization (always with normalized values)
        norm_v0 = self.normalize(self.values)
        bounds = self.normalize_bounds()
        with self:
            opt = minimize(self._minimize_func,
                           x0=norm_v0, args=args, bounds=bounds,
                           **kwargs)

        return _convert_optimize_result(self, opt)


class DualAnnealingMinimize(BaseMinimize):

    def run(self, *args, **kwargs):
        # Run minimization (always with normalized values)
        norm_v0 = self.normalize(self.values)
        bounds = self.normalize_bounds()
        with self:
            opt = dual_annealing(self._minimize_func,
                                 x0=norm_v0, args=args, bounds=bounds,
                                 **kwargs)
        return _convert_optimize_result(self, opt)


class MinimizeToDispatcher(AbstractDispatch):
    """ Base dispatcher from class passing from Minimize class """
    @staticmethod
    def _ensure_object(obj):
        if isinstance(obj, type):
            raise ValueError(f"Dispatcher on {obj} must not be called on the class.")


class MinimizeToskoptDispatcher(MinimizeToDispatcher):
    def dispatch(self, *args, **kwargs):
        import skopt
        minim = self._obj
        self._ensure_object(minim)
        if len(args) > 0:
            raise ValueError(f"{minim.__class__.__name__}.to.skopt only accepts keyword arguments")

        # First create the Space variable
        def skoptReal(v):
            low, high = v.bounds
            return skopt.space.Real(low, high, transform="identity", name=v.name)
        space = skopt.Space(list(map(skoptReal, self.variables)))

        # Extract sampled data-points
        if "x" in kwargs:
            Xi = kwargs.pop("x")
            yi = kwargs.pop("y")
        else:
            Xi = np.array(self.data.x)
            yi = np.array(self.data.y)

        if "models" not in kwargs:
            import sklearn
            # We can't use categorial (SVC) since these are regression models
            # fast, but should not be as accurate?
            #model = sklearn.svm.LinearSVR()
            # much slower, but more versatile
            # I don't know which one is better ;)
            model = sklearn.svm.SVR(cache_size=500)
            #model = sklearn.svm.NuSVR(kernel="poly", cache_size=500)
            # we need to fit to create auxiliary data
            warnings.warn(f"Converting to skopt without a 'models' argument forces "
                          f"{minim.__class__.__name__} to train a model for the sampled data. "
                          f"This may be slow depending on the number of samples...")
            model.fit(Xi, yi)
            kwargs["models"] = [model]

        result = skopt.utils.create_result(Xi, yi, space=space, **kwargs)
        return result

BaseMinimize.to.register("skopt", MinimizeToskoptDispatcher)
