""" This file implements a smooth interface between sisl and plotly express, to make visualization of sisl objects even easier

This goes hand by hand with the implementation of dataframe extraction in sisl
objects, which is not already implemented (https://github.com/zerothi/sisl/issues/220)
"""
from functools import wraps

import plotly.express as px
from sisl._dispatcher import AbstractDispatch

__all__ = ["sx"]


class WithSislManagement(AbstractDispatch):
    def __init__(self, px):
        self._obj = px

    def dispatch(self, method):
        """ Wraps the methods of the object to preprocess the inputs """
        @wraps(method)
        def with_sisl_support(*args, **kwargs):

            if args:
                # Try to generate the dataframe for this object.
                if hasattr(args[0], 'to_df'):
                    args[0] = args[0].to_df()
                else:
                    # Otherwise, we are just going to interpret it as if the user wants to get the attributes
                    # of the object. We will support deep attribute getting here using points as separators.
                    # (I don't know if this makes sense because there's probably hardly any attributes that are
                    # ready to be plotted, i.e. they are 1d arrays)
                    obj = args.pop(0)
                    for key, val in kwargs.items():
                        if isinstance(val, str):
                            attrs = val.split('.')
                            search_obj = obj

                            # We try to recursively get the attributes
                            for attr in attrs:
                                newval = getattr(obj, attr, None)
                                if newval is None:
                                    break
                                search_obj = newval

                            else:
                                # If we've gotten to the end of the loop, it is because we've found the attribute.
                                val = newval

                            # Replace the provided string by the actual value of the attribute
                            kwargs[key] = val

            return method(*args, **kwargs)

        return with_sisl_support

sx = WithSislManagement(px)
