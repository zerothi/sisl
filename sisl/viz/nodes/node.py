from __future__ import annotations

import inspect
from typing import Any, Optional
from collections import ChainMap
from collections.abc import Mapping

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from sisl.messages import SislError, info
from .context import lazy_context
from ..input_fields._input_field import InputField
from ..input_fields._typetofield import get_function_fields

class NodeError(SislError):
    def __init__(self, node, error):
        self._node = node
        self._error = error
    
    def __str__(self):
        return f"There was an error with node {self._node}. {self._error}"
    pass


class NodeCalcError(NodeError):
    
    def __init__(self, node, error, inputs):
        super().__init__(node, error)
        self._inputs = inputs
        
    def __str__(self):
        return (f"Couldn't generate an output for {self._node} with the current inputs."
            f" The error raised was: {self._error}. \nCurrent inputs: {self._inputs}"
        )

class NodeInputError(NodeError):
    
    def __init__(self, node, error, inputs):
        super().__init__(node, error)
        self._inputs = inputs
        
    def __str__(self):
        # Should make this more specific
        return (f"Some input is not right in {self._node} and could not be parsed"
            f" The error raised was: {self._error}."
        )

class NodeMeta(type):
    def __call__(self, *args, **kwargs):
        if self._lazy_computation:
            return super().__call__(*args, **kwargs)
        else:
            if self._from_function:
                return self._get(*args, **kwargs)
            else:
                return super().__call__(*args, **kwargs).get()

    def init(self, *args, **kwargs):
        """Initializes a node of this class.
        
        This is to be used if you want to initialize a lazy node, regardless of whether
        the lazy_computation switch is on or off.
        """
        return super().__call__(*args, **kwargs)

class Node(NDArrayOperatorsMixin, metaclass=NodeMeta):
    """Generic class for nodes.

    A node is a process that runs with some inputs and returns some outputs.
    Inputs can come from other nodes' outputs, and therefore outputs can be
    linked to another node's inputs.
  
    A node MUST be a pure function. That is, the output should only depend on
    the input. In that way, the output of the node only needs to be calculated
    when the inputs change.
    """
    # Object that will be the reference for output that has not been returned.
    _blank = object()
    # Whether debugging messages should be issued
    _debug: bool = False
    _debug_show_inputs: bool = False

    # Variables to store the input fields
    _class_input_fields: Mapping[str, InputField] = {}
    _input_fields: ChainMap[str, InputField] = ChainMap(_class_input_fields)
    # Dictionary to keep the default inputs
    _default_inputs: dict[str, Any]

    # Whether inputs should be parsed before being passed to the node
    _input_parsing: bool = True

    # Global flag that controls whether a node has to be lazily computed or not.
    # This variable MUST NOT be overwritten by subclasses, because context managers
    # use it to turn on and off lazy computation.
    _lazy_computation: bool = False

    # Whether this node class was created from a function (i.e. using the Node.from_func
    # decorator), as oposed to using class syntax.
    _from_function: bool = False

    _args_inputs_key: Optional[str] = None
    _kwargs_inputs_key: Optional[str] = None

    def __init__(self, *args, automatic_recalc=None, **kwargs):
        # Init the dictionary containing input fields that are specific to this instance of
        # the node. Input fields will by default belong to the node class, but a copy
        # will be generated when an input field needs some state.
        self._obj_input_fields = {}
        self._input_fields = self._input_fields.new_child(self._obj_input_fields)

        # Grab *args and convert them into kwargs, unless the node really has a specification
        # of a VAR_POSITIONAL argument. In that case, when we reach the VAR_POSITIONAL argument,
        # we will assign the rest of *args to it.
        if len(args) >= 0:
            for i, (key, arg) in enumerate(zip(self._class_input_fields, args)):
                if key == self._args_inputs_key:
                    kwargs[key] = args[i:]
                    break
                else:
                    kwargs[key] = arg

        if automatic_recalc is None:
            automatic_recalc = self.default_inputs.get("automatic_recalc", False)

        self._inputs = {
            "automatic_recalc": automatic_recalc,
        }
        if self._kwargs_inputs_key is not None:
            self._inputs[self._kwargs_inputs_key] = {}

        for k, v in kwargs.items():
            if k in self._input_fields:
                self._inputs[k] = v
            elif self._kwargs_inputs_key is not None:
                self._inputs[self._kwargs_inputs_key][k] = v

        self._input_nodes = {}
        self._output_links = []

        self._update_connections(self._inputs)

        self._prev_evaluated_inputs = self._inputs

        self._output = self._blank
        self._nupdates = 0

        self._outdated = True
    
    def __init_subclass__(cls):

        node_func = cls._get

        # Get the input fields that correspond to the type annotations
        cls._class_input_fields = get_function_fields(node_func)
        # The "self" argument, if there, should not get an input field.
        cls._class_input_fields.pop("self", None)

        cls._input_fields = ChainMap(cls._class_input_fields)

        # Get the signature of the function
        sig = inspect.signature(node_func)
        
        cls.__doc__ = node_func.__doc__

        # Use the function's signature for the __init__ function, so that the help message
        # is actually useful.
        replace_kwargs = {}
        if "self" not in sig.parameters:
            replace_kwargs['parameters'] = (
                inspect.Parameter("self", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
                *sig.parameters.values()
            )
        init_sig = sig.replace(**replace_kwargs)
        no_self_sig = init_sig.replace(parameters=tuple(init_sig.parameters.values())[1:])

        # Find out if there are arguments that are VAR_POSITIONAL (*args) or VAR_KEYWORD (**kwargs)
        # and register it so that they can be handled on init.
        cls._args_inputs_key = None
        cls._kwargs_inputs_key = None
        for key, parameter in no_self_sig.parameters.items():
            if parameter.kind == parameter.VAR_POSITIONAL:
                cls._args_inputs_key = key
            if parameter.kind == parameter.VAR_KEYWORD:
                cls._kwargs_inputs_key = key

        cls.__init__.__signature__ = init_sig
        cls.__signature__ = no_self_sig

        return super().__init_subclass__()
        
    @classmethod
    def from_func(cls, func, func_method="_get"):
        """Builds a node from a function.

        Parameters
        ----------
        func: function
            The function to be converted to a node
        func_method: str, optional
            The name of the method to which the function will be assigned
        """
        if isinstance(func, type) and issubclass(func, Node):
            func = func._get       
        return type(func.__name__, (cls, ), {func_method: staticmethod(func)})

    def is_output_outdated(self, evaluated_inputs):
        """Checks if the node needs to be ran"""
        # If there is no output, we clearly need to calculate it
        if self._output is self._blank:
            return True

        # If there are different input keys than there were before,
        # it is also obvious that the inputs are different.
        if set(self._prev_evaluated_inputs) != set(evaluated_inputs):
            return True

        def _is_equal(prev, curr):
            if prev is curr:
                return True

            if type(prev) != type(curr):
                return False
            try:
                if prev == curr:
                    return True
                return False 
            except:
                return False
        
        # Otherwise, check if the inputs remain the same.
        for key in self._prev_evaluated_inputs:
            # Get the previous and current values
            prev = self._prev_evaluated_inputs[key]
            curr = evaluated_inputs[key]

            if not _is_equal(prev, curr):
                return True

        return False

    def get(self, **temp_inputs):
        # Loop through all the inputs and evaluate them if they are nodes.
        evaluated_inputs = {}
        curr_inputs = {**self._inputs, **temp_inputs}
        for key, input_val in curr_inputs.items():
            if key in ('automatic_recalc', ):
                continue
            if isinstance(input_val, Node):
                input_val = input_val.get(**temp_inputs)
            
            # For the special args_inputs key (if any), we need to loop through all the items
            if key == self._args_inputs_key:
                input_val = tuple(
                    val.get(**temp_inputs) if isinstance(val, Node) else val
                    for val in input_val
                )
            if key == self._kwargs_inputs_key:
                input_val = {k: val.get(**temp_inputs) if isinstance(val, Node) else val
                    for k, val in input_val.items()
                }

            evaluated_inputs[key] = input_val

        if self._outdated or self.is_output_outdated(evaluated_inputs):
            try:
                parsed_inputs = self.parse_inputs(**evaluated_inputs)
                if self._debug:
                    if self._debug_show_inputs:
                        info(f"{self}: evaluated with inputs {evaluated_inputs}: {self._output}.") 
                    else:
                        info(f"{self}: evaluated because inputs changed.")
            except Exception as e:
                raise NodeInputError(self, e, evaluated_inputs)

            try:
                with lazy_context(False):
                    kwargs = parsed_inputs.copy()
                    args = ()
                    if self._args_inputs_key is not None:
                        args = kwargs.pop(self._args_inputs_key, ())
                    kwargs_inputs = {}
                    if self._kwargs_inputs_key is not None:
                        kwargs_inputs = kwargs.pop(self._kwargs_inputs_key, {})
                    self._output = self._get(*args, **kwargs, **kwargs_inputs)
                if self._debug:
                    if self._debug_show_inputs:
                        info(f"{self}: evaluated with inputs {parsed_inputs}: {self._output}.") 
                    else:
                        info(f"{self}: evaluated because inputs changed.")
            except Exception as e:
                raise NodeCalcError(self, e, parsed_inputs)
            
            self._nupdates += 1
            self._prev_evaluated_inputs = evaluated_inputs
            self._outdated = False
        else:
            if self._debug:
                info(f"{self}: no need to evaluate")    

        return self._output

    def _get(self, **inputs):
        raise NotImplementedError(f"{self.__class__.__name__} node does not implement _get")

    def get_tree(self):
        tree = {
            'node': self,
            'inputs': {},
        }

        for key, value in self._inputs.items():
            # For the special args_inputs key (if any), we need to loop through all the items
            if key == self._args_inputs_key:
                value = tuple(
                    val.get_tree() if isinstance(val, Node) else val
                    for val in value
                )
            if key == self._kwargs_inputs_key:
                input_val = {k: val.get_tree() if isinstance(val, Node) else val
                    for k, val in value.items()
                }

            if isinstance(value, Node):
                tree['inputs'][key] = value.get_tree()
            else:
                tree['inputs'][key] = value

        return tree

    @property
    def default_inputs(self):
        return {k: field.default for k, field in self._input_fields.items()}

    @property
    def inputs(self):
        return ChainMap(self._inputs, self.default_inputs)

    def get_input(self, key: str, parsed: bool = False):
        input_val = self.inputs[key]

        if parsed:
            input_val = self.parse_inputs(**{key: input_val})[key]
        
        return input_val
    
    def get_input_field(self, key):
        return self._input_fields[key]

    def parse_inputs(self, **inputs):
        """Parses the provided inputs to normalized representations.
        
        This function just calls the input field attached to each argument
        so that they do the job.
        """
        # Quick return if we are not supposed to parse inputs.
        if not self._input_parsing:
            return inputs

        for k, input_field in self._input_fields.items():
            if isinstance(input_field, InputField) and k in inputs:
                inputs[k] = input_field.parse(inputs[k])
    
        return inputs

    def update_inputs(self, *args, **inputs):
        # If no new inputs were provided, there's nothing to do
        if not inputs and len(args) == 0:
            return

        if len(args) > 0:
            if self._args_inputs_key is None:
                raise ValueError(f"{self}.update_inputs does not support positional arguments, please provide values as keyword arguments.")
            inputs[self._args_inputs_key] = args
        
        if self._kwargs_inputs_key is not None:
            inputs[self._kwargs_inputs_key] = self._inputs.get(self._kwargs_inputs_key, {}).copy()
            for k in inputs:
                if k not in self._input_fields:
                    inputs[self._kwargs_inputs_key][k] = inputs.pop(k)  

        # Otherwise, update the inputs
        self._inputs.update(inputs)

        # Now, update all connections between nodes
        self._update_connections(inputs)

        # Mark the node as outdated
        self._receive_outdated()

        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError(f"{self.__class__.__name__} does not allow the 'out' argument in ufuncs.")
        inputs = {f'input_{i}': input for i, input in enumerate(inputs)}
        return UfuncNode(ufunc=ufunc, method=method, input_kwargs=kwargs, **inputs)

    def __getitem__(self, key):
        return GetItemNode(data=self, key=key)

    def _update_connections(self, updated_inputs={}):
        for key, input in updated_inputs.items():
            # Get the old connected node (if any) and tell them
            # that we are no longer using their input
            old_connection = self._input_nodes.pop(key, None)
            if old_connection is not None:
                old_connection._receive_output_unlink(self)

            # If the new input is a node, create the connection
            if isinstance(input, Node):
                self._input_nodes[key] = input
                input._receive_output_link(self)

            # We need to handle the case of the args_inputs key.

    def _receive_output_link(self, node):
        for linked_node in self._output_links:
            if linked_node is node:
                break
        else:
            self._output_links.append(node)
    
    def _receive_output_unlink(self, node):
        for i, linked_node in enumerate(self._output_links):
            if linked_node is node:
                del self._output_links[i]
                break
    
    def _inform_outdated(self):
        """Informs nodes that are linked to our output that they are outdated.
        
        This is either because we are outdated, or because an input update has
        triggered an automatic recalculation.
        """
        for linked_node in self._output_links:
            linked_node._receive_outdated()
    
    def _receive_outdated(self):
        # Mark the node as outdated
        self._outdated = True
        # If automatic recalculation is turned on, recalculate output
        self._maybe_autoupdate()
        # Inform to the nodes that use our input that they are outdated
        # now.
        self._inform_outdated()

    def _maybe_autoupdate(self):
        """Makes this node recalculate its output if automatic recalculation is turned on"""
        if self._inputs['automatic_recalc']:
            self.get()

    def normalize(self, vmin=0, vmax=1):
        def _normalize(data, vmin, vmax, **kwargs):
            data_min, data_max = np.min(data), np.max(data)
            normalized = (data - data_min) / (data_max - data_min)
            return normalized * (vmax - vmin) + vmin
        return FuncNode(func=_normalize, data=self, vmin=vmin, vmax=vmax)


class FuncNode(Node):
    def _get(self, func, **kwargs):
        return func(**kwargs)

class GetItemNode(Node):
    def _get(self, data, key):
        return data[key]


class UfuncNode(Node):
    """Node that wraps a numpy ufunc."""
    def _get(self, ufunc, method, input_kwargs, **kwargs):
        # We need to 
        inputs = []
        i = 0
        while True:
            key = f"input_{i}"
            if key not in kwargs:
                break
            inputs.append(kwargs.pop(key))
            i += 1
        return getattr(ufunc, method)(*inputs, **input_kwargs)
