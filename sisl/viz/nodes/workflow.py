from abc import abstractmethod
from collections import defaultdict, ChainMap
from dataclasses import dataclass
import inspect

from .node import Node
from .context import lazy_context

@dataclass
class TestWorkflowInput:
    input_key: str

class Workflow(Node):
    # Workflows will by default be lazily computed.
    _lazy_computation: bool = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._generate_node_tree(**{k: v for k, v in self._inputs.items() if k not in ("automatic_recalc", )})

    @staticmethod
    def _prepare_class(cls):
        pass

    @abstractmethod
    def _generate_node_tree(self, **inputs):
        pass
   
    @classmethod
    def from_func(cls, func):
        """Builds a workflow from a function.

        Parameters
        ----------
        func: function
            The function to be converted to a node
        """
        return super().from_func(func, func_method="_workflow")

    def __init_subclass__(cls):
        # If this is just a subclass of Workflow that is not meant to be ran, continue 
        if not hasattr(cls, "_workflow"):
            cls._prepare_class(cls)
            return super().__init_subclass__()

        # Otherwise, do all the setting up of the class
        work_func = cls._workflow

        # Get the signature of the function
        sig = inspect.signature(work_func)
        
        # Trial run of the workflow to see how inputs flow.
        inps = {k: TestWorkflowInput(input_key=k) for k in sig.parameters}
        with lazy_context(True):   
            final_node = work_func(**inps)
        tree = final_node.get_tree()
        
        def determine_nodes(tree):
            # Build the nodes and linked inputs dictionaries
            linked_inputs = defaultdict(dict)
            nodes = {}

            def _add_node(tree):
                """Add nodes to the dictionary, leafs first, then root."""
                if "node" in tree:
                    node = tree['node']

                    # If the node is already registered, avoid registering it again.
                    for reg_node in nodes.values():
                        if node is reg_node:
                            return

                    # Loop through its inputs to find nodes in it
                    for inp_key, inp in tree['inputs'].items():
                        if isinstance(inp, dict):
                            _add_node(inp)
                    # If there are arg inputs, then also do the same for them
                    for inp in tree['inputs'].get("arg_inputs", []):
                        if isinstance(inp, dict):
                            _add_node(inp)

                    # Determine the name we are going to assign to this node inside the workflow.
                    node_name = node.__class__.__name__
                    name = node_name
                    i = 0
                    while name in nodes:
                        i += 1
                        name = f"{node_name}_{i}"

                    # Add it to the dictionary of nodes
                    nodes[name] = node

                    # Now that we know this node's name, we can link its inputs to the inputs of the
                    # workflow.
                    for inp_key, inp in tree['inputs'].items():
                        if isinstance(inp, TestWorkflowInput):
                            # Save the link
                            linked_inputs[name][inp_key] = inp.input_key
            
            _add_node(tree)
            
            return linked_inputs, nodes
        
        # Define methods of the new class
        def _prepare_class(cls):
            linked_inputs, nodes = determine_nodes(tree)
            cls._workflow_nodes = {k: v.__class__ for k, v in nodes.items()}
            cls._linked_inputs = linked_inputs
            
        def _generate_node_tree(self, **inputs):
            with lazy_context(True):
                self._final_node = work_func(**inputs)
            self._nodes = determine_nodes(self._final_node.get_tree())[1]
            return self._final_node

        # Set the functions required for the workflow to run properly
        cls._prepare_class = _prepare_class
        cls._generate_node_tree = _generate_node_tree
        cls._get = work_func

        cls._prepare_class(cls)
        return super().__init_subclass__()

    def __getattr__(self, key):
        if key != "_final_node":
            return getattr(self._final_node, key)
        raise AttributeError(key)

    def get(self):
        return self._final_node.get()
    
    def update_inputs(self, **inputs):
        # Be careful here: 
        
        # If we update from leafs to root: A node up on the tree might trigger a recalculation
        # even if this node still needs to get its inputs updated, which means useless computation
        
        # If we update from root to leafs: An update might trigger a recalculation with outdated data
        # from the leafs, who still need to update.
        
        # Is the solution to just create a new tree from the leaf?
        # Or avoid automatic recalculation altogether?
        
        for node_key, node in self._nodes.items():
            linked_inputs = self._linked_inputs[node_key]
            node_updates = {}
            
            for key, input_key in linked_inputs.items():
                if input_key in inputs:
                    node_updates[key] = inputs[input_key]
            
            if node_updates:
                self._nodes[node_key].update_inputs(**node_updates)
        
        self._inputs.update(inputs)
                
        return self