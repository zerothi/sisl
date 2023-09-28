from __future__ import annotations

import ast
import html
import inspect
import textwrap
from _ast import Dict
from collections import ChainMap
from types import FunctionType
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Type, Union

from sisl._environ import get_environ_variable, register_environ_variable
from sisl.messages import warn

from .context import temporal_context
from .node import DummyInputValue, Node
from .syntax_nodes import DictSyntaxNode, ListSyntaxNode, TupleSyntaxNode
from .utils import traverse_tree_backward, traverse_tree_forward

register_environ_variable(
    "SISL_NODES_EXPORT_VIS", default=False, 
    description="Whether the visualizations of the networks in notebooks are meant to be exported.", 
)

class WorkflowInput(DummyInputValue):
    pass

class WorkflowOutput(Node):
    
    @staticmethod
    def function(value: Any) -> Any:
        return value

class NetworkDescriptor:

    def __get__(self, instance, owner):
        return Network(owner)

class Network:

    _workflow: Type[Workflow]

    def __init__(self, workflow: Type[Workflow]):
        self._workflow = workflow

    @staticmethod
    def _get_edges(
        workflow: Type[Workflow], include_workflow_inputs: bool = False, edge_labels: bool = True
    ) -> List[Tuple[str, str, dict]]:
        """Get the edges that connect nodes in the workflow.
        
        Parameters
        ----------
        workflow: Type[Workflow]
            The workflow class to retreive the edges from.
        include_workflow: bool, optional
            Whether to include edges connecting the workflow itself to other nodes. If included, the edges
            with other nodes are the linked inputs (inputs redirected from workflow to nodes).
        edge_labels: bool, optional
            Whether to add labels to edges. If True, the labels are the key of the input that
            receives the output of the outgoing node.
        Returns
        ----------
        List[Tuple[str, str, dict]]:
            The list of edges. Each edge is a tuple with three items:
                - Outgoing node.
                - Incoming node.
                - Edge metadata
        """
        edges = []

        # Build the function that given two connected nodes will return the 
        # edge metadata or "props".
        def _edge_props(node_out, node_in, key) -> dict:
            props = {}
            if edge_labels:
                props['label'] = key
            
            props['title'] = f"{node_out.__class__.__name__}() -> {node_in.__class__.__name__}.{key}"
            return props

        # Get the workflow's nodes
        nodes = workflow.dryrun_nodes
        # Loop over them.
        for node_key, node in nodes.items():
            # Avoid adding the workflow inputs to the graph if the user doesn't want them.
            if not include_workflow_inputs and isinstance(node, WorkflowInput):
                continue

            # Loop over the inputs of this node that contain other nodes and
            # add the edges that connect them to this node.
            for other_key, input_node in node._input_nodes.items():
                if not include_workflow_inputs and isinstance(input_node, WorkflowInput):
                    continue

                edges.append((
                    workflow.find_node_key(input_node), 
                    node_key, 
                    _edge_props(input_node, node, other_key)
                ))
                    
        return edges

    def to_nx(self, workflow: Type[Workflow], include_workflow_inputs: bool = False, 
        edge_labels: bool = False
    ) -> "nx.DiGraph":
        """Convert a Workflow class to a networkx directed graph.

        The nodes of the graph are the node functions that compose the workflow.
        The edges represent connections between nodes, where one node's output is
        sent to another node's input.
        
        Parameters
        ----------
        workflow: Type[Workflow]
            The workflow class that you want to convert.
        include_workflow_inputs: bool, optional
            Whether to include the workflow inputs in the graph.
        edge_labels: bool, optional
            Whether to add labels to edges. If True, the labels are the key of the input that
            receives the output of the outgoing node.
        """
        import networkx as nx

        # Get the edges
        edges = self._get_edges(workflow, include_workflow_inputs=include_workflow_inputs, edge_labels=edge_labels)
        
        # And build the networkx directed graph
        graph = nx.DiGraph()
        graph.add_edges_from(edges, )

        for name, node_key in workflow.dryrun_nodes.named_vars.items():
            graph.nodes[node_key]['label'] = name
            
        return graph

    def to_pyvis(self, colorscale: str = "viridis", show_workflow_inputs: bool = False, 
        edge_labels: bool = True, node_help: bool = True,
        notebook: bool = False, hierarchial: bool = True, inputs_props: Dict[str, Any] = {}, node_props: Dict[str, Any] = {},
        leafs_props: Dict[str, Any] = {}, output_props: Dict[str, Any] = {},
        auto_text_color: bool = True,
        to_export: Union[bool, None] = None,
    ):        
        """Convert a Workflow class to a pyvis network for visualization.

        The nodes of the graph are the node functions that compose the workflow.
        The edges represent connections between nodes, where one node's output is
        sent to another node's input.
        
        Parameters
        ----------
        colorscale: str, optional
            The matplotlib colorscale to use for coloring.
        show_workflow_inputs: bool, optional
            Whether to include the workflow inputs in the graph.
        edge_labels: bool, optional
            Whether to add labels to edges. If True, the labels are the key of the input that
            receives the output of the outgoing node.
        node_help: bool, optional
            Whether to add the node class help as node metadata so that it can be shown on node hover.
        notebook: bool, optional
            Whether the network will be plotted in a jupyter notebook.
        hierarchial: bool, optional
            Whether the nodes vertical disposition needs to follow the order with which the nodes
            take part in the workflow.

            This can help understanding the structure of the workflow, but in some cases it might result
            in a too tight and hard to read layout.
        node_props: dict, optional
            Properties to assign to the nodes, on top of the default ones.
        leafs_props: dict, optional
            Properties to assign to the leaf nodes (those whose output is not connected to another node),
            on top of the default ones.
        output_props: dict, optional
            Properties to assign to the output node, on top of the default ones.
        auto_text_color: bool, optional
            Whether to automatically set the text color of the nodes to white or black depending on the
            background color.
        to_export: bool, optional
            Not important for normal use.
            Whether the notebook is thought to be exported to html.
            If None, it is taken from the "SISL_NODES_EXPORT_VIS" environment variable.
        """
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import networkx as nx
            from pyvis.network import Network as visNetwork
        except ModuleNotFoundError:
            raise ModuleNotFoundError("You need to install the 'networkx', 'pyvis' and 'matplotlib' packages to visualize workflows.")

        if to_export is None:
            to_export = get_environ_variable("SISL_NODES_EXPORT_VIS") != False
        
        # Get the networkx directed graph
        graph = self.to_nx(self._workflow, include_workflow_inputs=show_workflow_inputs, edge_labels=edge_labels)
        # Find out the generations of nodes (i.e. how many nodes are there until a beginning of the graph)
        topo_gens = list(nx.topological_generations(graph))

        # Find out whether the first generation is just the workflow inputs,
        # and whether the last generation is just the output node.
        # This will help us avoid these generations when coloring on generation,
        # allowing for more color range.
        inputs_are_first_gen = all(node in self._workflow.dryrun_nodes.inputs for node in topo_gens[0])

        output_node_key = "output"
        output_is_last_gen = len(topo_gens[-1]) == 1 and output_node_key == topo_gens[-1][0]

        # Then determine the range
        min_gen = 0
        max_gen = len(topo_gens) - 1
        if inputs_are_first_gen:
            min_gen += 1
        if output_is_last_gen:
            max_gen -= 1
        color_range = max_gen - min_gen

        # Two helper functions to get the help message for nodes (which will be shown on hover)
        def _get_node_help(node: Union[Node, Type[Node]]):
            if isinstance(node, Node):
                _get_node_inputs_str(node)
            else:
                node_cls = node
                short_doc = (node_cls.__doc__ or "").lstrip().split('\n')[0] or "No documentation"
                sig = "\n".join(str(param) for param in node_cls.__signature__.parameters.values())
                return f"Node class: {node_cls.__name__}\n{short_doc}\n............................................\n{sig}"

        def _get_node_inputs_str(node: Node) -> str:
            short_doc = (node.__class__.__doc__ or "").lstrip().split('\n')[0] or "No documentation"

            node_inputs_str = f"Node class: {node.__class__.__name__}\n{short_doc}\n............................................\n"

            def _render_value(v):
                if isinstance(v, WorkflowInput):
                    v = f"Linked( {self._workflow.__name__}.{v.input_key} )"
                elif isinstance(v, Node):
                    node_key = self._workflow.find_node_key(v, v.__class__.__name__)
                    v = f"Linked( {node_key}.output )"
                return v

            for k, v in node.inputs.items():
                if v is None:
                    pass
                elif k == node._args_inputs_key:
                    v = tuple(_render_value(v) for v in v)
                elif k == node._kwargs_inputs_key:
                    v = {k: _render_value(v) for k, v in v.items()}
                else:
                    v = _render_value(v)
                node_inputs_str += f"{k} = {v}\n"

            return node_inputs_str

        # Get the requested colorscale from matplotlib
        # in matplotlib > 3.5 mpl.colormaps[colorscale]
        # This is portable
        cmap = plt.get_cmap(colorscale)
        def rgb2gray(rgb):
            return rgb[0] * 0.2989 + rgb[1] * 0.5870 + rgb[2] * 0.1140
        # Loop through generations of nodes.
        for i, nodes in enumerate(topo_gens):

            # Get the level, color and shape for this generation
            level = float(i + 1)

            if color_range != 0:
                color_value = (i - min_gen) / color_range
            else:
                color_value = 0
            
            # Get the color for this generation
            rgb = cmap(color_value)
            color = mpl.colors.rgb2hex(rgb)
        
            shape = node_props.get("shape", "circle")

            # If automatic text coloring is requested, set the font color to white if the background is dark.
            # But only if the shape is an ellipse, circle, database, box or text, as these are the only ones
            # that have the text inside.
            font = {}
            if auto_text_color and shape in ["ellipse", "circle", "database", "box", "text"]:
                gray = rgb2gray(rgb)
                if gray <= 0.5:
                    font = {"color": "white"}

            # Loop through nodes of this generation and set their properties.
            for node in nodes:
                graph_node = graph.nodes[node]

                if node_help:
                    node_obj = self._workflow.dryrun_nodes.get(node)
                    title = _get_node_inputs_str(node_obj) if node_obj is not None else ""
                else:
                    title = ""

                graph_node.update({
                    "mass": 2, "shape": shape, "color": color, "level": level, 
                    "title": title, "font": font,
                    **node_props
                })

        # Set the props of leaf nodes (those that have their output linked to nothing)
        leaves = [k for k, v in graph.out_degree if v == 0]
        for leaf in leaves:
            graph.nodes[leaf].update({
                "shape": "square", "font": {"color": "black"}, **leafs_props
            })
        
        # Set the props of the output node
        graph.nodes[output_node_key].update({
            "color": "pink", "shape": "box", "label": "Output",
            "font": {"color": "black"}, **output_props
        })

        if show_workflow_inputs:
            for k, node in self._workflow.dryrun_nodes.inputs.items():
                # Find out the minimum level of the nodes that use this workflow input

                if k not in graph.nodes:
                    graph.add_node(k)

                if node._output_links:
                    min_level = min(
                        graph.nodes[self._workflow.find_node_key(node)].get("level", 0) 
                        for node in node._output_links
                    )
                else:
                    min_level = 2

                graph.nodes[k].update({
                    "color": "#ccc", "font": {"color": "black"}, "shape": "box", 
                    "label": f"Input({k})", "level": min_level - 1,
                    **inputs_props
                })   
        
        layout = {}
        if hierarchial:
            layout["hierarchial"] = True

        net = visNetwork(notebook=notebook, directed=True, layout=layout, cdn_resources="remote")
        net.from_nx(graph)
        
        net.toggle_physics(True)
        net.toggle_stabilization(True)
        return net

    @staticmethod
    def _show_pyvis(net: "pyvis.Network", notebook: bool, to_export: Union[bool, None]):
        """Shows a pyvis network.

        This is implemented here because pyvis implementation of `show` is very dangerous,
        as it can remove the `lib` directory if present in the working directory. It also
        needs to create a file in the working directory's scope.

        Parameters
        ----------
        net: pyvis.Network
            The network to show.
        notebook: bool
            Whether the plot is to be displayed in a jupyter notebook.
        """
        if to_export is None:
            to_export = get_environ_variable("SISL_NODES_EXPORT_VIS") != False

        # Get the html for the network plot.
        html_text = net.generate_html(notebook=notebook)

        if notebook:
            # Render the HTML in the notebook.
            
            from IPython.display import HTML, display

            # First option was to display as an HTML object
            # The "isolated" flag avoids the CSS to affect the rest of the notebook.
            # The wrapper div is needed because otherwise the HTML display is of height 0.
            # HOWEVER: IT DOESN'T DISPLAY when exported to HTML because the iframe that isolated=True
            # creates is removed.
            #obj = HTML(f'<div style="height:{net.height}">{html}</div>', metadata={"isolated": True}, )
            # Instead, we create an iframe ourselves, using the srcdoc attribute. The only thing that we need to worry
            # is that there are no double quotes in the html, otherwise the srcdoc attribute will be broken.
            # So we replace " by &#34;, the html entity for ".
            if to_export:
                escaped_html = html.escape(html_text)
                obj = HTML(f"""<iframe height='{net.height}' srcdoc="{escaped_html}" style="width:100%"></iframe>""")
            else:  
                obj = HTML(f'<div style="height:{net.height}">{html_text}</div>', metadata={"isolated": True}, )

            display(obj)
        else:
            # Write the html in a temporary file and open it with a web browser.
            # The temporary file must not be deleted before opening it with the
            # webbrowser.
            import tempfile
            import webbrowser

            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
                f.write(html_text)
                name = f.name
        
            webbrowser.open(name)

    def visualize(self, colorscale: str = "viridis", show_workflow_inputs: bool = True, 
        edge_labels: bool = True, node_help: bool = True,
        notebook: bool = False, hierarchial: bool = True, node_props: Dict[str, Any] = {},
        inputs_props: Dict[str, Any] = {}, leafs_props: Dict[str, Any] = {}, output_props: Dict[str, Any] = {},
        to_export: Union[bool, None] = None,
    ):
        """Visualize the workflow's network in a plot.

        The nodes of the graph are the node functions that compose the workflow.
        The edges represent connections between nodes, where one node's output is
        sent to another node's input.
        
        Parameters
        ----------
        colorscale: str, optional
            The matplotlib colorscale to use for coloring.
        show_workflow_inputs: bool, optional
            Whether to include the workflow inputs in the graph.
        edge_labels: bool, optional
            Whether to add labels to edges. If True, the labels are the key of the input that
            receives the output of the outgoing node.
        node_help: bool, optional
            Whether to add the node class help as node metadata so that it can be shown on node hover.
        notebook: bool, optional
            Whether the network will be plotted in a jupyter notebook.
        hierarchial: bool, optional
            Whether the nodes vertical disposition needs to follow the order with which the nodes
            take part in the workflow.

            This can help understanding the structure of the workflow, but in some cases it might result
            in a too tight and hard to read layout.
        node_props: dict, optional
            Properties to assign to the nodes, on top of the default ones.
        inputs_props: dict, optional
            Properties to assign to the workflow inputs, on top of the default ones.
        leafs_props: dict, optional
            Properties to assign to the leaf nodes (those whose output is not connected to another node),
            on top of the default ones.
        output_props: dict, optional
            Properties to assign to the output node, on top of the default ones.
        to_export: bool, optional
            Not important for normal use.
            Whether the notebook is thought to be exported to html.
            If None, it is taken from the "SISL_NODES_EXPORT_VIS" environment variable.
        """
        if to_export is None:
            to_export = get_environ_variable("SISL_NODES_EXPORT_VIS") != False

        net = self.to_pyvis(
            colorscale=colorscale, show_workflow_inputs=show_workflow_inputs, edge_labels=edge_labels, 
            node_help=node_help, notebook=notebook,
            hierarchial=hierarchial, node_props=node_props,
            inputs_props=inputs_props, leafs_props=leafs_props, output_props=output_props,
            to_export=to_export,
        )

        return self._show_pyvis(net, notebook=notebook, to_export=to_export)

class WorkflowNodes:

    inputs: Dict[str, WorkflowInput]
    workers: Dict[str, Node]
    output: WorkflowOutput
    named_vars: Dict[str, str]

    def __init__(self, inputs: Dict[str, WorkflowInput], workers: Dict[str, Node], output: WorkflowOutput, named_vars: Dict[str, str]):
        self.inputs = inputs
        self.workers = workers
        self.output = output

        self.named_vars = named_vars

        self._all_nodes = ChainMap(self.inputs, self.workers, {"output": self.output})

    @classmethod
    def from_workflow_run(cls, inputs: Dict[str, WorkflowInput], output: WorkflowOutput, named_vars: Dict[str, Node]):

        # Gather all worker nodes inside the workflow.
        workers = cls.gather_from_inputs_and_output(inputs.values(), output=output)

        # Construct the table that will map from "human friendly" names to node keys.
        _named_vars = {}
        for k, v in named_vars.items():
            for node_key, node in workers.items():
                if node is v:
                    _named_vars[k] = node_key
                    break

        return cls(inputs=inputs, workers=workers, output=output, named_vars=_named_vars)
    
    @classmethod
    def from_node_tree(cls, output_node):

        # Gather all worker nodes inside the workflow.
        workers = cls.gather_from_inputs_and_output([], output=output_node)
        
        # Dictionary that will store the workflow input nodes.
        wf_inputs = {}
        # The workers found by traversing are node instances that might be in use
        # by the user, so we should create copies of them and store them in this new_workers
        # dictionary. Additionally, we need a mapping from old nodes to new nodes in order
        # to update the links between nodes.
        new_workers = {}
        old_to_new = {}
        # Loop through the workers.
        for k, node in workers.items():
            # Find out the inputs that we should connect to the workflow inputs. We connect all inputs
            # that are not nodes, and that are not the args or kwargs inputs.
            node_inputs = {
                param_k: WorkflowInput(input_key=f"{node.__class__.__name__}_{param_k}", value=node.inputs[param_k]) 
                for param_k, v in node.inputs.items() if not (
                    isinstance(v, Node) or param_k == node._args_inputs_key or param_k == node._kwargs_inputs_key
                )
            }

            # Create a new node using the newly determined inputs. However, we keep the links to the old nodes
            # These inputs will be updated later.
            with temporal_context(lazy=True):
                new_workers[k] = node.__class__().update_inputs(**{**node.inputs, **node_inputs})

            # Register this new node in the mapping from old to new nodes.
            old_to_new[id(node)] = new_workers[k]
            
            # Update the workflow inputs dictionary with the inputs that we have determined
            # to be connected to this node. We use the node class name as a prefix to avoid
            # name clashes. THIS IS NOT PERFECT, IF THERE ARE TWO NODES OF THE SAME CLASS
            # THERE CAN BE A CLASH.
            wf_inputs.update({
                f"{node.__class__.__name__}_{param_k}": v for param_k, v in node_inputs.items()
            })  
        
        # Now that we have all the node copies, update the links to old nodes with
        # links to new nodes.
        for k, node in new_workers.items():
            
            new_node_inputs = {}
            for param_k, v in node.inputs.items():
                if param_k == node._args_inputs_key:
                    new_node_inputs[param_k] = [old_to_new[id(n)] if isinstance(n, Node) else n for n in v]
                elif param_k == node._args_inputs_key:
                    new_node_inputs[param_k] = {k: old_to_new[id(n)] if isinstance(n, Node) else n for k, n in v.items()}
                elif isinstance(v, Node) and not isinstance(v, WorkflowInput):
                    new_node_inputs[param_k] = old_to_new[id(v)]
            
            with temporal_context(lazy=True):
                node.update_inputs(**new_node_inputs)

        # Create the workflow output.
        new_output = WorkflowOutput(value=old_to_new[id(output_node)])
        
        # Initialize and return the WorkflowNodes object.
        return cls(inputs=wf_inputs, workers=new_workers, output=new_output, named_vars={})

    def __dir__(self) -> Iterable[str]:
        return dir(self.named_vars) + dir(self._all_nodes)

    def __getitem__(self, key):
        if key in self.named_vars:
            key = self.named_vars[key]
        return self._all_nodes[key]

    def __getattr__(self, key):
        if key in self.named_vars:
            key = self.named_vars[key]
        return self._all_nodes[key]

    def items(self):
        return self._all_nodes.items()

    def values(self):
        return self._all_nodes.values()

    def get(self, key):
        if key in self.named_vars:
            key = self.named_vars[key]
        return self._all_nodes.get(key)

    def __str__(self):
        return f"Inputs: {self.inputs}\n\nWorkers: {self.workers}\n\nOutput: {self.output}\n\nNamed nodes: {self.named_vars}"

    @staticmethod
    def gather_from_inputs_and_output(inputs: Sequence[WorkflowInput], output: WorkflowOutput) -> Dict[str, Node]:
        # Get a list of all nodes.
        nodes = []

        # Function that will receive a node and add it to the list if it's
        # not already there.
        def add_node(node):
            if any(node is reg_node for reg_node in nodes):
                return
            nodes.append(node)

        # Visit all nodes that depend on the inputs.
        traverse_tree_forward(inputs, add_node)
        traverse_tree_backward((output, ), add_node)

        # Now build a dictionary that contains all nodes.
        dict_nodes = {}
        for node in nodes:
            if isinstance(node, (WorkflowInput, WorkflowOutput)):
                continue

            # Determine the name we are going to assign to this node inside the workflow.
            node_name = node.__class__.__name__
            name = node_name
            i = 0
            while name in dict_nodes:
                i += 1
                name = f"{node_name}_{i}"

            # Add it to the dictionary of nodes
            dict_nodes[name] = node
        
        return dict_nodes

    def copy(self, inputs: Dict[str, Any] = {}) -> "WorkflowNodes":
        """Creates a copy of the workflow nodes.

        Parameters
        ----------
        inputs: dict, optional
            The inputs to be used in the copy. If not provided, the inputs of the original
            workflow will be used.
        """
        # First, create a copy of the inputs with the new values.
        new_inputs = {}
        for input_k, input_node in self.inputs.items():
            new_inputs[input_k] = input_node.__class__(
                input_key=input_node.input_key, value=inputs.get(input_k, input_node.value)
            )

        # Now create a copy of the worker nodes
        old_ids_to_key  = {
            id(node): key for key, node in self.workers.items()
        }
        new_output = []
        new_workers = {}
        
        def copy_node(node):
            node_id = id(node)
            
            # If it is a WorkflowInput node, return the already created copy.
            if isinstance(node, WorkflowInput):
                return new_inputs[node.input_key]
            # If it is a WorkflowOutput node, return the already created copy if it is there.
            elif isinstance(node, WorkflowOutput):
                if len(new_output) == 1:
                    return new_output[0]
            # If this is an already copied node, just return it.
            elif node_id not in old_ids_to_key:
                return node

            node_key = old_ids_to_key.get(node_id)
            # This is an old node that we already copied, so we can just return the copy.
            if node_key in new_workers:
                return new_workers[node_key]
            
            # This is an old node that we haven't copied yet, so we need to copy it.
            # Get the new inputs, copying nodes if needed.
            new_node_inputs = node.map_inputs(
                inputs=node.inputs, only_nodes=True,
                func=copy_node
            )
            
            # Initialize a new node with the new inputs.
            args, kwargs = node._sanitize_inputs(new_node_inputs)
            new_node = node.__class__(*args, **kwargs)

            # Add the new node. If it's the output node, we save it in its own variable.
            if isinstance(new_node, WorkflowOutput):
                new_output.append(new_node)
            else:
                new_workers[node_key] = new_node

            return new_node
        
        with temporal_context(lazy=True):
            traverse_tree_forward(list(self.inputs.values()), copy_node)
            traverse_tree_backward((self.output, ), copy_node)

        assert len(new_output) == 1 and isinstance(new_output[0], WorkflowOutput)
            
        return self.__class__(inputs=new_inputs, workers=new_workers, output=new_output[0], named_vars=self.named_vars)

class Workflow(Node):
    # The nodes of the initial dry run. This is a class property
    # that helps us understand the flow of the workflow.
    dryrun_nodes: WorkflowNodes

    # The nodes of the workflow instance.
    nodes: WorkflowNodes

    network = NetworkDescriptor()

    @classmethod
    def from_node_tree(cls, output_node: Node, workflow_name: Union[str, None] = None):
        """Creates a workflow class from a node.
        
        It does so by recursively traversing the tree in the inputs direction until
        it finds the leaves.
        All the nodes found are included in the workflow. For each node, inputs 
        that are not nodes are connected to the inputs of the workflow.

        Parameters
        ----------
        output_node: Node
            The final node, that should be connected to the output of the workflow.
        workflow_name: str, optional
            The name of the new workflow class. If None, the name of the output node
            will be used.

        Returns
        -------
        Workflow
            The newly created workflow class.
        """
        # Create the node manager for the workflow.
        dryrun_nodes = WorkflowNodes.from_node_tree(output_node)
        
        # Create the signature of the workflow from the inputs that were determined
        # by the node manager.
        signature = inspect.Signature(parameters=[
            inspect.Parameter(inp.input_key, inspect.Parameter.KEYWORD_ONLY, default=inp.value) for inp in dryrun_nodes.inputs.values()  
        ])

        def function(*args, **kwargs):
            raise NotImplementedError("Workflow class created from node tree. Calling it as a function is not supported.")

        function.__signature__ = signature

        # Create the class and return it.
        return type(
            workflow_name or output_node.__class__.__name__, 
            (cls,), 
            {"dryrun_nodes": dryrun_nodes, "__signature__": signature, "function": staticmethod(function)}
        )

    def setup(self, *args, **kwargs):
        self.nodes = self.dryrun_nodes
        super().setup(*args, **kwargs)

        self.nodes = self.dryrun_nodes.copy(inputs=self._inputs)

    def __init_subclass__(cls):
        # If this is just a subclass of Workflow that is not meant to be ran, continue 
        if not hasattr(cls, "function"):
            return super().__init_subclass__()
        # Also, if the node manager has already been created, continue.
        if "dryrun_nodes" in cls.__dict__:
            return super().__init_subclass__()

        # Otherwise, do all the setting up of the class

        # Initialize the workflow's vars dictionary, which will store the mapping from
        # named variables to nodes.
        named_vars = {}

        def assign_workflow_var(value: Any, var_name: str):
            original_name = var_name
            repeats = 0
            while var_name in named_vars:
                repeats += 1
                var_name = f"{original_name}_{repeats}"
            if var_name in named_vars:
                raise ValueError(f"Variable {var_name} has already been assigned a value, in workflows you can't overwrite variables.")
            named_vars[var_name] = value
            return value

        # Get the workflow function
        work_func = cls.function
        # Nodify it, passing the middleware function that will assign the variables to the workflow.
        work_func = nodify_func(work_func, assign_fn=assign_workflow_var)

        # Get the signature of the function.
        sig = inspect.signature(work_func)

        # Run a dryrun of the workflow, so that we can understand how the nodes are connected. 
        # To this end, nodes must behave lazily.
        with temporal_context(lazy=True): 
            # Define all workflow inputs.
            inps = {
                k: WorkflowInput(
                    input_key=k, 
                    value=param.default if param.default != inspect.Parameter.empty else Node._blank
                ) for k, param in sig.parameters.items()
                if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }

            # Run the workflow function.
            final_node = work_func(**inps)

            # Connect the final node to the output of the workflow.
            out = WorkflowOutput(value=final_node)
        
        # Store all the nodes of the workflow.
        cls.dryrun_nodes = WorkflowNodes.from_workflow_run(inputs=inps, output=out, named_vars=named_vars)

        return super().__init_subclass__()

    def __dir__(self) -> Iterable[str]:
        return [*super().__dir__(), *list(self._vars)]

    @classmethod
    def final_node_key(cls, *args) -> str:
        """Returns the key of the final (output) node of the workflow."""
        return cls.find_node_key(cls.dryrun_nodes.output._inputs['value'], *args)

    @classmethod
    def find_node_key(cls, node, *args) -> str:
        """Returns the identifier key of a node in this workflow"""
        for k, v in cls.dryrun_nodes.items():
            if v is node:
                return k

        if len(args) == 1:
            return args[0]

        raise ValueError(f"Could not find node {node} in the workflow. Workflow nodes {cls.dryrun_nodes.items()}")

    def get(self):
        """Returns the up to date output of the workflow.
        
        It will recompute it if necessary.
        """
        return self.nodes.output.get()
    
    def update_inputs(self, **inputs):
        """Updates the inputs of the workflow."""
        # Be careful here: 
        
        # We should implement something that halts automatic recalculation.
        # Otherwise the nodes will be recalculated every time we update each
        # individual input.

        for input_key, value in inputs.items():
            self.nodes.inputs[input_key].update_inputs(value=value)
        
        self._inputs.update(inputs)
                
        return self
    
    def _get_output(self):
        return self.nodes.output._output
    
    def _set_output(self, value):
        self.nodes.output._output = value
    
    _output = property(_get_output, _set_output)

class NodeConverter(ast.NodeTransformer):
    """AST transformer that converts a function into a workflow."""
    
    def __init__(self, *args, assign_fn: Union[str, None] = None, node_cls_name: str = "Node", **kwargs):
        super().__init__(*args, **kwargs)
        
        self.assign_fn = assign_fn
        self.node_cls_name = node_cls_name

    def visit_Call(self, node):
        """Converts some_module.some_attr(some_args) into Node.from_func(some_module.some_attr)(some_args)"""
        node2 = ast.Call(
            func=ast.Call(
                func=ast.Attribute(value=ast.Name(id=self.node_cls_name, ctx=ast.Load()), attr='from_func', ctx=ast.Load()),
                args=[self.visit(node.func)], keywords=[]),
            args=[self.visit(arg) for arg in node.args], 
            keywords=[self.visit(keyword) for keyword in node.keywords]
        )
        
        ast.fix_missing_locations(node2)
            
        return node2
    
    def visit_Assign(self, node):
        """Converts some_module.some_attr(some_args) into Node.from_func(some_module.some_attr)(some_args)"""
        
        if self.assign_fn is None:
            return self.generic_visit(node)
        if len(node.targets) > 1 or not isinstance(node.targets[0], ast.Name):
            return self.generic_visit(node)
        
        node.value = ast.Call(
            func=ast.Name(id=self.assign_fn, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg='value', value=self.visit(node.value)),
                ast.keyword(arg='var_name', value=ast.Constant(value=node.targets[0].id))
            ],
        )
        
        ast.fix_missing_locations(node.value)
        
        return node
    
    def visit_List(self, node):
        """Converts the list syntax into a call to the ListSyntaxNode."""
        if all(isinstance(elt, ast.Constant) for elt in node.elts):
            return self.generic_visit(node)

        new_node = ast.Call(
            func=ast.Name(id="ListSyntaxNode", ctx=ast.Load()),
            args=[self.visit(elt) for elt in node.elts],
            keywords=[]
        )

        ast.fix_missing_locations(new_node)

        return new_node
    
    def visit_Tuple(self, node):
        """Converts the tuple syntax into a call to the TupleSyntaxNode."""
        if all(isinstance(elt, ast.Constant) for elt in node.elts):
            return self.generic_visit(node)

        new_node = ast.Call(
            func=ast.Name(id="TupleSyntaxNode", ctx=ast.Load()),
            args=[self.visit(elt) for elt in node.elts],
            keywords=[]
        )

        ast.fix_missing_locations(new_node)

        return new_node
    
    def visit_Dict(self, node: ast.Dict) -> Any:
        """Converts the dict syntax into a call to the DictSyntaxNode."""
        if all(isinstance(elt, ast.Constant) for elt in node.values):
            return self.generic_visit(node)
        if not all(isinstance(elt, ast.Constant) for elt in node.keys):
            return self.generic_visit(node)
        
        new_node = ast.Call(
            func=ast.Name(id="DictSyntaxNode", ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg=key.value, value=self.visit(value))
                for key, value in zip(node.keys, node.values)
            ],
        )

        ast.fix_missing_locations(new_node)

        return new_node
    

    
def nodify_func(
    func: FunctionType, 
    transformer_cls: Type[NodeConverter] = NodeConverter, 
    assign_fn: Union[Callable, None] = None, 
    node_cls: Type[Node] = Node
) -> FunctionType:
    """Converts all calculations of a function into nodes.
    
    This is used for example to convert a function into a workflow.

    The conversion is done by getting the function's source code, parsing it 
    into an abstract syntax tree, modifying the tree and recompiling.

    Parameters
    ----------
    func : Callable
        The function to convert.
    transformer_cls : Type[NodeConverter], optional
        The NodeTransformer class to that is used to transform the AST.
    assign_fn : Union[Callable, None], optional
        A function that will be placed as middleware for variable assignments.
        It will be called with the following arguments:
            - value: The value assigned to the variable.
            - var_name: The name of the variable that will be assigned.
    node_cls : Type[Node], optional
        The Node class to which function calls will be converted.
    """
    # Get the function's namespace.
    closurevars = inspect.getclosurevars(func)
    func_namespace = {**closurevars.nonlocals, **closurevars.globals, **closurevars.builtins}

    # Get the function's source code.
    code = inspect.getsource(func)
    # Make sure the first line is at the 0 indentation level.
    code = textwrap.dedent(code)
    
    # Parse the source code into an AST.
    tree = ast.parse(code)

    # If the function has decorators, remove them. Perhaps in the future we can
    # support arbitrary decorators.
    decorators = tree.body[0].decorator_list
    if len(decorators) > 0:
        warn(f"Decorators are ignored for now on workflow creation. Ignoring {len(decorators)} decorators on {func.__name__}")
        tree.body[0].decorator_list = []
    
    # The alias of the assign_fn function, which we make sure does not conflict
    # with any other variable in the function's namespace.
    assign_fn_key = None
    if assign_fn is not None:
        assign_fn_key = "__assign_fn"
        while assign_fn_key in func_namespace:
            assign_fn_key += "_"

    # We also make sure that the name of the node class does not conflict with
    # any other variable in the function's namespace.
    node_cls_name = node_cls.__name__
    while assign_fn_key in func_namespace:
        assign_fn_key += "_"
    
    # Transform the AST.
    transformer = transformer_cls(assign_fn=assign_fn_key, node_cls_name=node_cls_name)
    new_tree = transformer.visit(tree)

    # Compile the new AST into a code object. The filename is fake, but it doesn't
    # matter because there is no file to map the code to. 
    # (we could map it to the original function in the future)
    code_obj = compile(new_tree, "compiled_workflows", "exec")
    
    # Add the needed variables into the namespace.
    namespace = {
        node_cls_name: node_cls, 
        "ListSyntaxNode": ListSyntaxNode, "TupleSyntaxNode": TupleSyntaxNode, "DictSyntaxNode": DictSyntaxNode, 
        **func_namespace, 
    }
    if assign_fn_key is not None:
        namespace[assign_fn_key] = assign_fn

    # Execute the code, and retrieve the new function from the namespace.
    exec(code_obj, namespace)
    new_func = namespace[func.__name__]

    return new_func
