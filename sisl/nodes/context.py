import contextlib

def set_lazy_computation(nodes: bool = True, workflows: bool = True):
    """Set the lazy computation mode for nodes and workflows.
    
    Parameters
    ----------
    nodes: bool, optional
        Whether lazy computation is turned on for nodes.
    workflows: bool, optional
        Whether lazy computation is turned on for workflows.
    """
    from .node import Node
    from .workflow import Workflow

    Node._lazy_computation = nodes
    Workflow._lazy_computation = workflows

@contextlib.contextmanager
def lazy_context(nodes: bool = True, workflows: bool = True):
    from .node import Node
    from .workflow import Workflow

    old_lazy = {
        "nodes": Node._lazy_computation,
        "workflows": Workflow._lazy_computation,
    }

    set_lazy_computation(nodes, workflows)
    try:
        yield
    except Exception as e:
        set_lazy_computation(**old_lazy)
        raise e
    
    set_lazy_computation(**old_lazy)