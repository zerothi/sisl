import contextlib
from collections import ChainMap
from typing import Any, Union

# The main sisl nodes context that all nodes will use by default as their base.
SISL_NODES_CONTEXT = dict(
    # Whether the nodes should compute lazily or immediately when inputs are updated.
    lazy=True,
    # On initialization, should the node compute? If None, defaults to `lazy`.
    lazy_init=None,
    # The level of logs stored in the node.
    log_level="INFO"
)

# Temporal contexts stack. It should not be used directly by users, the aim of this
# stack is to populate it when context managers are used. This is a chainmap and 
# not a simple dict because we might have nested context managers.
_TEMPORAL_CONTEXTS = ChainMap()

class NodeContext(ChainMap):
    """Extension of Chainmap that always checks on the temporal context first.
    
    Using this class is equivalent to forcing users to have the temporal context
    always in the first position of the chainmap. Since this is not a very nice
    thing to force on users, we use this class instead.

    Keys:
        lazy: bool
            If `False`, nodes will automatically recompute if any of their inputs 
            have changed, even if no other node needs their output yet.
        lazy_init: bool or None
            Whether the node should compute on initialization. If None, defaults to
            `lazy`.
        debug: bool
            Whether to print debugging information.
        debug_show_inputs:
            Whether to print the inputs of the node when debugging.
    """

    def __getitem__(self, key: str):
        if key in _TEMPORAL_CONTEXTS:
            return _TEMPORAL_CONTEXTS[key]
        else:
            return super().__getitem__(key)

@contextlib.contextmanager
def temporal_context(context: Union[dict, ChainMap, None] = None, **context_keys: Any):
    """Sets a context temporarily (until the context manager is exited).

    Parameters
    ----------
    context: dict or ChainMap, optional
        The context that should be updated temporarily. This could for example be
        sisl's main context or the context of a specific node class.
        
        If None, the keys and values are forced on all nodes.  
    **context_keys: Any
        The keys and values that should be used for the nodes context.

    Examples
    -------
    Forcing a certain context on all nodes:

    >>> from sisl.nodes import temporal_context
    >>> with temporal_context(lazy=False):
    >>>     # If a node class is called here, the computation will be performed
    >>>     # immediately and the result returned.

    Switching off lazy behavior for workflows:

    >>> from sisl.nodes import Workflow, temporal_context
    >>> with temporal_context(context=Workflow.context, lazy=False):
    >>>     # If a workflow is called here, the computation will be performed
    >>>     # immediately and the result returned, unless that specific workflow
    >>>     # class overwrites the lazy behavior.

    """
    if context is not None:
        # We have to temporally update a context dictionary. We keep a copy of the
        # original context so that we can restore it later.
        old_context = {k: context[k] for k in context_keys}
        context.update(context_keys)

        def _restore():
            # Restore the original context.
            context.update(old_context)
    else:
        # Add this temporal context on top of the temporal contexts stack.
        _TEMPORAL_CONTEXTS.maps.insert(0, context_keys)

        def _restore():
            # Remove the temporal context from the stack.
            del _TEMPORAL_CONTEXTS.maps[0]

    # We have entered the context, execute whatever code is inside the "with" block.
    try:
        yield
        _restore()
    except Exception as e:
        # The block has raised an exception, restore the context and re-raise.
        _restore()
        raise e