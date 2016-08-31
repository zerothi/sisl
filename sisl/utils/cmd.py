"""
Generic utility to create commonly used ArgumentParser
options etc.
"""

def default_namespace(*args, **kwargs):
    class CustomNamespace(object):
        pass
    namespace = CustomNamespace()
    namespace._actions_run = False
    namespace._actions = []
    for key in kwargs:
        setattr(namespace, key, kwargs[key])
    return namespace



def ensure_namespace(p, ns):
    """
    Ensure a namespace passed in case it is not.

    This is currently a hack for:
       https://bugs.python.org/issue27859
    """
    old_parse_known_args = p.parse_known_args
    def parse_known_args(self, argv, namespace=None):
        if namespace is None:
            return old_parse_known_args(argv, ns)
        return old_parse_known_args(argv, namespace)
    p.parse_known_args = parse_known_args


def dec_default_AP(*A_args, **A_kwargs):
    """
    Decorator for routines which takes a parser as argument
    and ensures that it is _not_ `None`.
    """
    def default_AP(func):
        # This requires that the first argument
        # for the function is the parser with default=None
        from argparse import ArgumentParser
    
        def new_func(self, parser=None, *args, **kwargs):
            if parser is None:
                # Create the new parser and insert in the
                # argument list
                parser = ArgumentParser(*A_args, **A_kwargs)
            return func(self, parser, *args, **kwargs)
        return new_func
    return default_AP


def dec_collect_actions(func):
    """
    Decorator for collecting actions until the namespace `_actions_run` is `True`.

    Note that the `Namespace` object is the 2nd argument.
    """
    def collect(self, *args, **kwargs):
        if args[1]._actions_run:
            return func(self, *args, **kwargs)
        # Else we append the actions to be performed
        namespace._actions.append( (self, args, kwargs) )
        return None
    return collect


def dec_run_actions(func):
    """
    Decorator for running collected actions.

    Note that the `Namespace` object is the 2nd argument.
    """
    def run(self, *args, **kwargs):
        args[1]._actions_run = True
        # Run all so-far collected actions
        for A, Aargs, Akwargs in args[1]._actions:
            A(*Aargs, **Akwargs)
        args[1]._actions_run = False
        args[1]._actions = []
        return func(self, *args, **kwargs)
    return run
