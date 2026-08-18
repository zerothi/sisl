# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Automatic CLI generation for sisl data classes.

This file contains the functions to automatically create a CLI to
generate and postprocess sisl's data classes.

There are multiple ways in which you might want to extend the CLI:

 - By adding a way of generating data from a certain file type. This is
 the most likely scenario. In that case, all the modifications required
 are OUTSIDE of this file. You need to:

    1. Create a function that generates the data from the file. Add type
    annotations to it. E.g.:

    ```python
    import DOSData

    def dos_from_bandsSile(
        sile: sisl.io.bandsSileSiesta,
        Erange: Tuple[float, float] = (-2, 2)
    ):
        # Get energies and DOS somehow
        ...
        E = ...
        DOS = ...

        return DOSData.new(E, DOS)
    ```

    2. Register your function as a way of creating DOSData:
    ```python
    DOSData.new.register("siesta-bands", dos_from_bandsSile)
    ```

    3. This data generating function should be automatically added to the CLI as
    'siesta-bands'. Enjoy!

  - By adding a post processing method to a data class. In this case, all you need to
    do is to add the method to the data class and make sure it is type annotated. The
    method should then automatically show up in the CLI.

  - By adding a new type of data. In this case, you need to create the data class and
    make sure that it is added to the CLI in this file. You need to add a new line to
    the `main` function that calls `add_data_group` with the new data class.

Alternatively, it is also possible that you might want to change the way the CLI works.
This is of course more complex, but here is a sketch of how the CLI works to help
pinpoint where you might want to make changes:

    - The CLI is a click CLI.
    - The CLI is created on `main`.
    - The main app is created and then we add a group for each data class. Each group
      contains:
        - Commands to generate data:
          We automatically gather all the possible sources from the `new` dispatcher
          of the data class. We then decorate them automatically with
          `decorate_data_gen_function` to be used in click based on type annotations.
        - Commands to post process data:
          We automatically gather all the methods of the data class that are not private
          and decorate them with `decorate_data_method` to be used in click based on type
          annotations.

    - The way we can generate and post process data in the same CLI is by allowing commands
      to be chained. However, since commands execute independently, we need a global storage
      so that the post processing commands can access the generated data. We store the
      generated data in the `GENERATED` global variable.

    - To decorate functions automatically based on type, we rely on the `decorate_click`
      function from `nodify`. However, there are some types that are specific to `sisl`. We
      need to define how those types will be generated from the values (strings) passed in
      the `CLI`. We do that in the `custom_type_kwargs` dictionary.

"""
import functools
import inspect
import sys
from typing import Callable, Optional

import rich_click as click
from nodify.cli._click import decorate_click

import sisl
from sisl.data import Data, DOSData

# Some configuration to tweak CLICK display.
click.rich_click.TEXT_MARKUP = "markdown"
click.rich_click.SHOW_ARGUMENTS = True

# Define all the possible annotations that are custom to sisl and
# should be supported.
custom_type_kwargs = {
    # All sile classes. If a CLI has an argument that is annotated as a sile
    # then the value provided to the CLI will be parsed simply as sile_cls(value),
    # i.e. a sile will be created from the user provided path.
    **{sile_cls: {"parser": sile_cls} for sile_cls in sisl.get_siles()},
}

# List that keeps track of the data that the CLI has generated.
# When chaining commands, the first command will generate data and
# the subsequent commands will be methods to apply on that data.
GENERATED = []


def decorate_data_gen_function(
    data_gen: Callable,
    func_args: tuple = (),
):
    """Decorates a data generating function to be used as a click group.

    It also wraps the function so that the generated data is stored in the
    GENERATED list.

    Parameters
    ----------
    data_gen :
        The data generating function.
    func_args :
        Additional (fixed) arguments to pass to the data generating function.
        If provided, the signature of the decorated function will not contain
        these arguments.
    """

    @functools.wraps(data_gen)
    def f(*args, **kwargs):
        data = data_gen(*func_args, *args, **kwargs)
        GENERATED.append(data)
        return data

    if len(func_args) > 0:
        # Remove the first N arguments from the signature, as they are
        # passed as fixed arguments to the function (the user can't change
        # them).
        sig = inspect.signature(f)
        f.__signature__ = sig.replace(
            parameters=list(sig.parameters.values())[len(func_args) :]
        )

    return decorate_click(f, custom_type_kwargs=custom_type_kwargs)


def decorate_data_method(method: Callable):
    """Decorates a method of a data class to be used as a click command.

    It also wraps the function so that the method takes the generated data
    from the GENERATED list.

    Parameters
    ----------
    method :
        The method to decorate.
    """

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        return method(GENERATED[-1], *args, **kwargs)

    # Remove the first argument, which is data that we will take from the
    # GENERATED list.
    sig = inspect.signature(method)
    wrapped.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:])

    return decorate_click(wrapped, custom_type_kwargs=custom_type_kwargs)


def add_data_group(
    data_cls: Data,
    parent_group: click.Group,
    *group_args,
    arg_sile: Optional[sisl.io.Sile] = None,
    **group_kwargs,
):
    """Given a data class, adds a click group to a parent group.

    It can work in two modes:

        - If no sile is provided, the app group contains subcommands to generate
        the data from different sources. E.g. for a data class "dos", the "dos"
        group is used as:
        ```
        sdata dos source_1 ...args
        sdata dos source_2 ...args
        ```
        - If a sile is provided, the app group already knows which source to use
        (because it is defined by the sile). E.g. if "siesta.EIG" has been provided
        as the first argument by the user, the "dos" group is used as:
        ```
        sdata siesta.EIG dos ...args
        ```

    No matter the case, the group will also contain commands that are the methods of
    the data class and that can be applied on the generated data (by chaining). E.g.:
    ```
    sdata dos source_1 ...args method_1 ...method_1_args method_2 ...method_2_args
    ```

    Parameters
    ----------
    data_cls :
        The data class to add to the app.
    parent_group :
        The click group that is the parent of this group. The data class group
        will be added to this parent group.
    group_args :
        Additional arguments to pass to the created click group.
    arg_sile :
        The sile that the app is working with because the user has provided it
        as the first argument. See this function's docstring for more information
        on how the function behaves depending on this argument.
    group_kwargs :
        Additional keyword arguments to pass to the created click group.
    """

    if arg_sile is None:
        # No file path was provided as an argument, therefore we need to register the group
        # with all possible sources.
        @parent_group.group(*group_args, chain=True, **group_kwargs)
        @functools.wraps(data_cls)
        def data_group():
            pass

        # Register all possible sources of data
        for (type_, dispatch), name in zip(
            data_cls.new.dispatcher.registry.items(), data_cls.new.names
        ):
            # The base dispatch uses object as the type, we don't care about it
            if type_ is object:
                continue

            # Register this dispatch
            data_group.command(name=name)(decorate_data_gen_function(dispatch))
    elif arg_sile.__class__ in data_cls.new.dispatcher.registry:
        # The user has provided a file path and the data class implements a way of being
        # created from that file. Register the group so that it already knows
        # the source type.
        dispatch = data_cls.new.dispatcher.registry[arg_sile.__class__]

        data_group = parent_group.group(
            *group_args, invoke_without_command=True, chain=True, **group_kwargs
        )(decorate_data_gen_function(dispatch, func_args=(arg_sile,)))
    else:
        # The user has provided a file path, but this data class does not implement
        # a way of being created from that file. Don't include the group on the app.
        return

    # Loop over all method of the data class and register them as commands of the group
    for k, method in inspect.getmembers(DOSData, predicate=inspect.isfunction):

        if k == "new" or k.startswith("_"):
            continue

        data_group.command()(decorate_data_method(method))


class OrderedCommandsGroup(click.RichGroup):

    def list_commands(self, ctx):
        return list(self.commands)


def main():
    """Function that creates the CLI for sisl's data classes.

    Before creating the CLI app, it checks the first argument to see
    if it is a file. If it is, the app will be created taking into
    account that we already know which file the user wants to work with.

    Parameters
    ----------

    See Also
    --------
    add_data_group
        Function that adds each data class to the CLI app. In its docstring
        you can see how the CLI is structured depending on whether the first
        argument is a file or not.
    """

    # Check if the first argument is a file. We just take the first argument
    # and try to get a sile from it. If this doesn't succeed but we suspect
    # that the first argument is a file because it contains a dot, we raise
    # the error that the sile is not implemented.
    arg_sile = None
    if len(sys.argv) > 1:
        try:
            arg_sile = sisl.get_sile(sys.argv[1])
        except NotImplementedError:
            if "." in sys.argv[1]:
                raise

    # Create the main application
    @click.group()
    def _app():
        """Sisl's data processing CLI.

        This CLI allows you to **generate and post process different types
        of data**.


        It has two operation modes, depending on the first argument:

        - **If the first argument is the type of data**. E.g. `sdata dos ...`.
        In this case, the CLI will request the source from which you want
        to generate the data, and you would do something like:

        ```
        sdata dos source_1 ...args
        ```

        - **If the first argument is a file**, the CLI will assume that you want
        to work with that file. It will only show you the data types that can
        be generated from that file, and you don't need to pass the source
        because it is inferred from the file. E.g.:

        ```
        sdata siesta.EIG dos ...args
        ```

        here the CLI already knows that the data is to be generated from
        a siesta.EIG file.
        """

    base_app = _app

    # If the first argument is a file, we add a group on top of that. Effectively,
    # this new group will be what the user will see as the base app when passing
    # a file.
    # We need to do this because then click will parse the file name as a group
    # name and will enter this app. Another option was to remove the file from
    # sys.argv, but some aesthetic problems arise. E.g. click doesn't show the
    # file name on the Usage section.
    if arg_sile is not None:
        hlp = f"""Data processing CLI for {arg_sile.__class__.__name__}

        **You have passed '{arg_sile.file}'.**

        The CLI assumes that you want to work with that file.

        The CLI will only show you the data types that can be generated from this file.
        Also, you won't need to specify the method to generate the data (source),
        because it is inferredfrom the file.

        To understand the two methods of operation, see the help of the main app:
        ```
        sdata --help
        ```

        """
        base_app = _app.group(
            invoke_without_command=True,
            no_args_is_help=True,
            name=sys.argv[1],
            help=hlp,
        )(lambda: None)

    # Now that we have the base app, add the data classes to it!
    add_data_group(
        DOSData, base_app, "dos", arg_sile=arg_sile, cls=OrderedCommandsGroup
    )
    # New data classes should be added here like:
    # add_data_group(DataClass, base_app, "some_name", arg_sile=arg_sile, cls=OrderedCommandsGroup)

    # Run the app.
    _app()


if __name__ == "__main__":
    main()
