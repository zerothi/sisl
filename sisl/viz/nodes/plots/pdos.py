from __future__ import annotations

from typing import Any, Literal, Sequence, Optional
import typing
from sisl.messages import SislError
from sisl.viz.nodes import Node
from sisl.viz.nodes.context import lazy_context
from sisl.viz.types import OrbitalStyleQuery
from ..processors.pdos import PDOSData, get_PDOS_requests
from ..processors.axes import get_axis_var

from ..plotters import PlotterNodeXY

from .plot import Plot

@Node.from_func
def accept_data(data: Optional[PDOSData] = None):
    if data is None:
        raise ValueError("You need to provide a PDOS data source in `pdos_data`")
    return data

@Node.from_func
def on_update(automatic_recalc=True, input: Any = None, fn: Optional[typing.Callable] = None, input_key: str = "input"):
    if fn is not None:
        kwargs = {input_key: input}
        return fn(**kwargs)

class PdosPlot(Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We have initialized the plot, bind listeners to the nodes so that we can update
        # input fields when they change their outputs.
        with lazy_context(True):
            self._listeners = [
                on_update(input=self._nodes['accept_data'], fn=self._update_requests_options, input_key="PDOSData"),
            ]

        # Run on init
        self._listeners[0].get()

    @staticmethod
    def _workflow(
        pdos_data: Optional[PDOSData] = None, requests: Sequence[OrbitalStyleQuery]=[{"name": "DOS"}], 
        Erange=[-2, 2], E_axis: Literal["x", "y"] = "x",
    ):
        pdos_data = accept_data(pdos_data)

        # The "get_PDOS_requests" node processes PDOS data to get the PDOS for
        # particular orbital requests. This is the old "set_data". We feed it
        # the PDOSData* node.
        PDOS_requests = get_PDOS_requests(
            pdos_data, requests=requests, Erange=Erange
        )

        # Determine what goes on each axis
        x = get_axis_var(axis="x", var="E", var_axis=E_axis, other_var="PDOS")
        y = get_axis_var(axis="y", var="E", var_axis=E_axis, other_var="PDOS")

        # A PlotterNode gets the processed data and creates abstract actions (backend agnostic)
        # that should be performed on the figure. The output of this node
        # must be fed to a canvas (backend specific).
        return PlotterNodeXY(data=PDOS_requests, x=x, y=y, what="line")

    def _update_requests_options(self, PDOSData: Optional[PDOSData]):
        if PDOSData is not None:
            requests_param = self.get_input_field('requests')
            if hasattr(requests_param, "update_options"):
                requests_param.update_options(PDOSData.attrs["geometry"], PDOSData.attrs["spin"])

    def _new_request(self, as_dict: bool = True, **kwargs):
        req_field = self.get_input_field("requests")

        # We make sure that if it is a NC or SOC calculation and no spin is specified,
        # we return the total DOS. Otherwise, "spin" would be set to None and we would
        # get the sum of "total", "x", "y" and "z", which is meaningless.
        spin = req_field.get_query_param("spin").params.spin
        if "spin" not in kwargs and not spin.is_diagonal:
            kwargs["spin"] = ["total"]

        return req_field.complete_query({"name": str(len(self.get_input("requests", parsed=True))), **kwargs}, as_dict=as_dict)

    def split_DOS(self, on="species", only=None, exclude=None, clean=True, **kwargs):
        """
        Splits the density of states to the different contributions.

        Parameters
        --------
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}, or list of str
            the parameter to split along.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values that should not be plotted
        clean: boolean, optional
            whether the plot should be cleaned before drawing.
            If False, all the requests that come from the method will
            be drawn on top of what is already there.
        **kwargs:
            keyword arguments that go directly to each request.

            This is useful to add extra filters. For example:
            `plot.split_DOS(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.

        Examples
        -----------

        >>> plot = H.plot.pdos()
        >>>
        >>> # Split the DOS in n and l but show only the DOS from Au
        >>> # Also use "Au $ns" as a template for the name, where $n will
        >>> # be replaced by the value of n.
        >>> plot.split_DOS(on="n+l", species=["Au"], name="Au $ns")
        """
        req_field = self.get_input_field('requests')

        if req_field is None:
            return SislError("The requests field is not set.")
        assert hasattr(req_field, "_generate_queries"), "The requests input field does not implement the _generate_queries method"
        
        requests = req_field._generate_queries(
            on=on, only=only, exclude=exclude, query_gen=self._new_request, **kwargs
        )

        # If the user doesn't want to clean the plot, we will just add the requests to the existing ones
        if not clean:
            requests = [*self.get_input("requests", parsed=True), *requests]

        return self.update_inputs(requests=requests)