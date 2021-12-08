# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sisl._help import isiterable

from .category import CategoryInput

from .basic import (
    TextInput, IntegerInput, FloatInput,
    DictInput, CreatableDictInput, OptionsInput,
    RangeInput, RangeSliderInput
)


class AtomCategoryInput(CategoryInput):
    pass


class AtomIndexCatInput(AtomCategoryInput, DictInput):

    def __init__(self, *args, fields=(), **kwargs):
        fields = [
            OptionsInput(key="in", name="Indices",
                params={
                    "placeholder": "Select indices...",
                    "options": [],
                    "isMulti": True,
                    "isClearable": True,
                    "isSearchable": True,
                }
            ),

            *fields
        ]

        super().__init__(*args, fields=fields, **kwargs)

    def parse(self, val):
        if isinstance(val, int):
            return val
        elif isiterable(val):
            return val
        else:
            return super().parse(val)

    def update_options(self, geom):
        self.get_param("in").modify("inputField.params.options",
               [{"label": f"{at} ({geom.atoms[at].symbol})", "value": at}
                for at in geom])


class AtomFracCoordsCatInput(AtomCategoryInput, RangeSliderInput):

    _default = {
        "default": [0, 1],
        "params": {"min": 0, "max": 1, "step": 0.01}
    }


class AtomCoordsCatInput(AtomCategoryInput, RangeInput):
    pass


class AtomZCatInput(AtomCategoryInput, IntegerInput):

    _default = {
        "params": {"min": 0}
    }


class AtomNeighboursCatInput(AtomCategoryInput, DictInput):

    def __init__(self, *args, fields=(), **kwargs):
        fields = [
            RangeInput(key="range", name=""),

            FloatInput(key="R", name="R"),

            TextInput(key="neigh_tag", name="Neighbour tag", default=None),
        ]

        super().__init__(*args, fields=fields, **kwargs)

    def parse(self, val):

        if isinstance(val, dict):
            val = {**val}
            if "range" in val:
                val["min"], val["max"] = val.pop("range")
            if "neigh_tag" in val:
                val["neighbour"] = {"tag": val.pop("neigh_tag")}

        return val


class AtomTagCatInput(AtomCategoryInput, TextInput):
    pass


class AtomSeqCatInput(AtomCategoryInput, TextInput):
    pass


class AtomSelect(CreatableDictInput):

    _default = {}

    def __init__(self, *args, fields=(), **kwargs):
        fields = [
            AtomIndexCatInput(key="index", name="Indices"),

            *[AtomFracCoordsCatInput(key=f"f{ax}", name=f"Fractional {ax.upper()}", default=[0, 1])
                for ax in "xyz"],

            *[AtomCoordsCatInput(key=ax, name=f"{ax.upper()} coordinate")
                for ax in "xyz"],

            AtomZCatInput(key="Z", name="Atomic number"),

            AtomNeighboursCatInput(key="neighbours", name="Neighbours"),

            AtomTagCatInput(key="tag", name="Atom tag"),

            AtomSeqCatInput(key="seq", name="Index sequence")
        ]

        super().__init__(*args, fields=fields, **kwargs)

    def update_options(self, geom):

        self.get_param("index").update_options(geom)

        return self

    def parse(self, val):
        if isinstance(val, dict):
            val = super().parse(val)

        return val


class SpeciesSelect(OptionsInput):

    _default = {
        "default": None,
        "params": {
            "placeholder": "Select species...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        self.modify("inputField.params.options",
                    [{"label": unique_at.symbol, "value": unique_at.symbol}
                     for unique_at in geom.atoms.atom])

        return self


class AtomicQuery(DictInput):

    _fields = {
        "atoms": {"field": AtomSelect, "name": "Atoms"}
    }
