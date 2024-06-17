# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" Default graphene models """
from sisl.utils import PropertyDict

# Here we import the specific details that are exposed
from ._hamiltonian import *

__all__ = ["graphene"]

# Define the graphene model
graphene = PropertyDict()
graphene.hamiltonian = GrapheneHamiltonian()
graphene.H = graphene.hamiltonian
