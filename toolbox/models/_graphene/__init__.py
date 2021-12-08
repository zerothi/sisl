""" Default graphene models """
from sisl.utils import PropertyDict

# Here we import the specific details that are exposed
from ._hamiltonian import *

__all__ = ['graphene']

# Define the graphene model
graphene = PropertyDict()
graphene.hamiltonian = GrapheneHamiltonian()
graphene.H = graphene.hamiltonian
