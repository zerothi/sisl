# [sisl](https://zerothi.github.io/sisl/index.html) #

[![Install sisl using PyPI](https://badge.fury.io/py/sisl.svg)](https://pypi.org/project/sisl)
[![Install sisl using conda](https://anaconda.org/conda-forge/sisl/badges/version.svg)](https://anaconda.org/conda-forge/sisl)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI for citation](https://zenodo.org/badge/doi/10.5281/zenodo.597181.svg)](https://doi.org/10.5281/zenodo.597181)
<!--- [![Documentation on RTD](https://readthedocs.org/projects/docs/badge/?version=latest)](http://sisl.readthedocs.io/en/latest/) -->
[![Join discussion on Discord](https://img.shields.io/discord/742636379871379577.svg?label=&logo=discord&logoColor=ffffff&color=green&labelColor=red)](https://discord.gg/5XnFXFdkv2)
[![Build Status](https://github.com/zerothi/sisl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/zerothi/sisl/actions/workflows/test.yaml)
[![Checkout sisl code coverage](https://codecov.io/gh/zerothi/sisl/branch/main/graph/badge.svg)](https://codecov.io/gh/zerothi/sisl)
[![Python versions](https://img.shields.io/pypi/pyversions/sisl.svg)](https://pypi.org/project/sisl/)

Copyright sisl developers 2025
Copyright Nick R. Papior 2015

sisl is an *open-source easy-to-use density functional theory API framework* to post-analyse density functional theory codes output
as well as providing tight-binding calculation capabilities.
It couples to a wide range of density functional theory codes and has a high connection with LCAO codes such as [Siesta][siesta]. The tight-binding matrices can be used in non-equilibrium Green function calculations with [TBtrans][tbtrans] as a backend. API for creating publication ready graphs and images.

## Features ##

- *Simple command-line interface*: To extract or quickly plot calculation output
- *Tight-binding API*: Easily create (non-) orthogonal tight-binding matrices and do electronic structure analysis
- *Manipulation of sparse matrices*: Extract, replace, append matrices and sub-matrices to one another
- *Post analyse DFT simulations*: Post-process LCAO Hamiltonians by doing (projected) density of states, inverse participation ratio and many more
- *Post analyse NEGF simulations*: Effectively create and post-analyse NEGF output from [TBtrans][tbtrans]
- *Real-space grid analysis*: Perform mathematical operations on DFT real-space grid outputs, spin-density differences and wavefunction plots
- *Conversion of geometries and real-space grid*: Easy conversion of geometry files and real-space grid file formats (cube, xsf, etc.)
- *User contributed toolboxes*: Users may contribute toolboxes for sharing methodologies
- *Interoperability with other codes*: [ASE][ase] and [pymatgen]
- *and many more features*

## Tutorials and examples ##

The easiest way to get started is to follow the tutorials [here](https://zerothi.github.io/sisl/tutorials.html) and the workshop material for [TranSiesta][siesta] [here][workshop].


## Documentation ##

Please find documentation here:

- [Documentation](https://zerothi.github.io/sisl/index.html)
- [API documentation](https://zerothi.github.io/sisl/api/index.html)
- [Installation](https://zerothi.github.io/sisl/installation.html)


## Community support ##

There are different places for getting information on using sisl, here is a short list
of places to search/ask for answers:

- Ask questions on the [Discord page][sisl@discord]
- Ask questions on the Github [issue page][sisl@issue]
- [Documentation][sisl@api], recommended reference page
- [Workshop][workshop] examples showing different uses

If sisl was used to produce scientific contributions, please use this [DOI][doi] for citation.
We recommend to specify the version of sisl in combination of this citation:

    @software{zerothi_sisl,
      author = {Papior, Nick},
      title  = {sisl: v<fill-version>},
      year   = {2025},
      doi    = {10.5281/zenodo.597181},
      url    = {https://doi.org/10.5281/zenodo.597181}
    }

To get the BibTeX entry easily you may issue the following command:

    sdata --cite

which fills in the version number.

## Contributing ##

Kindly read our [Contributing Guide](CONTRIBUTING.md) to learn and understand about our development process, how to propose bug fixes and improvements, and how to build and test your changes to sisl.

## Contributors ##
<a href="https://github.com/zerothi/sisl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zerothi/sisl" />
</a>

## License
sisl © 2015, Released under the Mozilla Public License v2.0.


<!---
Links to external and internal sites.
-->
[sisl@git]: https://github.com/zerothi/sisl
[sisl@api]: https://zerothi.github.io/sisl
[sisl@discord]: https://discord.gg/5XnFXFdkv2
[sisl@issue]: https://github.com/zerothi/sisl/issues
[sisl@pr]: https://github.com/zerothi/sisl/pulls
[siesta]: https://gitlab.com/siesta-project/siesta
[tbtrans]: https://gitlab.com/siesta-project/siesta
[workshop]: https://github.com/zerothi/ts-tbt-sisl-tutorial
[doi]: https://doi.org/10.5281/zenodo.597181
[mpl]: https://www.mozilla.org/en-US/MPL/2.0/
[ase]: https://wiki.fysik.dtu.dk/ase/
[pymatgen]: https://pymatgen.org/

<!---
Local variables for emacs to turn on flyspell-mode
% Local Variables:
%   mode: flyspell
%   tab-width: 4
%   indent-tabs-mode: nil
% End:
-->
