from __future__ import print_function, division


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('sisl', parent_package, top_path)

    # Add Cython extensions
    config.add_extension('_math_small', sources=['_math_small.c'])
    config.add_extension('_indices', sources=['_indices.c'])
    config.add_extension('_supercell', sources=['_supercell.c'])
    config.add_extension('_sparse', sources=['_sparse.c'])

    config.add_subpackage('geom')
    config.add_subpackage('io')
    config.add_subpackage('physics')
    config.add_subpackage('linalg')
    config.add_subpackage('shape')
    config.add_subpackage('unit')
    config.add_subpackage('utils')

    config.add_data_dir('tests')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
