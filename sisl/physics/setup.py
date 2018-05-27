from __future__ import print_function, division


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('physics', parent_package, top_path)

    config.add_extension('_matrix_k_dtype', sources=['_matrix_k_dtype.c'], include_dirs=['..'])
    config.add_extension('_matrix_k_nc_dtype', sources=['_matrix_k_nc_dtype.c'], include_dirs=['..'])
    config.add_extension('_matrix_k_so_dtype', sources=['_matrix_k_so_dtype.c'], include_dirs=['..'])
    config.add_extension('_matrix_diag_k_nc_dtype', sources=['_matrix_diag_k_nc_dtype.c'], include_dirs=['..'])
    config.add_extension('_matrix_k', sources=['_matrix_k.c'])

    config.add_extension('_matrix_dk_dtype', sources=['_matrix_dk_dtype.c'], include_dirs=['..'])
    config.add_extension('_matrix_dk', sources=['_matrix_dk.c'])

    config.add_data_dir('tests')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
