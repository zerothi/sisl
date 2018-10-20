from __future__ import print_function, division


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('physics', parent_package, top_path)

    # Bloch expansion
    config.add_extension('_bloch', sources=['_bloch.c'])

    # Regular M(k)
    config.add_extension('_phase', sources=['_phase.c'], include_dirs=['..'])
    config.add_extension('_matrix_phase', sources=['_matrix_phase.c'], include_dirs=['..'])
    config.add_extension('_matrix_phase_nc', sources=['_matrix_phase_nc.c'], include_dirs=['..'])
    config.add_extension('_matrix_phase_so', sources=['_matrix_phase_so.c'], include_dirs=['..'])
    config.add_extension('_matrix_phase_nc_diag', sources=['_matrix_phase_nc_diag.c'], include_dirs=['..'])
    config.add_extension('_matrix_k', sources=['_matrix_k.c'])

    # The factor matrix takes a [:, 3] array factor and returns a tuple of 3 quantities
    config.add_extension('_matrix_phase3', sources=['_matrix_phase3.c'], include_dirs=['..'])
    config.add_extension('_matrix_dk', sources=['_matrix_dk.c'])
    config.add_extension('_matrix_ddk', sources=['_matrix_ddk.c'])

    config.add_data_dir('tests')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
