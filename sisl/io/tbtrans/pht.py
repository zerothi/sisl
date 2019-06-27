from __future__ import print_function, division

from ..sile import add_sile
from .tbt import tbtncSileTBtrans, tbtavncSileTBtrans, Ry2K, Ry2eV


__all__ = ['phtncSilePHtrans', 'phtavncSilePHtrans']


class phtncSilePHtrans(tbtncSileTBtrans):
    """ PHtrans file object """
    _trans_type = 'PHT'
    _E2eV = Ry2eV ** 2

    def phonon_temperature(self, elec):
        """ Phonon bath temperature [Kelvin] """
        return self._value('kT', self._elec(elec))[0] * Ry2K

    def kT(self, elec):
        """ Phonon bath temperature [eV] """
        return self._value('kT', self._elec(elec))[0] * Ry2eV


class phtavncSilePHtrans(tbtavncSileTBtrans):
    """ PHtrans file object """
    _trans_type = 'PHT'
    _E2eV = Ry2eV ** 2

    def phonon_temperature(self, elec):
        """ Phonon bath temperature [Kelvin] """
        return self._value('kT', self._elec(elec))[0] * Ry2K

    def kT(self, elec):
        """ Phonon bath temperature [eV] """
        return self._value('kT', self._elec(elec))[0] * Ry2eV


for _name in ['chemical_potential', 'electron_temperature', 'kT',
              'current', 'current_parameter',
              'shot_noise', 'noise_power']:
    setattr(phtncSilePHtrans, _name, None)
    setattr(phtavncSilePHtrans, _name, None)


add_sile('PHT.nc', phtncSilePHtrans)
add_sile('PHT.AV.nc', phtavncSilePHtrans)
