from ..sile import add_sile
from .tbt import tbtncSileTBtrans, tbtavncSileTBtrans, Ry2K, Ry2eV


__all__ = ['phtncSileTBtrans', 'phtavncSileTBtrans']


class phtncSileTBtrans(tbtncSileTBtrans):
    """ PHtrans file object """
    _trans_type = 'PHT'

    def phonon_temperature(self, elec):
        """ Phonon bath temperature [Kelvin] """
        return self._value('kT', self._elec(elec))[0] * Ry2K

    def kT(self, elec):
        """ Phonon bath temperature [eV] """
        return self._value('kT', self._elec(elec))[0] * Ry2eV


class phtavncSileTBtrans(tbtavncSileTBtrans):
    """ PHtrans file object """
    _trans_type = 'PHT'


# Clean up methods
for _name in ['chemical_potential', 'electron_temperature',
              'shot_noise', 'noise_power',
              'current', 'current_parameter']:
    setattr(phtncSileTBtrans, _name, None)
    setattr(phtavncSileTBtrans, _name, None)

    
add_sile('PHT.nc', phtncSileTBtrans)
add_sile('PHT.AV.nc', phtavncSileTBtrans)
