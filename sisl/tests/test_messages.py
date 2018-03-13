from __future__ import print_function, division

import pytest

import warnings as w

import sisl.messages as sm

pytestmark = pytest.mark.messages


def test_deprecate():
    with pytest.warns(sm.SislDeprecation):
        sm.deprecate('Deprecation warning')


def test_deprecation():
    with pytest.warns(sm.SislDeprecation):
        w.warn(sm.SislDeprecation('Deprecation warning'))


def test_warn_method():
    with pytest.warns(sm.SislWarning):
        sm.warn('Warning')


def test_warn_specific():
    with pytest.warns(sm.SislWarning):
        sm.warn(sm.SislWarning('Warning'))


def test_warn_category():
    with pytest.warns(sm.SislWarning):
        sm.warn('Warning', sm.SislWarning)


def test_info_method():
    with pytest.warns(sm.SislInfo):
        sm.info('Information')


def test_info_specific():
    with pytest.warns(sm.SislInfo):
        sm.info(sm.SislInfo('Info'))


def test_info_category():
    with pytest.warns(sm.SislInfo):
        sm.info('Information', sm.SislInfo)


@pytest.mark.xfail(raises=sm.SislError)
def test_error():
    raise sm.SislError('This is an error')


@pytest.mark.xfail(raises=sm.SislException)
def test_exception():
    raise sm.SislException('This is an error')


def test_tqdm_eta_true():
    eta = sm.tqdm_eta(2, 'Hello', 'unit', True)
    eta.update()
    eta.update()
    eta.close()


def test_tqdm_false():
    eta = sm.tqdm_eta(2, 'Hello', 'unit', False)
    eta.update()
    eta.update()
    eta.close()
