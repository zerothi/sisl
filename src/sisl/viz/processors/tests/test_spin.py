from sisl.viz.processors.spin import get_spin_options


def test_get_spin_options():

    # Unpolarized spin
    assert len(get_spin_options("unpolarized")) == 0

    # Polarized spin
    options = get_spin_options("polarized")
    assert len(options) == 4
    assert 0 in options
    assert 1 in options
    assert "total" in options
    assert "z" in options

    # Non colinear spin
    options = get_spin_options("noncolinear")
    assert len(options) == 4
    assert "total" in options
    assert "x" in options
    assert "y" in options
    assert "z" in options

    options = get_spin_options("noncolinear", only_if_polarized=True)
    assert len(options) == 0