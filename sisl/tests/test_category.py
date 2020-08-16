from sisl.geom import AtomCategory

def test_geom_category_kw_and_call():
    # Check that categories can be built indistinctively using the kw builder
    # or directly calling them.
    cat1 = AtomCategory.kw(odd={})
    cat2 = AtomCategory(odd={})
    assert cat1 == cat2
