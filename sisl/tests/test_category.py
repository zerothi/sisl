from sisl.geom import AtomCategory
from sisl.geom.category import AtomOdd

def test_category_kw():

    # Check that we can get categories that don't accept any argument
    assert AtomCategory.kw(odd=True) == AtomOdd()

    # Check also that using this functionality with False returns the not category
    assert AtomCategory.kw(odd=False) == ~AtomOdd()