import xarray as xr

from sisl.viz.plotters.xarray import draw_xarray_xy


def test_empty_dataset():

    ds = xr.Dataset({"x": ("dim",  []), "y": ("dim", [])})

    drawings = draw_xarray_xy(ds, x="x", y="y")

    assert isinstance(drawings, list)
    assert len(drawings) == 0

def test_empty_dataarray():

    arr = xr.DataArray([], name="values", dims=['x'])

    drawings = draw_xarray_xy(arr, x="x")

    assert isinstance(drawings, list)
    assert len(drawings) == 0