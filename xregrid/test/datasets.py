from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
import dask.array as da


x = np.linspace(0., 2*np.pi, 100)
y = np.linspace(-10., 10., 50)
xx, yy = np.meshgrid(x, y)
indexx = range(len(x))
indexy = range(len(y))
data = np.sin(x) + y[:, np.newaxis]


dataarrays = {
    'd2d' : xr.DataArray(data[np.newaxis, :, :], 
                         dims=['time', 'ny', 'nx'], 
                         coords={'time': range(1), 'ny': indexy, 'nx': indexx, 
                                 'yy': (('ny', 'nx'), yy), 
                         'xx': (('ny', 'nx'), xx)}
                        ).to_dataset(name='f'),
             }

@pytest.fixture(params=['d2d'])
def d2d(request):
    return dataarrays[request.param]

