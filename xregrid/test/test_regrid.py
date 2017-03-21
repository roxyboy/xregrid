from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
import xregrid as xg
import pandas as pd
import os


from . datasets import d2d

def test_length_args(d2d):
    varname = list(d2d.data_vars.keys())[0]
    coords = list(d2d.coords.keys())
    nx = coords[0]
    ny = coords[1]
    xx = coords[2]
    yy = coords[3]
    time = coords[4]
    args = [time, ny, nx]
   
    with pytest.raises(RuntimeError):
        xg.stack_var(d2d, varname, yy, xx, True, False, *args)

    


def test_stack_regrid(d2d):

    varname = list(d2d.data_vars.keys())[0]
    args = ['time', 'ny', 'nx']
   
    d2d_stacked = xg.stack_var(d2d, varname, 'yy', 'xx', False, False, *args)
    expected_stack = np.array([-10., -9.93657608, -9.87340755, -9.81074876 ,-9.74885201,
                               -9.68796655, -9.62833754, -9.57020509, -9.51380326, -9.45935918,
                               -9.40709207, -9.35721239, -9.30992099, -9.26540829, -9.22385354,
                               -9.18542405, -9.15027457, -9.11854664, -9.090368, -9.06585214]
                             ) 
    np.testing.assert_allclose(d2d_stacked[varname][0].values[:20],
                                  expected_stack)
    
    x = np.linspace(0., 2*np.pi, 100)
    y = np.linspace(-10., 10., 50)
    new_x = np.linspace(x.min(), x.max(), 20)
    new_y = np.linspace(y.min(), y.max(), 20)
    new_xx, new_yy = np.meshgrid(new_x, new_y)
    
    args = ['yyxx', 'time', 'f', 'y', 'x']
    d2d_regrid = xg.regrid_var(d2d_stacked, new_xx, new_yy, True, False, *args)
    expected_regrid = np.array([[-9.73257958, -9.48514036, -9.1813844, -8.94961173, -8.82801679,
                                 -8.80307434, -8.87727508, -9.06563446, -9.33953548, -9.63855267,
                                 -9.95328406, -10.25230126, -10.52620227, -10.71456166, -10.78876239,
                                 -10.76381994, -10.64222501, -10.41045234, -10.10669638, -9.85925716],
                                [-8.91625305, -8.66881383, -8.36505787, -8.1332852, -8.01169026,
                                 -7.98674781, -8.06094855, -8.24930793, -8.52320895, -8.82222614,
                                 -9.13695753, -9.43597473, -9.70987574, -9.89823513, -9.97243586,
                                 -9.94749341, -9.82589848, -9.59412581, -9.29036985, -9.04293063]]
                              )
    np.testing.assert_allclose(d2d_regrid[varname][0].values[:2],
                                  expected_regrid)
    
