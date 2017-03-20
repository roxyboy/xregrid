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
        xg.stack_var(d2d, varname, xx, yy, True, False, *args)
