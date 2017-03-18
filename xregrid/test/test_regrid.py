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
    varname = d2d.data_vars.keys()[0]
    nx = d2d.coords.keys()[0]
    ny = d2d.coords.keys()[1]
    xx = d2d.coords.keys()[2]
    yy = d2d.coords.keys()[3]
    time = d2d.coords.keys()[4]
    args = [time, ny, nx]
   
    with pytest.raises(RuntimeError):
        xg.stack_var(d2d, xx, yy, True, False, *args)
