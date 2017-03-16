from __future__ import print_function
from future.utils import iteritems
import numpy as np
import xarray as xr
import pandas as pd
import warnings
import dask.array as da_ar
from scipy.spatial import KDTree, cKDTree

def regrid_var(ds, new_x, new_y, cython, *args):
    """
    Parameters
    --------------
    ds : xarray.Dataset
        the dataset where the original coord can be found
    original_coord_name : str
        name of the original coordinate
    new_coord: numpy.array
        coordinates for the labels
        
    Returns
    -------------
    da : xarray.Dataset
        new dataset with labels added
    """
    latlon = args[0]
    time = args[1]
    var = args[2]
    meri = args[3]
    zon = args[4]
    assert new_x.ndim == 2
    assert new_y.ndim == 2
    
    original_coords = ds[latlon]
    latlon_list = list(original_coords.values)
    newyx = zip(new_y.ravel(), new_x.ravel())
    new_index = np.arange(len(newyx))
    if cython:
        tree = cKDTree(newyx)
    else:
        tree = KDTree(newyx)
    distance, index = tree.query(latlon_list)
    
    newlabel = np.empty(len(index), dtype=tuple)
    for i in range(len(index)):
        newlabel[i] = newyx[index[i]]
    
    da_index = xr.DataArray(index, 
                            dims=original_coords.dims,
                            coords=original_coords.coords)

    da_label = xr.DataArray(newlabel, 
                            dims=original_coords.dims,
                            coords=original_coords.coords)
    
    Nt = ds[var].shape[0]
    da_numpy = np.zeros((Nt, len(new_y[:, 0]), len(new_x[0, :])))
    for t in range(Nt):
        da = ds[var][t].reset_coords(names=time, 
                                     drop=True).to_dataset(name=var).copy()
        da.coords['index'] = da_index
        da.coords['label'] = da_label
        da_grouped = da.groupby('index').mean()
        da_reindexed = da_grouped.reindex({'index': new_index})
        da_panda_index = pd.MultiIndex.from_arrays([new_y.ravel(), 
                                                    new_x.ravel()], names=[meri, zon])
        da_reindexed.coords[latlon] = ('index', da_panda_index)
        da_unstacked = da_reindexed.swap_dims({'index': latlon}).unstack(latlon)
        da_numpy[t] = da_unstacked[var].values

    da = xr.DataArray(da_numpy, dims=[time, meri, zon], 
                      coords={time: range(Nt), meri: new_y[:, 0], 
                              zon: new_x[0, :]}).to_dataset(name=var)

    return da
