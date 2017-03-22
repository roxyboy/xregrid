from __future__ import print_function
from future.utils import iteritems
import numpy as np
import xarray as xr
import pandas as pd
import profile
import warnings
import dask.array as da_ar
from scipy.spatial import KDTree, cKDTree

def stack_var(ds, varname, latname, lonname, maskroll, reset, *args):
    """
    Stacks the data in the spatial dimension and 
    returns a xarray.DataSet with the dimension of
    ['time', 'spatial'] for regrid_var().

    Parameters
    ------------
    ds : xarray.DataSet
    varname : string
    latname : string
    lonname : string
    maskroll : boolean
    reset : boolean
    args : list
    
    Returns
    ------------
    T_comp_xray : xarray.DataSet
    """
    T = ds[varname]
    lat = ds[latname]
    lon = ds[lonname]
    latlonname = latname + lonname
    
    if T.values.ndim != 3:
        raise RuntimeError('data must have at least the dimensions of time, lat and lon')
    time = args[0]
    nlat = args[1]
    nlon = args[2]

    if maskroll:
        if len(args) < 4:
            raise RuntimeError('args must have more than three elements')
        mask = args[3]
        nroll = args[4]
        T = T.where(mask).roll(nlon=nroll)
        lat = lat.where(mask).roll(nlon=nroll)
        lon = lon.where(mask).roll(nlon=nroll)
    
    if reset:
        if maskroll:
            if len(args) < 6:
                raise RuntimeError('args must have more than five elements')
            redundant_coords = []
            for i in range(5,len(args)):
                redundant_coords += [args[i],]
        else:
            if len(args) < 4:
                raise RuntimeError('args must have more than three elements')
            redundant_coords = []
            for i in range(3,len(args)):
                redundant_coords += [args[i],]
        T = T.reset_coords(redundant_coords, drop=True)
        lat = lat.reset_coords(redundant_coords, drop=True)
        lon = lon.reset_coords(redundant_coords, drop=True)
   
    T_stacked = T.stack(points=(nlat,nlon)).copy()
    lat_stacked = lat.stack(points=(nlat,nlon)).copy()
    lon_stacked = lon.stack(points=(nlat,nlon)).copy()

    Nt = T_stacked.shape[0]
    for t in range(Nt):
        if t == 0:
            T_comp = np.empty((Nt, 
                               np.ma.masked_invalid(T_stacked[t].values).compressed().shape[0]))
        T_comp[t] = np.ma.masked_invalid(T_stacked[t].values).compressed()
    lat_comp = np.ma.masked_invalid(lat_stacked.values).compressed()
    lon_comp = np.ma.masked_invalid(lon_stacked.values).compressed()

    latlon = list(zip(lat_comp.ravel(), lon_comp.ravel()))
    Nlatlon = len(latlon)
    latlon_array = np.empty(Nlatlon, dtype=object)
    for i in range(Nlatlon):
        latlon_array[i] = latlon[i]

    T_comp_xray = xr.DataArray(T_comp, dims=[time, latlonname], 
                               coords={time: range(Nt), 
                               latlonname: latlon_array}).to_dataset(name=varname)
    return T_comp_xray


def regrid_var(ds, new_x, new_y, cython, unstack, *args):
    """
    Regrids the data on orthogonal and non-orthogonal grids
    onto a new orthogonal grid using the scipy.spatial.KDTree
    algorithm.

    Parameters
    --------------
    ds : xarray.Dataset
        dataset where the original coord can be found
    new_x : numpy.array
        array of zonal coordinates to regrid the data on
    new_y : numpy.array
        array of meridional coordinates to regrid the data on
    cython : boolean
        criteria whether to use the cython package of KDTree
    unstack : boolean
        criteria whether to use indexing to unstack the data
    *args : list
        list of strings to name the coordinates and variable name
        
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
    
    dx = np.diff(new_x[0, :])[0]
    dy = np.diff(new_y[:, 0])[0]
    original_coords = ds[latlon]
   
    latlon_list = list(original_coords.values)
    newyx = list(zip(new_y.ravel(), new_x.ravel()))
    Nnewyx = len(newyx)
    new_index = np.arange(Nnewyx)
    if cython:
        tree = cKDTree(newyx)
    else:
        tree = KDTree(newyx)
    distance, index = tree.query(latlon_list)
    
    newlabel = np.empty(len(index), dtype=tuple)
    for i in range(len(index)):
        newlabel[i] = newyx[index[i]]
    
    # da_index = xr.DataArray(index, 
    #                         dims=original_coords.dims,
    #                         coords=original_coords.coords)

    # da_label = xr.DataArray(newlabel, 
    #                         dims=original_coords.dims,
    #                         coords=original_coords.coords)

    if ds[var].values.ndim != 2:
        raise RuntimeError('data should be stacked in the spatial dimension')
    Nt = ds[var].shape[0]
    da_numpy = np.zeros((Nt, len(new_y[:, 0]), len(new_x[0, :])))
    for t in range(Nt):
        da = ds[var][t].reset_coords(names=time, 
                                     drop=True).to_dataset(name=var).copy()        
        # da.coords['index'] = da_index
        da.coords['index'] = xr.Variable([latlon], index)
        # da.coords['label'] = da_label
        da_grouped = da.groupby('index').mean()
        da_reindexed = da_grouped.reindex({'index': new_index})
        if unstack:
            da_panda_index = pd.MultiIndex.from_arrays([new_y.ravel(), 
                                                       new_x.ravel()], 
                                                       names=[meri, zon])
            da_reindexed.coords[latlon] = ('index', 
                                           da_panda_index)
            da_unstacked = da_reindexed.swap_dims({'index': latlon}).unstack(latlon)[var].values
        else:
            da_unstacked = da_reindexed[var].values.reshape((len(new_y[:, 0]), 
                                                             len(new_x[0, :])))
        da_numpy[t] = da_unstacked

    da = xr.DataArray(da_numpy, dims=[time, meri, zon], 
                      coords={time: range(Nt), meri: new_y[:, 0], 
                              zon: new_x[0, :]}).to_dataset(name=var)

    return da
