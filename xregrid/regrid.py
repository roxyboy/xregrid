from __future__ import print_function
from future.utils import iteritems
import numpy as np
import xarray as xr
import pandas as pd
import warnings
import dask.array as da_ar
from scipy.spatial import KDTree, cKDTree

def stack_var(ds, varname, latname, lonname, maskroll, reset, *args):
    """
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
    latlon = latname + lonname
    if maskroll:
        assert len(args) > 3
        mask = args[0]
        nlon = args[1]
        T = T.where(mask).roll(nlon=nroll)
        lat = lat.where(mask).roll(nlon=nroll)
        lon = lon.where(mask).roll(nlon=nroll)
    
    time = args[2]
    nlat = args[3]
    nlon = args[4]
    
    if reset:
        assert len(args) == 7
        redundant_x = args[5]
        redundant_y = args[6]
        T = T.reset_coords([redundant_x, 
                     redundant_y], drop=True)
        lat = lat.reset_coords([redundant_x, 
                     redundant_y], drop=True)
        lon = lon.reset_coords([redundant_x, 
                     redundant_y], drop=True)
   
    T_stacked = T.stack(points=(nlat,nlon).copy()
    lat_stacked = lat.stack(points=(nlat,nlon).copy()
    lon_stacked = lon.stack(points=(nlat,lon)).copy()

    Nt = T_stacked.shape[0]
    for t in range(Nt):
        if t == 0:
            T_comp = np.empty((Nt, 
                               np.ma.masked_invalid(T_stacked[t].values).compressed().shape[0]))
        T_comp[t] = np.ma.masked_invalid(T_stacked[t].values).compressed()
    lat_comp = np.ma.masked_invalid(lat_stacked.values).compressed()
    lon_comp = np.ma.masked_invalid(lon_stacked.values).compressed()

    latlon = zip(lat_comp.ravel(), lon_comp.ravel())
    latlon_array = np.empty(len(latlon), dtype=object)
    for i in range(len(latlon)):
        latlon_array[i] = latlon[i]

    T_comp_xray = xr.DataArray(SST_comp, dims=[time, latlon], 
                               coords={time: range(Nt), 
                               latlon: latlon_array}).to_dataset(name=varname)
    return T_comp_xray

def regrid_var(ds, new_x, new_y, cython, *args):
    """
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
