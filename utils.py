import traceback
import warnings

import numpy as np
import xarray as xr

#import variable_defs

nmols_to_PgCyr = 1e-9 * 86400. * 365. * 12e-15


def global_mean(ds, ds_grid, compute_vars, normalize=True, include_ms=False):
    """
    Compute the global mean on a POP dataset. 
    Return computed quantity in conventional units.
    """

    other_vars = list(set(ds.variables) - set(compute_vars))
    print(other_vars)
    if include_ms: # marginal seas!
        surface_mask = ds_grid.TAREA.where(ds_grid.KMT > 0).fillna(0.)
    else:
        surface_mask = ds_grid.TAREA.where(ds_grid.REGION_MASK > 0).fillna(0.)        
    
    masked_area = {
        v: surface_mask.where(ds[v].notnull()).fillna(0.) 
        for v in compute_vars
    }
    print('going to compute global totals')
    with xr.set_options(keep_attrs=True):
        
        #dso = xr.Dataset({
        #    v: (ds[v] * masked_area[v]).sum(['nlat', 'nlon'])
        #    for v in compute_vars
        #})
        
        dso = xr.Dataset()
        
        for v in compute_vars:
            print(v)
            dso[v] = (ds[v] * masked_area[v]).sum(['nlat', 'nlon'])
        
        
        #if normalize:
        #    print('normalizing')
        #    dso = xr.Dataset({
        #        v: dso[v] / masked_area[v].sum(['nlat', 'nlon'])
        #        for v in compute_vars
        #    })            
                
    return dso
    
def adjust_pop_grid(tlon,tlat,field):
    nj = tlon.shape[0]
    ni = tlon.shape[1]
    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = np.where(np.greater_equal(tlon,min(tlon[:,0])),tlon-360.,tlon)
    lon  = np.concatenate((tlon,tlon+360.),1)
    lon = lon[:,xL:xR]

    if ni == 320:
        lon[367:-3,0] = lon[367:-3,0]+360.
    lon = lon - 360.
    lon = np.hstack((lon,lon[:,0:1]+360.))
    if ni == 320:
        lon[367:,-1] = lon[367:,-1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:,0] = lon[:,0]-1e-8
    
    #-- periodicity
    lat  = np.concatenate((tlat,tlat),1)
    lat = lat[:,xL:xR]
    lat = np.hstack((lat,lat[:,0:1]))

    field = np.ma.concatenate((field,field),1)
    field = field[:,xL:xR]
    field = np.ma.hstack((field,field[:,0:1]))
    return lon,lat,field


def normal_lons(lons):

    lons_norm=np.full((len(lons.nlat), len(lons.nlon)), np.nan)

    lons_norm_firstpart = lons.where(lons<=180.)
    lons_norm_secpart = lons.where(lons>180.) - 360.

    lons_norm_firstpart = np.asarray(lons_norm_firstpart)
    lons_norm_secpart = np.asarray(lons_norm_secpart)

    lons_norm[~np.isnan(lons_norm_firstpart)] = lons_norm_firstpart[~np.isnan(lons_norm_firstpart)]
    lons_norm[~np.isnan(lons_norm_secpart)] = lons_norm_secpart[~np.isnan(lons_norm_secpart)]

    lons_norm=xr.DataArray(lons_norm)
    lons_norm=lons_norm.rename({'dim_0':'nlat'})
    lons_norm=lons_norm.rename({'dim_1':'nlon'})
    
    return(lons_norm)


def calc_area(nx,ny,lats):

    area = xr.DataArray(np.zeros([ny,nx]), dims=('lat','lon'))

    j=0

    for lat in lats:

        pi     =    3.14159265359
        radius = 6378.137

        deg2rad = pi / 180.0

        resolution_lat =1./12. #res in degrees
        resolution_lon =1./12. #res in degrees

        elevation = deg2rad * (lat + (resolution_lat / 2.0))

        deltalat = deg2rad * resolution_lon
        deltalon = deg2rad * resolution_lat

        area[j,:] = (2.0*radius**2*deltalon*np.cos(elevation)*np.sin((deltalat/2.0)))

        j = j + 1

    return(area)