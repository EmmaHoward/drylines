import xarray as xr
import numpy as np
import os
import sys




def integral(varlist, year, mon, exppath,exp, x0, y0, radius=5):
    """
    compute spatial integral
    """
    data = {}
    r = 6371000
    for var in varlist:
        data[var] = xr.open_mfdataset(os.path.join(exppath,var,f"{year}",f"{var}*_{year}{mon:02d}*"))[var]
        data[var] = data[var].sel(longitude=slice(x0-radius*1.1,x0+radius*1.1),latitude=slice(y0+radius*1.1,y0-radius*1.1))
    data = xr.Dataset(data)
    lon = data.longitude
    lat = data.latitude
    dxdy = 2*np.pi*r/360*np.abs(lon.roll(longitude=1)-lon) \
        * (2*np.pi*r/360*np.abs(lat.roll(latitude=1)-lat)*np.cos(lat*np.pi/180))

    mask = ((data.longitude-x0)**2+(data.latitude-y0)**2)<radius**2
    integrals = {}
    for var in varlist:
        integrals[var] = (data[var]*dxdy).where(mask).sum(['latitude','longitude']).rename(var)
        integrals[var].attrs = data[var].attrs
    integrals = xr.Dataset(integrals)
 
    x0_str = f"{np.abs(x0)}"+["E","W"][x0<0]
    y0_str = f"{np.abs(y0)}"+["N","S"][y0<0]

    integrals.to_netcdf(f"/g/data/tp28/dev/eh6215/qflux/{exp}/integrals_{year}{mon:02d}_{x0_str}_{y0_str}.nc", encoding = {var:{'zlib':True} for var in varlist})


def moisture_flux(year, mon, exppath,exp, x0, y0, radius=5, delta = 1):
    """
    compute moisture flux into domain
    also returns tcw along boundary
    """
    # load data
    qu_in = xr.open_mfdataset(os.path.join(exppath,'viwve',f"{year}",f"viwve*_{year}{mon:02d}*"))['p71.162']
    qv_in = xr.open_mfdataset(os.path.join(exppath,'viwvn',f"{year}",f"viwvn*_{year}{mon:02d}*"))['p72.162']
    q_in = xr.open_mfdataset(os.path.join(exppath,'tcw',f"{year}",f"tcw*_{year}{mon:02d}*"))['tcw']

    angles=np.arange(0,360,delta)/180*np.pi

    lat = xr.DataArray(y0 + radius*np.cos(angles), dims='bearing')
    lon = xr.DataArray(x0 + radius*np.sin(angles), dims='bearing')
    
    qu = qu_in.sel(latitude=lat, longitude=lon, method='nearest')  
    qv = qv_in.sel(latitude=lat, longitude=lon, method='nearest')  
    q = q_in.sel(latitude=lat, longitude=lon, method='nearest')    

    qu['bearing'] = angles*180/np.pi
    qu['bearing'].attrs['units'] = 'degrees'
    
    qv['bearing'] = angles*180/np.pi
    qv['bearing'].attrs['units'] = 'degrees'

    q['bearing'] = angles*180/np.pi
    q['bearing'].attrs['units'] = 'degrees'
    
    flux = (qv*np.cos(angles)+qu*np.sin(angles)).rename("moisture_flux",longitude='lon',latitude='lat')
    
    # grid-spacing:  np.sqrt((2*np.pi*r/360*(flux.lat - flux.lat.shift(z=1,)))**2+(2*np.pi*r/360*np.cos(flux.lat*np.pi/180)*(flux.lon - flux.lon.shift(z=1)))**2)
        
    flux.attrs['x0'] = x0
    flux.attrs['y0'] = y0
    flux.attrs['radius'] = radius
    flux.attrs['units'] = "kg m-1 s-1"
    
    q.attrs['x0'] = x0
    q.attrs['y0'] = y0
    q.attrs['radius'] = radius
    q.attrs['units'] = "kg m-2"
    
    ds = xr.Dataset({'moisture_flux':flux,'tcw':q})
    
    x0_str = f"{np.abs(x0)}"+["E","W"][x0<0]
    y0_str = f"{np.abs(y0)}"+["N","S"][y0<0]
    ds.to_netcdf(f"/g/data/tp28/dev/eh6215/qflux/{exp}/qflux_{year}{mon:02d}_{x0_str}_{y0_str}.nc", \
                   encoding = {'moisture_flux':{'zlib':True}, \
                               'tcw'          :{'zlib':True}})
    
    
if __name__ == "__main__":
  for mon in range(1,13):
    year = int(os.environ['year'])
    era5_path = "/g/data/rt52/era5/single-levels/reanalysis/"
    varlist = ['mtpr', 'tcw', 'e']    # variable names on NCI: mean total precip rate, total column water, evaporation
    moisture_flux(year,mon,era5_path,'ERA5',24,-18,radius=8)
    integral(varlist,year,mon,era5_path,"ERA5",24,-18,radius=8)
    #moisture_flux(year,mon,era5_path,'ERA5',133,-18)
    #integral(varlist,year,mon,era5_path,"ERA5",133,-18)
    