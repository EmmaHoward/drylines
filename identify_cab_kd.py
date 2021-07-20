
import numpy as np
import iris
import matplotlib.pyplot as plt
from drylines import find_edge,find_ridge,dxdy,ddx,ddy
from cartopy import crs as ccrs
from scipy.ndimage import gaussian_filter
import cmocean

#    Read Data using iris

data = iris.load("era5_nearsurface_2016.nc")
dp = data.extract("2 metre dewpoint temperature")[0]
#t = data.extract("2 metre temperature")[0]
u = data.extract("100 metre U wind component")[0]
v = data.extract("100 metre V wind component")[0]
sp = data.extract("surface_air_pressure")[0]
times = u.coord('time').units.num2date(u.coord('time').points)
lat = u.coord('latitude').points
lon = u.coord('longitude').points
lon2,lat2=np.meshgrid(lon,lat)

# Create masks for acceptable CAB and KD locations

#landmask = maskoceans(lon2,lat2,np.ones(lon2.shape))
mask_cab = (lon2<=30)*(lat2<=-5)*(lat2>=-18)#*(1-landmask.mask)
mask_tkd = (lon2<=30)*(lat2<=-12)#*(1-landmask.mask)

# Calculate humidity from dewpoint
# see eoas.ubc.ac/books/Practical_Meteorology/prmet102/Ch04-watervapor-v102b.pdf 
# for formulae

#r=np.exp(1.0/(1.844e-4)*(1.0/t.data-1.0/dp.data))*100   # Relative Humidity
e=611.3*np.exp(1.0/(1.844e-4)*(1.0/273.15-1.0/dp.data))  # Water Vapour Pressure
q=0.622*e/(sp.data-e*(1-0.622))[:,::-1]                          # Specific Humidity


# Calculate divergence from smoothed U100 and V100
us,vs=[],[]
for i,t in enumerate(times):
  us.append(gaussian_filter(u[i].data,2))
  vs.append(gaussian_filter(v[i].data,2))

us,vs=np.array(us),np.array(vs)
dx,dy = dxdy(lon,lat)
div = ddx(us,dx)+ddy(vs,dy)

# Functions for plotting results
def q_fig(d,tt,points,filt):
#    m=Basemap(llcrnrlat=-40,llcrnrlon=10,urcrnrlat=0,urcrnrlon=41.5)
#    m.drawcoastlines()
#    m.drawcountries()
    plt.title(tt)
    plt.contourf(lon,lat,d,np.linspace(0,0.018,20),cmap=cmocean.cm.matter)
    plt.pcolor(lon,lat,np.ma.masked_array(points,1-filt),vmin=-np.pi,vmax=np.pi,cmap=cmocean.cm.phase)
 
def u_fig(d,tt,points,filt):
#    m=Basemap(llcrnrlat=-40,llcrnrlon=10,urcrnrlat=0,urcrnrlon=41.5)
#    m.drawcoastlines()
#    m.drawcountries()
    plt.title(tt)
    plt.contourf(lon,lat,d,np.linspace(-2e-4,2e-4,20),cmap='RdBu')
    plt.pcolor(lon,lat,np.ma.masked_array(points,1-filt),vmin=-np.pi,vmax=np.pi,cmap=cmocean.cm.phase)
 
#Find dryline CABs. See drylines.py for a description of the inputs
# set plotfreq to 1 in order to see a rough plot of q and detected locations for each day
cab_q=find_edge(q,lon,lat,2,theta_min=-np.pi/4,theta_max=np.pi/6,mag_min=0.003,minlen=10,spatial_mask=mask_cab,relative="Grid Cell",output='sparse',plotfreq=0,times=None,makefig=q_fig)

#Find dryline KDs. See drylines.py for a description of the inputs
# set plotfreq to 1 in order to see a rough plot of q and detected locations for each day
kd_q=find_edge(q,lon,lat,2,theta_max=np.pi/2,theta_min=np.pi/6,mag_min=0.003,minlen=10,spatial_mask=mask_tkd,relative="Grid Cell",output='sparse',plotfreq=0,times=None,makefig=q_fig)

#Find convergence-line CABs. See drylines.py for a description of the inputs
# set plotfreq to 1 in order to see a rough plot of div and detected locations for each day
cab_u=find_ridge(div,lon,lat,1,theta_min=-np.pi/4,theta_max=np.pi/6,mag_min=2e-5,minlen=10,spatial_mask=mask_cab,output='sparse',plotfreq=0,times=None,makefig=u_fig,sign=-1)

#Find convergence-line KDs. See drylines.py for a description of the inputs
# set plotfreq to 1 in order to see a rough plot of div and detected locations for each day
kd_u=find_ridge(div,lon,lat,1,theta_max=np.pi/2,theta_min=np.pi/6,mag_min=2e-5,minlen=10,spatial_mask=mask_tkd,output='sparse',plotfreq=0,times=None,makefig=u_fig,sign=-1)


# Plot spatial frequency heatmaps
ax=plt.subplot(221,projection=ccrs.PlateCarree())
ax.coastlines()
plt.pcolor(lon,lat,cab_q.sum(axis=0),vmin=0,vmax=10,cmap=cmocean.cm.rain)
plt.title("Dryline CAB")

plt.subplot(222,projection=ccrs.PlateCarree())
ax.coastlines()
plt.pcolor(lon,lat,kd_q.sum(axis=0),vmin=0,vmax=10,cmap=cmocean.cm.rain)
plt.title("Dryline KD")

plt.subplot(223,projection=ccrs.PlateCarree())
ax.coastlines()
plt.pcolor(lon,lat,cab_u.sum(axis=0),vmin=0,vmax=10,cmap=cmocean.cm.rain)
plt.title("Convergence Line CAB")

ax=plt.subplot(224,projection=ccrs.PlateCarree())
ax.coastlines()
plt.pcolor(lon,lat,kd_u.sum(axis=0),vmin=0,vmax=10,cmap=cmocean.cm.rain)
plt.title("Convergence Line KD")

plt.suptitle("Frequencies in September 2016")
plt.show()


