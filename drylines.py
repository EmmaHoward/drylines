
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,label
from canny_mod import canny,canny_div
import cmocean


def dxdy(lon,lat):
  # calculates grid spacing in metres
  if len(lon.shape)==1:
    lon2,lat2=np.meshgrid(lon,lat)
  if len(lon.shape)==2:
    lon2,lat2=lon,lat
  r = 6371000
  dx = r*np.cos(lat2*np.pi/180.0)*np.gradient(lon2)[1]*np.pi/180.0
  dy = np.gradient(lat2)[0]*r*np.pi/180.0
  return dx,dy

def get_dxdy(lon,lat,relative='Grid Cell'):
  # calculates grid spacing in various units
  assert(relative in ["Grid Cell","Weighted Grid Cell","Degrees","metres","meters"])
  if relative=="Grid Cell":
    dx,dy = 1.0,1.0
  elif relative=="Weighted Grid Cell":
    dx,dy=dxdy(lon,lat)
    mean = (dx.abs().mean()+dy.abs().mean())/2.0
    dx/= mean
    dy/= mean
  elif relative=="Degrees":
    assert len(lon.shape)==1 and len(lat.shape)==1
    dx,dy = np.zeros(lon.shape),np.zeros(lat.shape)
    dx[1:-1]=(lon[2:]-lon[:-2])/2.0
    dy[1:-1]=(lat[2:]-lat[:-2])/2.0
    dx[0],dx[-1] = lon[1]-lon[0],lon[-1]-lon[-2] 
    dy[0],dy[-1] = lat[1]-lat[0],lat[-1]-lat[-2]
    dx,dy=np.meshgrid(dx,dy) 
  elif relative in ["metres","meters"]:
    dx,dy=dxdy(lon,lat)
  return(dx,dy)

def ddy(var,dy):
#
# Calculate meridional derivative. Assumes y is 2nd last dimension, works for 2,3,4 dim grids
#  first order centred difference unless boundary point or next to masked value
#
    if len(var.shape) <2:
        print("Cannot not supported for shapes of 0 or 1. Just use np.gradient?")
        return
    if len(var.shape) >4:
        print("Cannot not supported for shape > 4")
        return
    if type(dy)==float:
      dy=dy*np.ones((var.shape[-2],var.shape[-1]))
    if len(var.shape) == 4:
        nt,nz,ny,nx = var.shape
        dvar = np.ma.zeros((nt,nz,ny,nx))
        tmp2 = np.ma.zeros((nt,nz,ny,nx))
        tmp3 = np.ma.zeros((nt,nz,ny,nx))
        for i in range(ny-2):
            tmp2[:,:,i+1] = (var[:,:,i+2]-var[:,:,i+1])/(dy[i+1])
            tmp3[:,:,i+1] = (var[:,:,i+1]-var[:,:,i])/(dy[i+1])
        dvar = np.ma.mean((tmp2,tmp3),axis=0).copy()#tmp4.copy()
        dvar[:,:,0] = (var[:,:,1]-var[:,:,0])/dy[0]
        dvar[:,:,-1] = (var[:,:,-1]-var[:,:,-2])/dy[-1]
    if len(var.shape) == 3:
        nz,ny,nx = var.shape
        dvar = np.ma.zeros((nz,ny,nx))
        tmp2 = np.ma.zeros((nz,ny,nx))
        tmp3 = np.ma.zeros((nz,ny,nx))
        for i in range(ny-2):
            tmp2[:,i+1] = (var[:,i+2]-var[:,i+1])/(dy[i+1])
            tmp3[:,i+1] = (var[:,i+1]-var[:,i])/(dy[i+1])
        dvar = np.ma.mean((tmp2,tmp3),axis=0).copy()#tmp4.copy()
        dvar[:,0] = (var[:,1]-var[:,0])/dy[0]
        dvar[:,-1] = (var[:,-1]-var[:,-2])/dy[-1]
    if len(var.shape) == 2:
        ny,nx = var.shape
        dvar = np.ma.zeros((ny,nx))
        tmp2 = np.ma.zeros((ny,nx))
        tmp3 = np.ma.zeros((ny,nx))
        for i in range(ny-2):
            tmp2[i+1] = (var[i+2]-var[i+1])/(dy[i+1])
            tmp3[i+1] = (var[i+1]-var[i])/(dy[i+1])
        dvar = np.ma.mean((tmp2,tmp3),axis=0).copy()#tmp4.copy()
        dvar[0] = (var[1]-var[0])/dy[0]
        dvar[-1] = (var[-1]-var[-2])/dy[-1]
    return dvar

def ddx(var,dx):
#
# Calculate zonal derivative. Assumes x is last dimension, works for 2,3,4 dim grids
#  first order centred difference unless boundary point or next to masked value
#
    if len(var.shape) <2:
        print("Cannot not supported for shapes of 0 or 1. Just use np.gradient?")
        return
    if len(var.shape) >4:
        print("Cannot not supported for shape > 4")
        return
    if type(dx)==float:
      dx=dx*np.ones((var.shape[-2],var.shape[-1]))
    if len(var.shape) == 4:
        nt,nz,ny,nx = var.shape
        dvar = np.ma.zeros((nt,nz,ny,nx))
        tmp2 = np.ma.zeros((nt,nz,ny,nx))
        tmp3 = np.ma.zeros((nt,nz,ny,nx))
        for i in range(nx-2):
            tmp2[:,:,:,i+1] = (var[:,:,:,i+2]-var[:,:,:,i+1])/(dx[:,i+1])
            tmp3[:,:,:,i+1] = (var[:,:,:,i+1]-var[:,:,:,i])/(dx[:,i+1])
        dvar = np.ma.mean((tmp2,tmp3),axis=0).copy()
        dvar[:,:,:,0] = (var[:,:,:,1]-var[:,:,:,0])/dx[:,0]
        dvar[:,:,:,-1] = (var[:,:,:,-1]-var[:,:,:,-2])/dx[:,-1]
    if len(var.shape) == 3:
        nz,ny,nx = var.shape
        dvar = np.ma.zeros((nz,ny,nx))
        tmp2 = np.ma.zeros((nz,ny,nx))
        tmp3 = np.ma.zeros((nz,ny,nx))
        for i in range(nx-2):
            tmp2[:,:,i+1] = (var[:,:,i+2]-var[:,:,i+1])/(dx[:,i+1])
            tmp3[:,:,i+1] = (var[:,:,i+1]-var[:,:,i])/(dx[:,i+1])
        dvar = np.ma.mean((tmp2,tmp3),axis=0).copy()
        dvar[:,:,0] = (var[:,:,1]-var[:,:,0])/dx[:,0]
        dvar[:,:,-1] = (var[:,:,-1]-var[:,:,-2])/dx[:,-1]
    elif len(var.shape) == 2:
        ny,nx = var.shape
        dvar = np.ma.zeros((ny,nx))
        tmp2 = np.ma.zeros((ny,nx))
        tmp3 = np.ma.zeros((ny,nx))
        for i in range(nx-2):
            tmp2[:,i+1] = (var[:,i+2]-var[:,i+1])/(dx[:,i+1])
            tmp3[:,i+1] = (var[:,i+1]-var[:,i])/(dx[:,i+1])
        dvar = np.ma.mean((tmp2,tmp3),axis=0).copy()
        dvar[:,0] = (var[:,1]-var[:,0])/dx[:,0]
        dvar[:,-1] = (var[:,-1]-var[:,-2])/dx[:,-1]
    return dvar




def find_edge(data,lon,lat,sigma,theta_max=3.2,theta_min=-3.2,mag_min=0,minlen=1,spatial_mask=1,relative="Grid Cell",output='sparse',plotfreq=0,times=None,makefig=None):
   """
   Apply an edge filter to 3D atmospheric fields (probably surface q or RH with current setup)
   Calls a slightly modified version of the skimage canny algorithm
   Parameters
   ----------
   data : 3D array
      dataset to detect edges in, dimensions time,lat,lon
   lon : 1D array
      longitude coordinate
   lat : 1D array
      latitude coordinate
   sigma : float
      standard deviation for Gaussian smoothing, units are in number of grid cells
   theta_max : float
      Maximum angle (in radians) of edges to preserve
   theta_min : float
      Minimum angle (in radians) of edges to preserve
   mag_min : float
      Minimum gradient for thresholding 
   minlen : int
      Minimum length of an edge (in grid cells) for that edge to preserved
   spatial_mask : 2D boolean array or 1
      Masked location where edges will be sought
   relative : "Grid Cell","Weighted Grid Cell","Degrees","metres" or "meters"
      denominator units for mag_min
   output : "sparse" or "lists"
      Controls output type. 
      If "sparse", returns a sparse boolean array with same shape as data,
                   containing 1s where edge is detected
      If "lists", returns 2 lists of lists, containing lat and lon cooordinates
                   of detected edges for each day
   plotfreq : float between 0 and 1
      Frequency for plotting output. 0 for no plots, 1 for all plots.
      May help you tune the theta_max,theta_min,mag_min,minlen and spatial_mask parameters
      If you're running on a long time series, set to something like 0.01 to occasionally check
      that it's doing something sensible
   times : list of datetime objects
      Times for plotting purposes
   makefig : function
      option to make more sophisticated plots if plotfreq>0 
      leave as None to get basic plots
   """
 # Assertations that input shapes are correct
   assert(len(data.shape)==3)
   assert(len(lon.shape) == len(lat.shape))
   assert(len(lon.shape) in [1,2])
   if len(lon.shape)==1 and len(lat.shape)==1:
     assert(data.shape[1]==len(lat))
     assert(data.shape[2]==len(lon))
     lon2,lat2=np.meshgrid(lon,lat)
   elif len(lon.shape)==2 and len(lat.shape)==2:
     assert(data.shape[1]==len(lat.shape[0]))
     assert(data.shape[2]==len(lat.shape[1]))
     lon2,lat2=lon,lat
   if type(spatial_mask) != int:
     assert(len(spatial_mask.shape)==2)
     assert(spatial_mask.shape==data[0].shape)
   assert(relative in ["Grid Cell","Weighted Grid Cell","Degrees","metres","meters"])
 # Create spatial grid weights 
   dx,dy=get_dxdy(lon,lat,relative=relative)
   if type(dx) in [float,int]:
     dx=dx*np.ones(lon2.shape,dtype=float)
     dy=dy*np.ones(lon2.shape,dtype=float)
   assert output in ['sparse','lists']
   if output=='sparse':
     gather = []
   elif output=='lists':
     gather=[],[]
   if times==None:
     times=range(len(data))
   else:
     assert(len(times)==len(data))
   pi=0
   # iterate through time
   for tt,d in zip(times,data):
     # find all edges
     points=canny(d,sigma=sigma,dx=dx,dy=dy,low_threshold=mag_min,high_threshold=mag_min)#,mask=1-mask
     # restrict to desired angles and locations
     filt=(points<theta_max)*(points>theta_min)*spatial_mask
     # filter out edges that are too short
     mask,n=label(filt,structure=np.ones((3,3),dtype=bool))
     nmasked=np.array([(mask==j).sum() for j in range(n+1)])
     nmasked[0]=0
     filt = (nmasked[mask]>minlen)*filt
     # add to output array or lists
     if output=='sparse':
       gather.append(filt)
     if output=='lists':
        flat=filt.flatten()
        gather[0].append(lon2.flatten()[flat!=0])
        gather[1].append(lat2.flatten()[flat!=0])
     rand=np.random.rand()
     if rand < plotfreq:
        # plot edges and background data
        pi+=1
        plt.subplot(5,6,pi,aspect=1)
        if makefig==None:
          plt.title(tt)
          plt.contourf(lon,lat,d,np.linspace(data.min(),data.max(),20),cmap='Greys')
          plt.pcolor(lon,lat,np.ma.masked_array(points,1-filt),vmin=-np.pi,vmax=np.pi,cmap=cmocean.cm.phase)
        else:
          makefig(d,tt,points,filt)
        if pi == 30:
          plt.show()
          pi=0
   if plotfreq>0:
     plt.show()
   if output=='sparse':
     gather=np.array(gather)
   return gather


#
# size limits
#

def find_ridge(data,lon,lat,sigma,theta_max=3.2,theta_min=-3.2,mag_min=0,sign=-1,minlen=1,spatial_mask=1,output='sparse',plotfreq=0,times=None,makefig=None,window=4):
   """
   Apply an ridge filter to 3D atmospheric fields (Designed for wind convergence but can be used elsewhere)
   Parameters
   ----------
   data : 3D array
      dataset to detect ridges in, dimensions time,lat,lon
   lon : 1D array
      longitude coordinate
   lat : 1D array
      latitude coordinate
   sigma : float
      standard deviation for Gaussian smoothing, units are in number of grid cells
   theta_max : float
      Maximum angle (in radians) of ridges to preserve
   theta_min : float
      Minimum angle (in radians) of ridges to preserve
   mag_min : float
      Minimum value for thresholding
   sign: 1 or -1
      Whether to seek out ridges (1) or troughs (-1)
   minlen : int
      Minimum length of an ridge (in grid cells) for that edge to preserved
   spatial_mask : 2D boolean array or 1
      Masked location where edges will be sought
   output : "sparse" or "lists"
      Controls output type. 
      If "sparse", returns a sparse boolean array with same shape as data,
                   containing 1s where ridge is detected
      If "lists", returns 2 lists of lists, containing lat and lon cooordinates
                   of detected ridges for each day
   plotfreq : float between 0 and 1
      Frequency for plotting output. 0 for no plots, 1 for all plots.
      May help you tune the theta_max,theta_min,mag_min,minlen and spatial_mask parameters
      If you're running on a long time series, set to something like 0.01 to occasionally check
      that it's doing something sensible
   times : list of datetime objects
      Times for plotting purposes
   makefig : function
      option to make more sophisticated plots if plotfreq>0 
      leave as None to get basic plots
   window : integer
      number of grid-cells either side of each point to use when calculating eigenvectors of the
      inertial tensor (used to find ridge directions)
   """
   # Assertations that input shapes are correct
   assert(len(data.shape)==3)
   assert(len(lon.shape) == len(lat.shape))
   assert(len(lon.shape) in [1,2])
   if len(lon.shape)==1 and len(lat.shape)==1:
     assert(data.shape[1]==len(lat))
     assert(data.shape[2]==len(lon))
     lon2,lat2=np.meshgrid(lon,lat)
   elif len(lon.shape)==2 and len(lat.shape)==2:
     assert(data.shape[1]==len(lat.shape[0]))
     assert(data.shape[2]==len(lat.shape[1]))
     lon2,lat2=lon,lat
   if type(spatial_mask) != int:
     assert(len(spatial_mask.shape)==2)
     assert(spatial_mask.shape==data[0].shape)
   assert output in ['sparse','lists']
   if output=='sparse':
     gather = []
   elif output=='lists':
     gather=[],[]
   if times==None:
     times=range(len(data))
   else:
     assert(len(times)==len(data))
   pi=0
   vmax=max(data.max(),-data.min())
   # iterate through time
   for tt,d in zip(times,data):
     if sigma == None:
       ds=d.copy()
     else:
       ds = gaussian_filter(d,sigma) 
     # find all ridges
     points=canny_div(ds*sign,lon2,lat2,low_threshold=mag_min,high_threshold=mag_min,window=window)
     # restrict to desired angles and locations
     filt=((points<theta_max)*(points>theta_min)*spatial_mask).filled(0)
     # filter out ridges that are too short
     mask,n=label(filt,structure=np.ones((3,3),dtype=bool))
     nmasked=np.array([(mask==j).sum() for j in range(n+1)])
     nmasked[0]=0
     filt = (nmasked[mask]>minlen)*filt
     # add to output array or lists
     if output=='sparse':
       gather.append(filt)
     if output=='lists':
        flat=filt.flatten()
        gather[0].append(lon2.flatten()[flat!=0])
        gather[1].append(lat2.flatten()[flat!=0])
     rand=np.random.rand()
     if rand < plotfreq:
        k=max(1,int(len(lon)/20))
        pi+=1
        plt.subplot(5,6,pi,aspect=1)
        if makefig==None:
        # plot edges and background data
          plt.title(tt)
          plt.contourf(lon,lat,d,np.linspace(-vmax,vmax,20),cmap='RdBu')
          plt.pcolor(lon,lat,np.ma.masked_array(points,1-filt),vmin=-np.pi,vmax=np.pi,cmap=cmocean.cm.phase)
        else:
          makefig(d,tt,points,filt)
        if pi == 30:
          plt.show()
          pi=0
   if plotfreq>0:
     plt.show()
   if output=='sparse':
     gather=np.array(gather)
   return gather

