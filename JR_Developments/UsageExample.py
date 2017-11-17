# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:17:42 2017

@author: 200021424
"""


from GISDataFunctions import helper_functions as hf
from GISDataFunctions import WVCalKernel as WVCK
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#import numpy as np

# Open NTF file for processing
img = 'D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/05SEP16WV021200016SEP05162552-M1BS-500881026010_01_P006_________GA_E0AAAAAAIAAG0.NTF'
data = hf.loadRasters(img)
rpcInformation = hf.loadRPC(img)
bounds = hf.getTileBounds(img)


# Getting point cloud information:
pointCloudFile = 'D:/Core3D/Data/Vricon_Point_Cloud/data/0813636w_301725n_20170425T205946Z_ptcld.las'
pointLat, pointLong, elevation, rgbColor, cornerPoints = hf.getPointcloudLatLong(pointCloudFile,bounds[0,0],bounds[0,1])


# Start plotting:
fig = plt.figure()
ax = fig.add_subplot(111);
m = Basemap(projection='cyl',llcrnrlon=bounds[3,1]-.1,llcrnrlat=bounds[3,0]-.1,urcrnrlon=bounds[1,1]+.1,urcrnrlat=bounds[1,0]+.1, epsg=4326)

# Plot a map for the specified view: xpixel for a higher resolution can be set
# to 2000 instead of 200
#m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=200, verbose=True, zorder=1).get_array()

# Draw roads from osm_roads:
m.readshapefile('D:/Core3D/development/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_roads', 'Streets',drawbounds = False)

for shape in m.Streets:
    x, y, = zip(*shape)
    m.plot(x, y, linewidth = 2, color='green', alpha=.4)

# Grab the tile shape (PIXEL_SHAPE):
m.readshapefile('D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/16SEP05162552-M1BS-500881026010_01_P006_PIXEL_SHAPE', 'corners',drawbounds = False)

for shape in m.corners:
    x, y, = zip(*shape)
    m.plot(x, y, linewidth = 2, color='white', alpha=.4) 
    

# get the patch along with its upper left and lower right image coordinates:
croppedData, x0, y0, x1, y1 = hf.getPatch(cornerPoints, data, rpcInformation) 

# Image Calibration:
wv2 = WVCK.WV2params()
test = WVCK.RadiometricCalibrator(img, wv2,sub_array = croppedData)

# get the associated lat long information for every point in the image:
long, lat, plotPatch = hf.projectImage(croppedData, x0, y0, x1, y1, cornerPoints, rpcInformation)

# plot the result:
m.pcolormesh(long,lat,plotPatch[:,:,1],cmap='gray')



#if(hf.boundingInImage(pointCloudFile, cornerPoints)):
ax.scatter(pointLong, pointLat, c=rgbColor, marker="s")
plt.show()