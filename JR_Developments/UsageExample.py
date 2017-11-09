# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:17:42 2017

@author: 200021424
"""

import gdal
from GISDataFunctions import helper_functions as hf
from lxml import objectify
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Open NTF file for processing
img = 'D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/05SEP16WV021200016SEP05162552-M1BS-500881026010_01_P006_________GA_E0AAAAAAIAAG0.NTF'
ds = gdal.Open(img)

# get raster data from first band:
data = ds.GetRasterBand(1).ReadAsArray()

xmlFile = 'D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/16SEP05162552-M1BS-500881026010_01_P006.XML'
# XML file associated with NTF:
rpcInformation = hf.loadRPC(xmlFile)

# Setup plotting for sanity check. Assume the lat long points are provided:\
# (here taken from the xml file)
result = objectify.parse(xmlFile) 
bounds = np.zeros((4,2))
bounds[0,0] = result.find('TIL').find('TILE').find('ULLAT');
bounds[0,1] = result.find('TIL').find('TILE').find('ULLON');
bounds[1,0] = result.find('TIL').find('TILE').find('URLAT');
bounds[1,1] = result.find('TIL').find('TILE').find('URLON');
bounds[2,0] = result.find('TIL').find('TILE').find('LRLAT');
bounds[2,1] = result.find('TIL').find('TILE').find('LRLON');
bounds[3,0] = result.find('TIL').find('TILE').find('LLLAT');
bounds[3,1] = result.find('TIL').find('TILE').find('LLLON');

fig = plt.figure()
ax = fig.add_subplot(111);
m = Basemap(projection='cyl',llcrnrlon=bounds[3,1]-.1,llcrnrlat=bounds[3,0]-.1,urcrnrlon=bounds[1,1]+.1,urcrnrlat=bounds[1,0]+.1, epsg=4326)

# Plot a map for the specified view: xpixel for a higher resolution can be set
# to 2000 instead of 200
m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=200, verbose=True, zorder=1).get_array()

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
    
# Select corner points of interest:
cornerPoints = np.zeros((4,2))
deltaLong = abs(m.corners[0][0][0]-m.corners[0][2][0])/2.2
deltaLat1= abs(m.corners[0][0][1]-m.corners[0][3][1])/2.2
deltaLat2= abs(m.corners[0][1][1]-m.corners[0][2][1])/2.2
cornerPoints[0, 0] = m.corners[0][0][0]+deltaLong
cornerPoints[0, 1] = m.corners[0][0][1]-deltaLat1
cornerPoints[1, 0] = m.corners[0][1][0]-deltaLong
cornerPoints[1, 1] = m.corners[0][1][1]-deltaLat2
cornerPoints[2, 0] = m.corners[0][2][0]-deltaLong
cornerPoints[2, 1] = m.corners[0][2][1]+deltaLat2
cornerPoints[3, 0] = m.corners[0][3][0]+deltaLong
cornerPoints[3, 1] = m.corners[0][3][1]+deltaLat1

# get the patch along with its upper left and lower right image coordinates:
croppedData, x0, y0, x1, y1 = hf.getPatch(cornerPoints, data, rpcInformation)

# get the associated lat long information for every point in the image:
long, lat, plotPatch = hf.projectImage(croppedData, x0, y0, x1, y1, cornerPoints, rpcInformation)

# plot the result:
m.pcolormesh(lat,long,plotPatch,cmap='gray')