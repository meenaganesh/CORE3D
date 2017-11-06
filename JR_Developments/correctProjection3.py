# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 09:49:23 2017

@author: 200021424
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:00:38 2017

@author: 200021424
"""

from laspy.file import File

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import pickle
import gdal
import numpy as np
import osr
import utm
from lxml import objectify
from skimage import exposure
from affine import Affine
#from scipy.optimize import minimize


# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()
def get_pointcloud_latlong(inFile, refLat, refLon):
    latConv = np.zeros(len(inFile.x))
    longConv = np.zeros(len(inFile.x))
    # getting utm codes:
    tempLa, tempLo, regValue, regCode = utm.from_latlon(refLat, refLon)
    for i in range(0,len(inFile.x),5000):
        latConv[i], longConv[i] = utm.to_latlon(inFile.x[i], inFile.y[i],regValue,regCode)
    elevation = inFile.z;
    elevation = elevation[latConv != 0]
    latConv = latConv[latConv != 0]
    longConv = longConv[longConv != 0]
    return latConv, longConv, elevation

def read_raster(x_block_size, y_block_size, ds, bandCounter,xstart, ystart, xwidth, ywidth):
    band = ds.GetRasterBand(bandCounter)
    #xsize = min([band.XSize, 1*x_block_size])
    #ysize = min([band.YSize, 1*y_block_size])

    xsize = band.XSize
    if(xsize > xstart+xwidth):
        xsize = ystart+ywidth
    ysize = band.YSize
    if(ysize > ystart+ywidth):
        ysize = ystart+ywidth
    print(xsize, ysize)
    finalData = np.empty( shape=(0, 0) )
    blocks = 0
    for y in range(ystart, ysize, y_block_size):
        print(blocks)
        result = []
        array = np.empty([x_block_size, y_block_size]);
        #print blocks
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(xstart, xsize, x_block_size):
            if x + x_block_size < xsize:
                cols = x_block_size
            else:
                cols = xsize - x
            array = band.ReadAsArray(x, y, cols, rows)
            result.append(array)

            
            try:
                #array[array>0]=1
                print("we got them")
            except:
                print("could not find them")
            blocks += 1
        
        rowData = np.empty( shape=(0, 0) )
        for j in range(0,len(result),1):
            if(j == 0):
                rowData= result[j]
            else:
                rowData = np.concatenate((rowData,result[j]), axis=1)
        if(y == ystart):
            finalData = rowData
        else:
            finalData = np.concatenate((finalData, rowData))
        
    '''
    figure = plt.figure();
    ax = figure.add_subplot(111);
    ax.imshow(finalData, cmap=cm.BrBG_r)
    plt.show();            
    '''
    return finalData

def transform(x, cornerPoints, pixelPoints):
    fwd = Affine.from_gdal(*tuple(x))
    return np.max(np.max(np.sqrt(abs(np.array(fwd*(pixelPoints[:,0],pixelPoints[:,1])).T-cornerPoints))))

def loadRPC(filename):
    xmlObject = objectify.parse(filename) 
    rpcDict = {};
    temp = xmlObject.find('RPB').find('IMAGE');
    rpcDict['ERRBIAS'] = float(temp.find('ERRBIAS'))
    rpcDict['ERRRAND'] = float(temp.find('ERRRAND'))
    rpcDict['LINEOFFSET'] = int(temp.find('LINEOFFSET'))
    rpcDict['SAMPOFFSET'] = int(temp.find('SAMPOFFSET'))
    rpcDict['LATOFFSET'] = float(temp.find('LATOFFSET'))
    rpcDict['LONGOFFSET'] = float(temp.find('LONGOFFSET'))
    rpcDict['HEIGHTOFFSET'] = float(temp.find('HEIGHTOFFSET'))
    rpcDict['LINESCALE'] = int(temp.find('LINESCALE'))
    rpcDict['SAMPSCALE'] = int(temp.find('SAMPSCALE'))
    rpcDict['LATSCALE'] = float(temp.find('LATSCALE'))
    rpcDict['LONGSCALE'] = float(temp.find('LONGSCALE'))
    rpcDict['HEIGHTSCALE'] = float(temp.find('HEIGHTSCALE'))
    
    rpcDict['LINENUMCOEF'] = [float(s) for s in temp.find('LINENUMCOEFList').find('LINENUMCOEF').text.split(' ')] #int(temp.find('LINENUMCOEFList').find('LINENUMCOEF').text)
    rpcDict['LINEDENCOEF'] = [float(s) for s in temp.find('LINEDENCOEFList').find('LINEDENCOEF').text.split(' ')] 
    rpcDict['SAMPNUMCOEF'] = [float(s) for s in temp.find('SAMPNUMCOEFList').find('SAMPNUMCOEF').text.split(' ')] 
    rpcDict['SAMPDENCOEF'] = [float(s) for s in temp.find('SAMPDENCOEFList').find('SAMPDENCOEF').text.split(' ')] 
    return rpcDict    
'''
def getImage(cornerPoints,image, rpcInformation):

    normLat = (cornerPoints[0,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (cornerPoints[0,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= 0
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x0 = normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET']
    y0 = normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET']
    normLat = (cornerPoints[3,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (cornerPoints[3,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= 0
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x1 = normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET']
    y1 = normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET']    
    return normRow, normCol
'''
def getPolyValue(coefficients, normLat, normLong, normHeight):
    P = normLat
    L = normLong
    H = normHeight
    return coefficients[0]+coefficients[1]*L+coefficients[2]*P+coefficients[3]*H+\
    coefficients[4]*L*P+coefficients[5]*L*H+coefficients[6]*P*H+\
    coefficients[7]*L*L+coefficients[8]*P*P+coefficients[9]*H*H+\
    coefficients[10]*P*L*H+coefficients[11]*L*L*L+coefficients[12]*L*P*P+\
    coefficients[13]*L*H*H+coefficients[14]*L*L*P+coefficients[15]*P*P*P+\
    coefficients[16]*P*H*H+coefficients[17]*L*L*H+coefficients[18]*P*P*H+\
    coefficients[19]*H*H*H
            
inFile = File('D:/Core3D/Data/Vricon_Point_Cloud/data/0813635w_301459n_20170425T205860Z_ptcld.las', mode='r')

#inFile2 = File('D:/Core3D/Data/jacksonville/satellite_imagery/WV3/MSI/0813635w_301459n_20170425T205860Z_ptcld.las', mode='r')

#img = 'D:/Core3D/Data/jacksonville/satellite_imagery/WV3/SWIR/01NOV15WV031100015NOV01162032-A1BS-500900785020_01_P001_________SW_U0AAAAAACAAC0.NTF'
#img = 'D:/Core3D/Data/jacksonville/satellite_imagery/WV3/MSI/01MAY15WV031200015MAY01160357-M1BS-500648062030_01_P001_________GA_E0AAAAAAKAAK0.NTF'
#img = 'D:/Core3D/Data/jacksonville/satellite_imagery/WV3/MSI/19APR15WV031100015APR19161439-M1BS-501504474050_01_P001_________GA_E0AAAAAAKAAK0.NTF'
img = 'D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/05SEP16WV021200016SEP05162552-M1BS-500881026010_01_P006_________GA_E0AAAAAAIAAG0.NTF'
ds = gdal.Open(img, gdal.GA_ReadOnly)
band = ds.GetRasterBand(1)
block_sizes = band.GetBlockSize()
x_block_size = block_sizes[0]
y_block_size = block_sizes[1]
ystart = 0
xstart = 0
xwidth = 15000
ywidth = 15000

data = read_raster(x_block_size, y_block_size,ds,1,xstart , ystart, xwidth,ywidth)
dataShapebefore = data.shape


#result = objectify.parse('D:/Core3D/Data/jacksonville/satellite_imagery/WV3/SWIR/15NOV01162032-A1BS-500900785020_01_P001.XML')
#result = objectify.parse('D:/Core3D/Data/jacksonville/satellite_imagery/WV3/MSI/15MAY01160357-M1BS-500648062030_01_P001.XML')
#result = objectify.parse('D:/Core3D/Data/jacksonville/satellite_imagery/WV3/MSI/15APR19161439-M1BS-501504474050_01_P001.XML')
result = objectify.parse('D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/16SEP05162552-M1BS-500881026010_01_P006.XML') 
bounds = np.zeros((4,2))
bounds[0,0] = result.find('TIL').find('TILE').find('ULLAT');
bounds[0,1] = result.find('TIL').find('TILE').find('ULLON');
#elevation: ULHAE>-1.945000000000000e+01</ULHAE>
bounds[1,0] = result.find('TIL').find('TILE').find('URLAT');
bounds[1,1] = result.find('TIL').find('TILE').find('URLON');
bounds[2,0] = result.find('TIL').find('TILE').find('LRLAT');
bounds[2,1] = result.find('TIL').find('TILE').find('LRLON');
bounds[3,0] = result.find('TIL').find('TILE').find('LLLAT');
bounds[3,1] = result.find('TIL').find('TILE').find('LLLON');
m = Basemap(projection='cyl',llcrnrlon=bounds[3,1]-.1,llcrnrlat=bounds[3,0]-.1,urcrnrlon=bounds[1,1]+.1,urcrnrlat=bounds[1,0]+.1, epsg=4326)

rpcInformation = loadRPC('D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/16SEP05162552-M1BS-500881026010_01_P006.XML')
m.readshapefile('D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/16SEP05162552-M1BS-500881026010_01_P006_PIXEL_SHAPE', 'pixel',drawbounds = False)

for shape in m.pixel:
    x, y, = zip(*shape)
    m.plot(x, y, linewidth = 2, color='white', alpha=.4) 


cornerPoints = np.zeros((4,2))
deltaLong = abs(m.pixel[0][0][0]-m.pixel[0][1][0])/10
deltaLat= abs(m.pixel[0][1][1]-m.pixel[0][3][1])/10
cornerPoints[0, 0] = m.pixel[0][0][0]#+deltaLong
cornerPoints[0, 1] = m.pixel[0][0][1]#-deltaLat
cornerPoints[1, 0] = m.pixel[0][1][0]#-deltaLong
cornerPoints[1, 1] = m.pixel[0][1][1]#-deltaLat
cornerPoints[2, 0] = m.pixel[0][2][0]#-deltaLong
cornerPoints[2, 1] = m.pixel[0][2][1]#+deltaLat
cornerPoints[3, 0] = m.pixel[0][3][0]#+deltaLong
cornerPoints[3, 1] = m.pixel[0][3][1]#+deltaLat

normLat = (cornerPoints[0,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
normLong = (cornerPoints[0,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
normHeight= 0
normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
x0 =np.floor( normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
y0 = np.floor(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])
if(x0 < 0):
    x0 = 0
if(y0 < 0):
    y0 = 0
normLat = (cornerPoints[2,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
normLong = (cornerPoints[2,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
normHeight= 0
normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
x1 = np.ceil(normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
y1 = np.ceil(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])   

#Grab image information:
#latOffset = float(result.find('RPB').find('IMAGE').find('LATOFFSET'))
#longOffset = float(result.find('RPB').find('IMAGE').find('LONGOFFSET'))
#imageXScale = float(result.find('RPB').find('IMAGE').find('LINESCALE'))*2
#imageYScale = float(result.find('RPB').find('IMAGE').find('SAMPSCALE'))*2   
#Eliminate blank pixels:
data = data[int(x0):int(x1),:]
data = data[:,int(y0):int(y1)]



'''
bounds = np.zeros((4,2))
bounds[0,0] = result.find('IMD').find('BAND_C').find('ULLAT');
bounds[0,1] = result.find('IMD').find('BAND_C').find('ULLON');
#elevation: ULHAE>-1.945000000000000e+01</ULHAE>
bounds[1,0] = result.find('IMD').find('BAND_C').find('URLAT');
bounds[1,1] = result.find('IMD').find('BAND_C').find('URLON');
bounds[2,0] = result.find('IMD').find('BAND_C').find('LRLAT');
bounds[2,1] = result.find('IMD').find('BAND_C').find('LRLON');
bounds[3,0] = result.find('IMD').find('BAND_C').find('LLLAT');
bounds[3,1] = result.find('IMD').find('BAND_C').find('LLLON');
'''





#ds = None


# create a grid of xy coordinates in the original projection
#xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
#ystart = ystart-35;
#xstart = xstart+46;

ystart = y0
xstart = x0
xy_source = np.mgrid[ystart:y1:1,xstart:x1:1]#dataShapebefore[1]/data.shape[1],
                     #dataShapebefore[0]/data.shape[0]]
#xy_source = np.mgrid[0:dataShapebefore[1]:dataShapebefore[1]/data.shape[1],0:dataShapebefore[0]:dataShapebefore[0]/data.shape[0]]
v_min, v_max = np.percentile(data, (0.2, 99.8))

better_contrast = exposure.rescale_intensity(data, in_range=(v_min, v_max)).T
plotCoordinates = xy_source[:,0:better_contrast.shape[0],0:better_contrast.shape[1]]

fig = plt.figure()
ax = Axes3D(fig)

ax.set_axis_off()


temp = m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=2000, verbose=True, zorder=1).get_array()


#m.readshapefile('D:/Core3D/development/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_roads', 'Streets',drawbounds = False)

m.readshapefile('D:/Core3D/Data/jacksonville/satellite_imagery/WV2/MSI/16SEP05162552-M1BS-500881026010_01_P006_PIXEL_SHAPE',\
                       'pixel',drawbounds = False)

for shape in m.pixel:
    x, y, = zip(*shape)
    #ax.plot3D(x,y,0,linewidth = 2, color='black', alpha=.4 )
    m.plot(x, y, linewidth = 2, color='white', alpha=.4) 

m.readshapefile('D:/Core3D/development/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_roads', 'Streets',drawbounds = False)

for shape in m.Streets:
    x, y, = zip(*shape)
    #ax.plot3D(x,y,0,linewidth = 2, color='green', alpha=.4 )
    m.plot(x, y, linewidth = 2, color='green', alpha=.4)
    



#gt[0] = m.pixel[0][0][0]
#gt[3] = m.pixel[0][0][1]


pixelPoints = np.zeros((4,2))
pixelPoints[1, 0] = data.shape[1]
pixelPoints[2, 1] = data.shape[0]
pixelPoints[2, 0] = data.shape[1]
pixelPoints[3, 1] = data.shape[0]
'''
cornerPoints = np.zeros((4,2))
cornerPoints[0, 0] = m.pixel[0][0][0]
cornerPoints[0, 1] = m.pixel[0][0][1]
cornerPoints[1, 0] = m.pixel[0][1][0]
cornerPoints[1, 1] = m.pixel[0][1][1]
cornerPoints[2, 0] = m.pixel[0][2][0]
cornerPoints[2, 1] = m.pixel[0][2][1]
cornerPoints[3, 0] = m.pixel[0][3][0]
cornerPoints[3, 1] = m.pixel[0][3][1]
'''
proj = osr.SpatialReference()
proj.SetWellKnownGeogCS( "EPSG:4326" )
ds.SetProjection(proj.ExportToWkt())
gt = gdal.GCPsToGeoTransform(ds.GetGCPs())
'''
gt = np.array(gt)
gt[0] = -81.69290000000001#longOffset
gt[3] = 30.4513#latOffset
'''
gcp_list = []

for i in range(0,4):
    pixel = int(pixelPoints[i,0])
    line = int(pixelPoints[i,1])
    x = cornerPoints[i,0]
    y = cornerPoints[i,1]
    z = 0;
    print(x,y,z,pixel,line)
    gcp = gdal.GCP(x, y, z, pixel, line)
    gcp_list.append(gcp)

gt2 = gdal.GCPsToGeoTransform(gcp_list)


fwd = Affine.from_gdal(*tuple(gt2))
plotCoordinates2 = fwd*plotCoordinates
#ax.view_init(azim=0, elev=90)
m.contourf(plotCoordinates2[0],plotCoordinates2[1],better_contrast,cmap='gray', alpha = 1)
plt.show
'''
ax.contourf(plotCoordinates2[0],plotCoordinates2[1],better_contrast,cmap='gray', alpha = 1)
#m.ax.plot_surface(plotCoordinates2[0],plotCoordinates2[1],better_contrast,cmap='gray', alpha = 0.9)
#m.pcolormesh(plotCoordinates2[0],plotCoordinates2[1],better_contrast,cmap='gray')
#im1 = m.pcolormesh(xy_source[0,:], xy_source[1,:], data.T, cmap='gray')
lat, long, elevation = get_pointcloud_latlong(inFile,bounds[0,0],bounds[0,1])
#ax.add_collection3d(m.scatter(long,lat,3, c='b', marker='.'))
#plt.show

#fig = plt.figure();
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(lat, long, elevation, c='b', marker='.')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Elevation')
plt.show
'''