# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:07:27 2017

@author: 200021424
"""


import gdal
import numpy as np
from scipy.optimize import minimize
#import utm
from pyproj import Proj, transform
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from affine import Affine
from laspy.file import File

# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

def getPointcloudLatLong(filename, refLat, refLong):
    # Convert the UTM information in the .las files to lat/long
    # Input:
    # filename: Complete path to .las file to be loaded
    # refLat: Reference Latitude to get EPSG information
    # refLong: Reference Longitude to get EPSG information
    # Output:
    # latConv: Resulting point latitudes
    # longConv: Resulting point longitudes
    # elevation: Resulting elevation
    # rgb: RGB values for each point
    # cornerPoints: Approximate bounding box cornerpoints
    inFile = File(filename, mode='r')
    latConv = np.zeros(len(inFile.x))
    longConv = np.zeros(len(inFile.x))
    EPSG=32700-np.max(np.round((45+refLat)/90)*100,0)+np.max(np.round((183+refLong)/6),0)
    inProj = Proj(init='epsg:'+str(int(EPSG)))
    outProj = Proj(init='epsg:4326')
    longConv, latConv = transform(inProj, outProj, inFile.x, inFile.y)

    elevation = inFile.z;
    rgb = np.zeros((elevation.shape[0],4),dtype=np.float64)
    rgb[:,0] = inFile.red/65536 # red
    rgb[:,1] = inFile.green/65536 # green
    rgb[:,2] = inFile.blue/65536 # blue
    rgb[:,3] = 0.75

    # Determine approximate bounding box
    # TODO: Calculate the actual bounding box and not just pull points from 
    # the point cloud
    cornerPoints = np.array([longConv,latConv]).T
    temp = ConvexHull(cornerPoints)
    keepIndx = temp.neighbors[0:2,:].flatten()
    return latConv, longConv, elevation, rgb, cornerPoints[temp.vertices][keepIndx,:]



def boundingInImage(filename, imageBoundingBox):
    # Check if a point cloud is contained in an image bounding box
    # Input:
    # filename: Point cloud .las filename
    # imageBoundingBox: Numpy array (4,2) giving the lat long of the four corners of the image
    # Output:
    # Boolean indicating if the point cloud is in the image
    inFile = File(filename, mode='r')
    EPSG=32700-np.max(np.round((45+imageBoundingBox[0,1])/90)*100,0)+np.max(np.round((183+imageBoundingBox[0,0])/6),0)
    inProj = Proj(init='epsg:'+str(int(EPSG)))
    outProj = Proj(init='epsg:4326')
    long0, lat0 = transform(inProj, outProj, inFile.header.min[0], inFile.header.min[1])
    long1, lat1 = transform(inProj, outProj, inFile.header.max[0], inFile.header.max[1])
    polygon = Polygon(imageBoundingBox) 
    inBox = True
    if(not polygon.contains(Point(long0, lat0))):
        inBox = False
    if(not polygon.contains(Point(long0, lat1))):
        inBox = False
    if(not polygon.contains(Point(long1, lat0))):
        inBox = False
    if(not polygon.contains(Point(long1, lat1))):
        inBox = False
    return inBox        

def loadRasters(filename):
    # Load the NTF raster file
    # Input:
    # filename: NTF filename
    # Output:
    # Numpy array size X x Y x #Bands 
    ds = gdal.Open(filename)
    xSize = ds.RasterXSize
    ySize = ds.RasterYSize
    rasterCount = ds.RasterCount
    data = np.zeros((ySize, xSize, rasterCount))
    for rasterCounter in range(1, rasterCount+1):
        data[:,:,rasterCounter-1] = ds.GetRasterBand(rasterCounter).ReadAsArray()
    return data

def loadRPC(ntfFilename):
    # Get the RPC information needed to correct image
    # Input:
    # ntfFilename: Name of .NFT file to work with
    # Output:
    # Dictionary with the RPC information
    options = gdal.InfoOptions([], format='json')
    jsonObject = gdal.Info(ntfFilename,options=options)
    rpcDict = {};
    jsonObject.get('metadata')['RPC']
    rpcDict['LINEOFFSET'] = int(jsonObject.get('metadata')['RPC']['LINE_OFF'])
    rpcDict['SAMPOFFSET'] = int(jsonObject.get('metadata')['RPC']['SAMP_OFF'])
    rpcDict['LATOFFSET'] = float(jsonObject.get('metadata')['RPC']['LAT_OFF'])
    rpcDict['LONGOFFSET'] = float(jsonObject.get('metadata')['RPC']['LONG_OFF'])
    rpcDict['HEIGHTOFFSET'] = float(jsonObject.get('metadata')['RPC']['HEIGHT_OFF'])
    rpcDict['LINESCALE'] = int(jsonObject.get('metadata')['RPC']['LINE_SCALE'])
    rpcDict['SAMPSCALE'] = int(jsonObject.get('metadata')['RPC']['SAMP_SCALE'])
    rpcDict['LATSCALE'] = float(jsonObject.get('metadata')['RPC']['LAT_SCALE'])
    rpcDict['LONGSCALE'] = float(jsonObject.get('metadata')['RPC']['LONG_SCALE'])
    rpcDict['HEIGHTSCALE'] = float(jsonObject.get('metadata')['RPC']['HEIGHT_SCALE'])
   
    rpcDict['LINENUMCOEF'] =  [float(s) for s in list(filter(None, jsonObject.get('metadata')['RPC']['LINE_NUM_COEFF'].split(' ')))]
    rpcDict['LINEDENCOEF'] = [float(s) for s in list(filter(None, jsonObject.get('metadata')['RPC']['LINE_DEN_COEFF'].split(' ')))] 
    rpcDict['SAMPNUMCOEF'] = [float(s) for s in list(filter(None, jsonObject.get('metadata')['RPC']['SAMP_NUM_COEFF'].split(' ')))] 
    rpcDict['SAMPDENCOEF'] = [float(s) for s in list(filter(None, jsonObject.get('metadata')['RPC']['SAMP_DEN_COEFF'].split(' ')))] 
    return rpcDict    

def getTileBounds(ntfFilename):
    # Get the tile bounds from the NTF file
    # Input:
    # ntfFilename: Name of .NFT file to work with
    # Output:
    # Numpy array of size (4,2) with longitude and latitude of the four corners of the image
    options = gdal.InfoOptions([], format='json')
    jsonObject = gdal.Info(ntfFilename,options=options)
    bounds = np.zeros((4,2))
    bounds[0,0] = jsonObject.get('gcps')['gcpList'][0]['y']
    bounds[0,1] = jsonObject.get('gcps')['gcpList'][0]['x']
    bounds[1,0] = jsonObject.get('gcps')['gcpList'][1]['y']
    bounds[1,1] = jsonObject.get('gcps')['gcpList'][1]['x']
    bounds[2,0] = jsonObject.get('gcps')['gcpList'][2]['y']
    bounds[2,1] = jsonObject.get('gcps')['gcpList'][2]['x']
    bounds[3,0] = jsonObject.get('gcps')['gcpList'][3]['y']
    bounds[3,1] = jsonObject.get('gcps')['gcpList'][3]['x']
    return bounds
    
# Get image associated with a bounding box described by four corner points.
def getPatch(cornerPoints,image, rpcInformation):
    normLat = (cornerPoints[0,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (cornerPoints[0,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= (0-rpcInformation['HEIGHTOFFSET'])/rpcInformation['HEIGHTSCALE']
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x0 =np.floor( normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
    y0 = np.floor(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])

    normLat = (cornerPoints[1,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (cornerPoints[1,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= (0-rpcInformation['HEIGHTOFFSET'])/rpcInformation['HEIGHTSCALE']
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x1 = np.ceil(normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
    y1 = np.ceil(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])   

    normLat = (cornerPoints[2,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (cornerPoints[2,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= (0-rpcInformation['HEIGHTOFFSET'])/rpcInformation['HEIGHTSCALE']
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x2 =np.floor( normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
    y2 = np.floor(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])

    normLat = (cornerPoints[3,1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (cornerPoints[3,0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= (0-rpcInformation['HEIGHTOFFSET'])/rpcInformation['HEIGHTSCALE']
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x3 =np.floor( normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
    y3 = np.floor(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])  
    
    x0 = np.min([x0,x1])
    x0 = np.max([0,x0])
    x1 = np.max([x2,x3])
    x1 = np.min([x1,image.shape[0]])
    y0 = np.min([y0,y3])
    y0 = np.max([0,y0])
    y1 = np.max([y1, y2])
    y1 = np.min([y1, image.shape[1]])
    image = image[int(x0):int(x1),:,:]
    returnImage = image[:,int(y0):int(y1),:]
    return returnImage, x0, y0, x1, y1


def getPolyValue(coefficients, normLat, normLong, normHeight):
    # Transform provided in the RPC standard.
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
    
# CoordinateError is used by tie inverse lookup to find the nearest lat long
# corresponding to a pixel coordinate
def coordinateError(input, x, y,rpcInformation):
    normLat = (input[1]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (input[0]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= (0-rpcInformation['HEIGHTOFFSET'])/rpcInformation['HEIGHTSCALE']
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x0 =np.floor( normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
    y0 = np.floor(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])
    return np.sqrt(pow(x-x0,2)+pow(y-y0,2))

# Get the pixel coordinate corresponding to lat long coordinate
def getCorrespondingPixel(input,rpcInformation):
    normLat = (input[0]-rpcInformation['LATOFFSET'])/rpcInformation['LATSCALE']
    normLong = (input[1]-rpcInformation['LONGOFFSET'])/rpcInformation['LONGSCALE']
    normHeight= (0-rpcInformation['HEIGHTOFFSET'])/rpcInformation['HEIGHTSCALE']
    normRow = getPolyValue(rpcInformation['LINENUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['LINEDENCOEF'],normLat, normLong, normHeight)
    normCol = getPolyValue(rpcInformation['SAMPNUMCOEF'],normLat, normLong, normHeight)/getPolyValue(rpcInformation['SAMPDENCOEF'],normLat, normLong, normHeight)
    x0 =np.floor( normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
    y0 = np.floor(normCol*rpcInformation['SAMPSCALE']+rpcInformation['SAMPOFFSET'])
    return x0, y0

# Inverse lookup finds the lat/long information for a given pixel.
def inverseLookup(x0, target, rpcInformation):
    res = minimize(coordinateError, x0, args=(target[0], target[1], rpcInformation), method='Powell',
                options={'maxfev': 200000, 'xtol': 1e-12, 'disp': True})
    resultLat = res.x[0]
    resultLong = res.x[1]
    return  resultLat, resultLong
            
# Take a small patch of an image, its bounding box (x0, y0, x1, y1) along with 
# the original collection of 4 lat/long coordinates of interest and RPC information
# Return the longitude, latitud for each pixel in the image.
def projectImage(inputPatch, x0, y0, x1, y1, cornerPoints, rpcInformation):
    cornerPoints[0,:] = inverseLookup(cornerPoints[0,:],(x0,y0),rpcInformation)
    cornerPoints[1,:] = inverseLookup(cornerPoints[1,:],(x0,y1),rpcInformation)
    cornerPoints[2,:] = inverseLookup(cornerPoints[2,:],(x1,y1),rpcInformation)
    cornerPoints[3,:] = inverseLookup(cornerPoints[3,:],(x1,y0),rpcInformation)
    xySource = np.mgrid[y0:y1:1,x0:x1:1]
    inputPatch = inputPatch.transpose(1,0,2)
    plotCoordinates = xySource[:,0:inputPatch.shape[0],0:inputPatch.shape[1]]
    pixelPoints = np.ones((4,2))
    pixelPoints[0,0] = y0
    pixelPoints[0,1] = x0
    pixelPoints[1, 0] = y1
    pixelPoints[1, 1] = x0
    pixelPoints[2, 0] = y1
    pixelPoints[2, 1] = x1
    pixelPoints[3, 0] = y0
    pixelPoints[3, 1] = x1
    
    gcp_list = [] #List of Ground Control Points

    for i in range(0,4):
        pixel = pixelPoints[i,0].item()
        line = pixelPoints[i,1].item()
        x = cornerPoints[i,0].item()
        y = cornerPoints[i,1].item()
        z = 0;

        gcp = gdal.GCP(x, y, z, pixel, line)
        gcp_list.append(gcp)
    polygon = Polygon(cornerPoints)  
    for i in range(0,25):
        a = np.random.uniform(.8,1,1)[0]
        b = np.random.uniform(.8,1,1)[0]
        x = (a*cornerPoints[0,0]+(2-a)*cornerPoints[2,0])/2
        y = (b*cornerPoints[0,1]+(2-b)*cornerPoints[2,1])/2
        point = Point(x,y)
        if(polygon.contains(point)):
            line, pixel = getCorrespondingPixel((y,x),rpcInformation)
            z = 0
            gcp = gdal.GCP(x, y, z, pixel, line)
            gcp_list.append(gcp)
    # Generating the Geo Transform:
    gt2 = gdal.GCPsToGeoTransform(gcp_list)
    fwd = Affine.from_gdal(*gt2)
    long, lat= fwd*plotCoordinates
    return long, lat, inputPatch