# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:07:27 2017

@author: 200021424
"""


import gdal
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
import shapefile
import math
from pyproj import Proj, transform
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from affine import Affine
from laspy.file import File
from PIL import Image, ImageDraw
from scipy.spatial import distance as dist
import imreg_dft as ird
from pathlib import Path
import json
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

 
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([bl, br,tr ,tl ], dtype="float32")#[tl, tr, br, bl]

def getPointcloudLatLong(filename):
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
    parentDirectory = Path(filename).parents[1]
    metadatafile = str(parentDirectory)+'\\metadata.json';
    projectionData = json.load(open(metadatafile))
    inFile = File(filename, mode='r')
    #latConv = np.zeros(len(inFile.x))
    #longConv = np.zeros(len(inFile.x))
    #EPSG=32700-np.max(np.round((45+refLat)/90)*100,0)+np.max(np.round((183+refLong)/6),0)
    EPSG = projectionData['srs']['epsg']
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
    temp = cornerPoints[ConvexHull(cornerPoints).vertices]
    pi2 = np.pi/2
    # calculate edge angles
    edges = np.zeros((len(temp)-1, 2))
    edges = temp[1:] - temp[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))
    
     # apply rotations to the hull
    rot_points = np.dot(rotations, temp.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)#3,2
    rval[1] = np.dot([x2, y2], r)#4,3
    rval[2] = np.dot([x2, y1], r)#1,0
    rval[3] = np.dot([x1, y1], r)#2,1  
    # order the points top left, top right, lower right, lower left:
    returnValues = order_points(rval)
     
    
    
    #keepIndx = temp.neighbors[0:2,:].flatten()
    return latConv, longConv, elevation, rgb, returnValues#cornerPoints[temp.vertices][keepIndx,:]



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
    if not polygon.contains(Point(long0, lat0)):
        inBox = False
    if not polygon.contains(Point(long0, lat1)):
        inBox = False
    if not polygon.contains(Point(long1, lat0)):
        inBox = False
    if not polygon.contains(Point(long1, lat1)):
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
    if 'gcps' in jsonObject:
        bounds[0,0] = jsonObject.get('gcps')['gcpList'][0]['x']
        bounds[0,1] = jsonObject.get('gcps')['gcpList'][0]['y']
        bounds[1,0] = jsonObject.get('gcps')['gcpList'][1]['x']
        bounds[1,1] = jsonObject.get('gcps')['gcpList'][1]['y']
        bounds[2,0] = jsonObject.get('gcps')['gcpList'][2]['x']
        bounds[2,1] = jsonObject.get('gcps')['gcpList'][2]['y']
        bounds[3,0] = jsonObject.get('gcps')['gcpList'][3]['x']
        bounds[3,1] = jsonObject.get('gcps')['gcpList'][3]['y']
    elif 'cornerCoordinates' in jsonObject:
        bounds[0,:] = jsonObject['cornerCoordinates']['upperLeft']
        bounds[1,:] = jsonObject['cornerCoordinates']['upperRight']
        bounds[2,:] = jsonObject['cornerCoordinates']['lowerRight']
        bounds[3,:] = jsonObject['cornerCoordinates']['lowerLeft']
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
    x3 = np.floor( normRow*rpcInformation['LINESCALE']+rpcInformation['LINEOFFSET'])
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
    invCornerPoints = np.zeros(cornerPoints.shape)
    invCornerPoints[0,:] = inverseLookup(cornerPoints[0,:],(x0,y0)+(-1,1),rpcInformation)
    invCornerPoints[1,:] = inverseLookup(cornerPoints[1,:],(x0,y1)+(1,1),rpcInformation)
    invCornerPoints[2,:] = inverseLookup(cornerPoints[2,:],(x1,y1)+(1,-1),rpcInformation)
    invCornerPoints[3,:] = inverseLookup(cornerPoints[3,:],(x1,y0)+(-1,-1),rpcInformation)
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
        x = invCornerPoints[i,0].item()
        y = invCornerPoints[i,1].item()
        z = 0;

        gcp = gdal.GCP(x, y, z, pixel, line)
        gcp_list.append(gcp)
    #polygon = Polygon(cornerPoints)  
    #for i in range(0,25):
    a = 1#np.random.uniform(.8,1,1)[0]
    b = 1#np.random.uniform(.8,1,1)[0]
    x = (a*invCornerPoints[0,0]+(2-a)*invCornerPoints[2,0])/2
    y = (b*invCornerPoints[0,1]+(2-b)*invCornerPoints[2,1])/2
        #point = Point(x,y)
        #if(polygon.contains(point)):
    line, pixel = getCorrespondingPixel((y,x),rpcInformation)
    z = 0
    gcp = gdal.GCP(x, y, z, pixel, line)
    gcp_list.append(gcp)
    # Generating the Geo Transform:
    gt2 = gdal.GCPsToGeoTransform(gcp_list)
    fwd = Affine.from_gdal(*gt2)
    long, lat= fwd*plotCoordinates
    
    
    cornerPoints3 = np.zeros((4,2))
    cornerPoints3[0,:] = [long[0,0],lat[0,0]]
    cornerPoints3[3,:] = [long[0,-1],lat[0,-1]]
    cornerPoints3[2,:] = [long[-1,-1],lat[-1,-1]]
    cornerPoints3[1,:] = [long[-1,0],lat[-1,0]]
    transformMatrix = solve_affine(cornerPoints3, invCornerPoints);
    
    input = np.vstack((long.flatten(), lat.flatten(), np.ones((1,np.prod(long.shape)))))
    updatedPerspective = transformMatrix*input;
    long = updatedPerspective[0].reshape(long.shape)
    lat = updatedPerspective[1].reshape(lat.shape)
    print("error",np.sum(pow(cornerPoints3-invCornerPoints,2)))
    
 
    return long, lat, inputPatch

def solve_affine(x, y):
    x = np.transpose(np.matrix([x[0,:],x[1,:],x[2,:],x[3,:]]))
    y = np.transpose(np.matrix([y[0,:],y[1,:],y[2,:],y[3,:]]))
    # add ones on the bottom of x and y
    x = np.matrix(np.vstack((x,[1,1,1,1])))
    y = np.matrix(np.vstack((y,[1,1,1,1])))
    # solve for A2
    A2 = y * x.I
    # return function that takes input x and transforms it
    # don't need to return the 4th row as it is 
    return A2# lambda x: (A2*np.vstack((np.matrix(x).reshape(3,1),1)))[0:3,:]

def shp_matrix(shapefile_name, boundingBox,x0, y0,x1, y1, rpcInformation, image=None):
    """
    Function to convert a shape file to png
    """
    #shapefile_name = 'D:/Core3D/development/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_roads.shp'
    #shapefile_name = 'D:/Core3D/Data/jacksonville/open_street_map/SHP/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_buildings.shp'
    r = shapefile.Reader(shapefile_name)
    
    width = int(x1-x0)
    height = int(y1-y0)
    if image is None:
        im = Image.new('RGBA', (width, height), (0, 0, 0, 0)) 
    else:
        im = Image.fromarray(np.uint8(image))
        
    draw = ImageDraw.Draw(im)
    #NULL = 0
    #POINT = 1
    #POLYLINE = 3
    #POLYGON = 5
    #MULTIPOINT = 8
    #POINTZ = 11
    #POLYLINEZ = 13
    #POLYGONZ = 15
    #MULTIPOINTZ = 18
    #POINTM = 21
    #POLYLINEM = 23
    #POLYGONM = 25
    #MULTIPOINTM = 28
    #MULTIPATCH = 31
    for shape in r.shapes():
        if shape.shapeType == 3:
            for index in range(1,len(shape.points)):
                plotPoints = []
                #if polygon.contains(Point(shape.points[index-1][0],shape.points[index-1][1])) or polygon.contains(Point(shape.points[index][0],shape.points[index][1])):
                pixelLocation = getCorrespondingPixel((shape.points[index-1][1],shape.points[index-1][0]),rpcInformation)
                pixelLocation = ((pixelLocation[0]-x0),(pixelLocation[1] - y0)) 
                plotPoints.append(pixelLocation)
                pixelLocation = getCorrespondingPixel((shape.points[index][1],shape.points[index][0]),rpcInformation)
                pixelLocation = ((pixelLocation[0]-x0),(pixelLocation[1] - y0)) 
                plotPoints.append(pixelLocation)
                draw.line(plotPoints, fill=(255,255,255,255))

        if shape.shapeType == 5:
            #inBox = False
            #for long,lat in shape.points:
            #    if polygon.contains(Point(long,lat)):
            #        inBox = True
            #if inBox == True:
            plotPoints = []
            for pointCounter in range(0,len(shape.points)):
                pixelLocation = getCorrespondingPixel((shape.points[pointCounter][1],shape.points[pointCounter][0]),rpcInformation);
                pixelLocation = (pixelLocation[0]-x0,pixelLocation[1] - y0) 
                plotPoints.append(pixelLocation)
            draw.polygon(plotPoints, fill=(255,255,255,255))
        


    
    return np.array(im)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def shapeAlignImage(image, shapeImage):
    constraints = {}
    constraints['angle'] = (0,.1)
    constraints['scale'] = (1,.001)
    constraints['tx'] = (0,5)
    constraints['ty'] = (0,5)
    if image.shape[2] > 1:
        result = ird.similarity(rgb2gray(shapeImage),rgb2gray(image[:,:,[4,2,1]]/3), numiter=10, constraints = constraints)
    else:
        result = ird.similarity(rgb2gray(shapeImage),np.squeeze(image/3), numiter=10, constraints = constraints)
        #ird.translation(rgb2gray(shapeImage),rgb2gray(image[:,:,[4,2,1]]/3))
    #tvec = result["tvec"].round(4)
    returnResult = np.zeros(image.shape)
    if len(image.shape) == 3:
        for index in range(0,image.shape[2]):
            im = ird.transform_img(image[:,:,index], scale = result['scale'], angle = result['angle'], tvec = (result['tvec'][0],result['tvec'][1]), order=3)
            if image.dtype == np.dtype('uint16'):
                returnResult [:,:,index] = np.uint16(im)
            elif image.dtype == np.dtype('uint8'):
                returnResult [:,:,index] = np.uint16(im)
            else:
                returnResult [:,:,index] = im
    else:
        im = ird.transform_img(image[:,:], scale = result['scale'], angle = result['angle'], tvec = (result['tvec'][0],result['tvec'][1]), order=3)
        if image.dtype == np.dtype('uint16'):
            returnResult [:,:] = np.uint16(im)
        elif image.dtype == np.dtype('uint8'):
            returnResult [:,:] = np.uint16(im)
        else:
            returnResult [:,:] = im
    return returnResult, result['scale'], result['angle'], result['tvec'] 
#elevationMatrix, scale, angle,tvec
def shiftImage(image, scale, angle, tvec):
    returnResult = np.zeros(image.shape)
    if len(image.shape) == 3:
        for index in range(0,image.shape[2]):
            x = np.arange(0, image.shape[1])
            y = np.arange(0, image.shape[0])
            #mask invalid values
            array = np.ma.masked_invalid(image[:,:,index])
            xx, yy = np.meshgrid(x, y)
            #get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            temp = array[~array.mask]

            result = interpolate.griddata((x1, y1), temp.ravel(),(xx, yy),method='nearest')
            im = ird.transform_img(result.data, scale = scale, angle = angle, tvec = tvec, order=3)

            if image.dtype == np.dtype('uint16'):
                returnResult [:,:,index] = np.uint16(im)
            elif image.dtype == np.dtype('uint8'):
                returnResult [:,:,index] = np.uint8(im)
            else:
                returnResult [:,:,index] = im
    else:
        x = np.arange(0, image.shape[1])
        y = np.arange(0, image.shape[0])
        #mask invalid values
        array = np.ma.masked_invalid(image)
        xx, yy = np.meshgrid(x, y)
        #get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        temp = array[~array.mask]

        result = interpolate.griddata((x1, y1), temp.ravel(),(xx, yy),method='nearest')
        im = ird.transform_img(result.data, scale = scale, angle = angle, tvec = tvec, order=3)
        if image.dtype == np.dtype('uint16'):
            returnResult [:,:] = np.uint16(im)
        elif image.dtype == np.dtype('uint8'):
            returnResult [:,:] = np.uint8(im)
        else:
            returnResult [:,:] = im

    return returnResult

def getZMap(plotPatch, x0, y0, pointLat, pointLong, elevation, rpcInformation):
    zMap = np.empty((plotPatch.shape[0], plotPatch.shape[1]))
    zMap[:] = np.nan
    for entry in range(0,len(pointLat)):
        row, col = getCorrespondingPixel((pointLat[entry], pointLong[entry]),rpcInformation)
        if math.isnan(zMap[int(col-y0),int(row-x0)]):
            zMap[int(col-y0),int(row-x0)] = elevation[entry]
        else:
            zMap[int(col-y0),int(row-x0)] = np.maximum(zMap[int(col-y0),int(row-x0)],elevation[entry])
    
    return zMap
    

