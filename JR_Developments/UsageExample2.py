# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:17:42 2017

@author: 200021424
"""


from GISDataFunctions import helper_functions as hf
from GISDataFunctions import WVCalKernel as WVCK
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import glob
import re


examples = {}



# Getting point cloud information:
# Get all point cloud files for location:
pointCloudFiles = glob.glob('D:\Core3D\Data\Vricon_Point_Cloud\data\*.las')
#for index in range(0, len(pointCludFiles)):

shapeFiles = glob.glob('D:\Core3D\Data\jacksonville\open_street_map\SHP\*.shp')
 # for each point cloud, find Pan, SWIR and MSI information:   
pointCloudFile = pointCloudFiles[0]
pointLat, pointLong, elevation, rgbColor, cornerPoints = hf.getPointcloudLatLong(pointCloudFile) 

# for each point cloud, find Pan, SWIR and MSI information:
panFiles = glob.glob('D:\Core3D\Data\jacksonville\satellite_imagery\WV2\PAN\*.NTF')
panFiles.extend(glob.glob('D:\Core3D\Data\jacksonville\satellite_imagery\WV3\PAN\*.NTF'))

exampleCounter=0;
for panFile in panFiles:
    bounds = hf.getTileBounds(panFile)
    if hf.boundingInImage(pointCloudFile, bounds):
        exampleCounter += 1
        panData = hf.loadRasters(panFile)
        panRPCInformation = hf.loadRPC(panFile)
        croppedData, x0, y0, x1, y1 = hf.getPatch(cornerPoints, panData, panRPCInformation) 
        if panFile.find('\\WV3\\') == -1:
            wvX = WVCK.WV2params()
        else:
            wvX = WVCK.WV3params()
        calibratedData = WVCK.RadiometricCalibrator(panFile, wvX,sub_array = croppedData)
        calibratedData.calibrate()
        long, lat, plotPatch = hf.projectImage(calibratedData.get_calibrated_data(), x0, y0, x1, y1, cornerPoints, panRPCInformation)
        image = None
        exampleCollection = {}
        for file in shapeFiles:
            image = hf.shp_matrix(file, cornerPoints,x0, y0,x1, y1, panRPCInformation, image)
            result = re.search('_osm_(.*).shp',shapeFiles[0])
            exampleCollection[result.group(1)] = hf.shp_matrix(file, cornerPoints,x0, y0,x1, y1, panRPCInformation)
        plotPatch2, scale, angle, tvec =  hf.shapeAlignImage(plotPatch, image)
        elevationMatrix = hf.getZMap(plotPatch, x0, y0, pointLat, pointLong, elevation, panRPCInformation)
        elevationMatrix2 = hf.shiftImage(elevationMatrix, scale, angle,tvec)
        exampleCollection['elevation'] = elevationMatrix2
        exampleCollection['long'] = long
        exampleCollection['lat'] = lat
        result = re.search('PAN(.*)-',panFile)    
        examples['PAN_'+result.group(1).replace("\\","")] = exampleCollection
        if exampleCounter > 4:
            break



swirFiles = glob.glob('D:\Core3D\Data\jacksonville\satellite_imagery\WV2\SWIR\*.NTF')
swirFiles.extend(glob.glob('D:\Core3D\Data\jacksonville\satellite_imagery\WV3\SWIR\*.NTF'))
exampleCounter=0
for swirFile in swirFiles:
    bounds = hf.getTileBounds(swirFile)
    if hf.boundingInImage(pointCloudFile, bounds):
        exampleCounter += 1
        swirData = hf.loadRasters(swirFile)
        swirRPCInformation = hf.loadRPC(swirFile)
        croppedData, x0, y0, x1, y1 = hf.getPatch(cornerPoints, swirData, swirRPCInformation) 
        if swirFile.find('\\WV3\\') == -1:
            wvX = WVCK.WV2params()
        else:
            wvX = WVCK.WV3params()
        calibratedData = WVCK.RadiometricCalibrator(swirFile, wvX,sub_array = croppedData)
        calibratedData.calibrate()
        long, lat, plotPatch = hf.projectImage(calibratedData.get_calibrated_data(), x0, y0, x1, y1, cornerPoints, swirRPCInformation)
        image = None
        exampleCollection = {}
        for file in shapeFiles:
            image = hf.shp_matrix(file, cornerPoints,x0, y0,x1, y1, swirRPCInformation, image)
            result = re.search('_osm_(.*).shp',shapeFiles[0])
            exampleCollection[result.group(1)] = hf.shp_matrix(file, cornerPoints,x0, y0,x1, y1, swirRPCInformation)
        plotPatch2, scale, angle, tvec =  hf.shapeAlignImage(plotPatch, image)
        elevationMatrix = hf.getZMap(plotPatch, x0, y0, pointLat, pointLong, elevation, swirRPCInformation)
        elevationMatrix2 = hf.shiftImage(elevationMatrix, scale, angle,tvec)
        exampleCollection['elevation'] = elevationMatrix2
        exampleCollection['long'] = long
        exampleCollection['lat'] = lat        
        result = re.search('SWIR(.*)-',swirFile)    
        examples['SWIR_'+result.group(1).replace("\\","")] = exampleCollection
        if exampleCounter > 4:
            break

msiFiles = glob.glob('D:\Core3D\Data\jacksonville\satellite_imagery\WV2\MSI\*.NTF')
msiFiles.extend(glob.glob('D:\Core3D\Data\jacksonville\satellite_imagery\WV3\MSI\*.NTF'))
exampleCounter=0
for msiFile in msiFiles:
    bounds = hf.getTileBounds(msiFile)
    if hf.boundingInImage(pointCloudFile, bounds):
        exampleCounter += 1
        msiData = hf.loadRasters(msiFile)
        msiRPCInformation = hf.loadRPC(msiFile)
        croppedData, x0, y0, x1, y1 = hf.getPatch(cornerPoints, msiData, msiRPCInformation) 
        if msiFile.find('\\WV3\\') == -1:
            wvX = WVCK.WV2params()
        else:
            wvX = WVCK.WV3params()
        calibratedData = WVCK.RadiometricCalibrator(msiFile, wvX,sub_array = croppedData)
        calibratedData.calibrate()
        long, lat, plotPatch = hf.projectImage(calibratedData.get_calibrated_data(), x0, y0, x1, y1, cornerPoints, msiRPCInformation)
        image = None
        exampleCollection = {}
        for file in shapeFiles:
            image = hf.shp_matrix(file, cornerPoints,x0, y0,x1, y1, msiRPCInformation, image)
            result = re.search('_osm_(.*).shp',shapeFiles[0])
            exampleCollection[result.group(1)] = hf.shp_matrix(file, cornerPoints,x0, y0,x1, y1, msiRPCInformation)
        plotPatch2, scale, angle, tvec =  hf.shapeAlignImage(plotPatch, image)
        elevationMatrix = hf.getZMap(plotPatch, x0, y0, pointLat, pointLong, elevation, msiRPCInformation)
        elevationMatrix2 = hf.shiftImage(elevationMatrix, scale, angle,tvec)
        exampleCollection['elevation'] = elevationMatrix2
        exampleCollection['long'] = long
        exampleCollection['lat'] = lat        
        result = re.search('MSI(.*)-',msiFile)    
        examples['MSI_'+result.group(1).replace("\\","")] = exampleCollection
        if exampleCounter > 4:
            break











