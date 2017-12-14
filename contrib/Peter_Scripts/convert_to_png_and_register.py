# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:59:45 2017

@author: Peter
Convert shp and ntf to png and align with imreg_dft.

'Result' contains the transformation of the rescaled images.

To do:
    -include rescaling from mapping to png into result
    -hyperparameters

"""

import shapefile
import pngcanvas
import os
from PIL import Image
import PIL.ImageOps
import scipy as sp
import imreg_dft as ird
from osgeo import gdal
import numpy as np

def shp_to_png(shapefile_name,output_directory):
    """
    Function to convert a shape file to png
    """
    
    r = shapefile.Reader(shapefile_name)
    
    input_file_path, input_file_name = os.path.split(shapefile_name)
    input_file_name_prefix=str.split(input_file_name,'.')[0]
    
    # Determine bounding box x and y distances and then calculate an xyratio
    # that can be used to determine the size of the generated PNG file. A xyratio
    # of greater than one means that PNG is to be a landscape type image whereas
    # an xyratio of less than one means the PNG is to be a portrait type image.
    xdist = r.bbox[2] - r.bbox[0]
    ydist = r.bbox[3] - r.bbox[1]
    xyratio = xdist/ydist
#    xyratio = 1
    image_max_dimension = 1000 # Change this to desired max dimension of generated PNG
    if (xyratio >= 1):
        iwidth  = image_max_dimension
        iheight = int(image_max_dimension/xyratio)
    else:
        iwidth  = int(image_max_dimension/xyratio)
        iheight = image_max_dimension
    
    # Iterate through all the shapes within the shapefile and draw polyline
    # representations of them onto the PNGCanvas before saving the resultant canvas
    # as a PNG file
    xratio = iwidth/xdist
    yratio = iheight/ydist
    pixels = []
    c = pngcanvas.PNGCanvas(iwidth,iheight)
    for shape in r.shapes():
        for x,y in shape.points:
            px = int(iwidth - ((r.bbox[2] - x) * xratio))
            py = int((r.bbox[3] - y) * yratio)
            pixels.append([px,py])
        c.polyline(pixels)
        pixels = []
        
    f = open(output_directory+"%s.png" % input_file_name_prefix, "wb")
    f.write(c.dump())
    f.close()
    
    return input_file_name_prefix+'.png',np.array([xdist,ydist])

def invert_and_save_image(file_name,output_path):
    image=Image.open(output_path+file_name)
    image2=image.convert('L')
#    new_image=255-image
    inverted_image=PIL.ImageOps.invert(image2)
    inverted_image=inverted_image.convert('1')
    inverted_image.save(output_path+'inverted_'+file_name)
    return 'inverted_'+file_name

def make_square_and_save(file_name,output_path,length):  
    size=length,length
    image=Image.open(output_path+file_name)
    square_image=image.resize(size)
    square_image.save(output_path+'square_'+file_name)
    return()

def binarize_array(numpy_array, threshold):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array

def ntf_to_bw(input_path,filename,output_path,threshold):
    src_ntf=gdal.Open(input_path+filename)

    r_band=np.array(src_ntf.GetRasterBand(1).ReadAsArray())
    g_band=np.array(src_ntf.GetRasterBand(2).ReadAsArray())
    b_band=np.array(src_ntf.GetRasterBand(3).ReadAsArray())

    array=(np.maximum(r_band,g_band,b_band)+np.minimum(r_band,g_band,b_band))/2
    binarized_array=binarize_array(array, threshold)

    #rgbArray = np.zeros((r_band.shape[0],r_band.shape[1],3), 'uint8')
    #rgbArray[..., 0] = r_band
    #rgbArray[..., 1] = g_band
    #rgbArray[..., 2] = b_band
    #
    #im=Image.fromarray(rgbArray)
    #im=im.convert('L')

    im=Image.fromarray(binarized_array)
    im=im.convert('L')
    
    size=1000,1000
    im=im.resize(size)
    im=im.convert('1')
    output_filename=str.split(filename,'.')[0]+str('.png')
    im.save(output_path+output_filename)
    
    #Properly close the datasets to flush to disk
    src_ntf=None
    return r_band.shape

# Open shapefile with Python Shapefile Library
shapefile_name='C:/PROJECTS/CORE3D/06_data/jacksonville/open_street_map/SHP/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_roads.shp' # e.g. england_oa_2001
output_directory='C:/PROJECTS/CORE3D/05_python_dev/output/'

# Convert shp to png and save
png_file_name,shp_ratio=shp_to_png(shapefile_name,output_directory)
inverted_output_png_file=invert_and_save_image(png_file_name,output_directory)
make_square_and_save(inverted_output_png_file,output_directory,1000)

# Read in ntf file
ntf_path='C:/PROJECTS/CORE3D/06_data/jacksonville/satellite_imagery/WV2/MSI/'
ntf_name='05SEP16WV021200016SEP05162552-M1BS-500881026010_01_P006_________GA_E0AAAAAAIAAG0.NTF'
ntf_xyratio=ntf_to_bw(ntf_path,ntf_name,output_directory,300)

# Begin registration
# the template
im0=sp.misc.imread(output_directory+'05SEP16WV021200016SEP05162552-M1BS-500881026010_01_P006_________GA_E0AAAAAAIAAG0.png', True)
# the image to be transformed
im1=sp.misc.imread(output_directory+'square_inverted_ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_roads.png', True)

# Contains transformation of png files
result=ird.similarity(im0, im1, numiter=3)

# Activate to plot results
#assert "timg" in result
## Maybe we don't want to show plots all the time
#if os.environ.get("IMSHOW", "yes") == "yes":
#    import matplotlib.pyplot as plt
#    ird.imshow(im0, im1, result['timg'])
#    plt.show()