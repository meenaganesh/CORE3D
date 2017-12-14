# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:29:45 2017

@author: Peter

"""

import scipy as sp
import imreg_dft as ird
from PIL import Image
import numpy as np
from osgeo import gdal
from shutil import copyfile

def tif_to_png(in_path,out_path,tif_image):
    '''
    Read tif image tile and save to png with equal number of pixels
    '''
    png_im=np.array(Image.open(in_path+tif_image))
    png_name=tif_image.split('.')[0]
    sp.misc.imsave(out_path+png_name+'.png',png_im)
    print('I saved a file.')
    return()


def compute_translation(path,src,target):
    '''
    Compute translation of src tile to target tile
    contained in /path/
    
    Inputs:
        path: folder containing tiles
        src: source tile in .png format
        target: target tile in .png format
    Outputs:
        tvec: translation vector measured in pixels
    '''
    im0 = sp.misc.imread(path+src, True)
    im1 = sp.misc.imread(path+target, True)
    
    result = ird.translation(im0,im1)
    tvec = result['tvec']
    
    return tvec

def translate_geotiff_tile(in_path,out_path,tile,trans):
    '''
    Translates a geotiff tile by amount trans (in pixels) 
    and saves to new geotiff
    '''
    tile_name=tile.split('.')[0]
    # create copy
    copyfile(in_path+tile, out_path+tile_name+'_trans.tif')
    # read in copy
    
    ds = gdal.Open(out_path+tile_name+'_trans.tif', gdal.GA_Update)
    # get the geotransform as a tuple of 6
    gt = ds.GetGeoTransform()
    # unpack geotransform into variables
    x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = gt

    # compute shift of 1 pixel RIGHT in X direction
    shift_x = -trans[1] * x_res
    # compute shift of 2 pixels UP in Y direction
    # y_res likely negative, because Y decreases with increasing Y index
    shift_y = -trans[0] * y_res
    
    # make new geotransform
    gt_update = (x_tl + shift_x, x_res, dx_dy, y_tl + shift_y, dy_dx, y_res)
    # assign new geotransform to raster
    ds.SetGeoTransform(gt_update)
    # ensure changes are committed
    ds.FlushCache()
    ds = None
    
    print('I translated a tile.')
    return()

# Paths to tiles and pngs
input_tiles_folder='/home/sernamlim/data/CORE3D/jacksonville/output/'
output_tiles_folder='/home/sernamlim/data/CORE3D/jacksonville/output/aligned/'
tiles_png_folder='/home/sernamlim/data/CORE3D/jacksonville/output/aligned/'

# Source and target of translation
pan_filename='wv2_pan_03_14.tif'
building_filename='buildings_03_14.tif'

# Convert tif to png for imreg-dft
tif_to_png(input_tiles_folder,tiles_png_folder,pan_filename)
tif_to_png(input_tiles_folder,tiles_png_folder,building_filename)

# Input files for registration algorithm 
pan_png_filename='wv2_pan_03_14.png'
building_png_filename='buildings_03_14.png'

# Compute translation
translation_vector=compute_translation(tiles_png_folder,
                    pan_png_filename,building_png_filename)

# Translate tile and save in output directory
translate_geotiff_tile(input_tiles_folder,output_tiles_folder,
                       pan_filename,translation_vector)
