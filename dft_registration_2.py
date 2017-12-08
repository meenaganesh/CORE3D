# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:29:45 2017

@author: Peter

"""
import os
import imreg_dft as ird
import numpy as np
from osgeo import gdal
from shutil import copyfile
import scipy as sp

def read_geotif_into_array(tif):
    '''
    Reads first raster band of tif file into a uint16
    numpy array and closes the tif file
    '''
    
    ds = gdal.Open(tif)
    arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    ds.FlushCache()
    ds = None
    return arr

def read_building_tif_into_array(bldg):
    '''
    Reads building tif into numpy array
    '''
    arr = sp.misc.imread(bldg, True)# .astype(np.uint16)
    return arr


def comp_trans(src,target):
    '''
    Compute translation of src array to target array
    
    Inputs:
        src: numpy array representing source tile
        target: numpy array representing target tile
    Outputs:
        tvec: translation vector measured in pixels
    '''
    
    result = ird.translation(src,target)
    tvec = result['tvec']
    
    return tvec

def trans_geotiff_tile(in_path,out_path,tile,trans):
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

def align_row_col_tiles(in_path,out_path,row,col):
    '''
    Reads tiles with given row and column
    Translates wv2 and wv3 based on amounts derived from pan images
    Saves translated files in out_path
    '''
    
    path=in_path
    row_col_str=str('%02d' % row)+str('_')+str('%02d' % col)
    
    files = []
    for ii in os.listdir(path):
        if os.path.isfile(os.path.join(path,ii)) and row_col_str in ii:
            files.append(ii)
    print(files)

    num_files=len(files)
    target_filename = []
    for ii in range(num_files):
        if 'buildings' in files[ii]:
            target_filename.append(files[ii])
    print(target_filename)

    if len(target_filename) == 1:
        tar_filename = target_filename[0]
        target_arr=read_building_tif_into_array(in_path+tar_filename)
        for ii in range(num_files):
            file_name=files[ii].split('.')[0]
            file_info=file_name.split('_')
            tile_type = file_info[:-2]
            if ((tile_type==['wv2','pan']) or (tile_type==['wv3','pan'])):
                src_filename=files[ii]
                src_arr=read_geotif_into_array(in_path+src_filename)
                trans_vec=comp_trans(src_arr,target_arr)
                trans_geotiff_tile(in_path,out_path,src_filename,trans_vec)
                src_filename_2=src_filename.replace('pan','msi')
                trans_geotiff_tile(in_path,out_path,src_filename_2,trans_vec)
                print(tile_type[0]+' msi and pan files translated.')
    else:
        print('No building file found for alignment. No files modified.')           
            
    return()

# Paths to tiles
in_folder='/PROJECTS/CORE3D/05_python_dev/input_tif/'
out_folder='/PROJECTS/CORE3D/05_python_dev/output_tif/'
row=3
col=14

align_row_col_tiles(in_folder,out_folder,row,col)