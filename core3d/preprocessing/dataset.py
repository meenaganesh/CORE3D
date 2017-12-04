import os,sys

import shutil
from osgeo import gdal,ogr
import tempfile


class RasterSet:
    def __init__(self, name, folder, pattern, geo):
        self.name = name
        self.folder = folder
        self.pattern = pattern
        self.geo = geo

class VectorSet:
    def __init__(self, name, file, value):
        self.name = name
        self.file = file
        self.value = value

class DataSet:
    def __init__(self):
        self.rasters = []
        self.vectors = []
        self.size_y = self.size_x = float('inf')
        self.ul_x = self.ul_y = float('inf')
        self.lr_x = self.lr_y = -float('inf')
        self.res_x = self.res_y = float('inf')
        self.tile_x = 4096
        self.tile_y = 4096

    def get_te(self):
        # TODO - for different hemispheres we need to reorder this as
        return str(self.ul_x) + ' ' + str(self.lr_y) + ' ' + str(self.lr_x) + ' ' + str(self.ul_y)

    def create_tiles(self, target_folder):
        os.makedirs(target_folder)
        self.create_vector_tiles(target_folder)
        self.create_raster_tiles(target_folder)

    def create_raster_tiles(self, target_folder):
        for r in self.rasters:
            # It names the rasters the same name as the vrt - so we will create a temporary dir to be able to name
            # the vrt file without worrying about collision
            v_dir = tempfile.mkdtemp()
            vrt = os.path.join(v_dir, r.name + '.vrt')
            x = self.get_te()
            os.system('gdalbuildvrt {} -tr {} {} -te {} -srcnodata 0 -vrtnodata 0 {}'.format(
                vrt, self.res_x, self.res_y, self.get_te(), os.path.join(r.folder, r.pattern)))
            os.system('gdal_retile.py -ps {} {} -co "TILED=yes" -co "INTERLEAVE=PIXEL" -targetDir {} {}'.format(
                self.tile_x, self.tile_y, target_folder, vrt))
            shutil.rmtree(v_dir)

    def create_vector_tiles(self, target_folder):
        for r in self.vectors:
            v_dir = tempfile.mkdtemp()
            vrt = os.path.join(v_dir, r.name + '.vrt')
            tif = os.path.join(v_dir, r.name + '.tif')
            # ERROR 1: JPEGPreEncode:Strip/tile too large for JPEG
            os.system('gdal_rasterize -burn {} -ot Byte -tr {} {} -co "COMPRESS=LZW" {} {}'.format(r.value,
                self.res_x, self.res_y, r.file, tif))
            os.system('gdalbuildvrt {} -te {} -srcnodata 0 -vrtnodata 0 {}'.format(
                vrt, self.get_te(), tif))
            os.system('gdal_retile.py -ps {} {} -co "TILED=yes" -co "COMPRESS=JPEG" -targetDir {} {}'.format(
                self.tile_x, self.tile_y, target_folder, vrt))
            shutil.rmtree(v_dir)

    def add_raster_set(self, name, folder, pattern):
        tdir = tempfile.mkdtemp()
        vrt = os.path.join(tdir,name+'.vrt')
        exe = 'gdalbuildvrt {} {}'.format(vrt, os.path.join(folder,pattern))
        os.system(exe)
        ds = gdal.Open(vrt)
        self.size_x = ds.RasterXSize
        self.size_y = ds.RasterYSize
        geo = ds.GetGeoTransform()
        shutil.rmtree(tdir)
        self.rasters.append(RasterSet(name, folder, pattern, geo))

        # calculate local extents
        ulx = geo[0]
        uly = geo[3]
        resx = geo[1]
        resy = abs(geo[5])
        lrx = ulx + self.size_x * resx
        lry = uly + self.size_y * geo[5]

        # maximize to determine overall extents
        self.ul_x = min(self.ul_x, ulx)
        self.ul_y = min(self.ul_y, uly)
        self.res_x = min(self.res_x, resx)
        self.res_y = min(self.res_y, resy)
        self.lr_x = max(self.lr_x, lrx)
        self.lr_y = max(self.lr_y, lry)

    def add_shape_set(self, name, file, value):
        self.vectors.append(VectorSet(name, file, value))



ds = DataSet()
dir = '/raid/data/wdixon/jacksonville/satellite_imagery/'
shape_file = '/raid/data/wdixon/jacksonville/open_street_map/SHP/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_buildings.shp'

ds.add_raster_set('wv2_msi',os.path.join(dir,'WV2/MSI'), "*.tif")
ds.add_raster_set('wv2_pan',os.path.join(dir,'WV2/PAN'), "*.tif")
ds.add_raster_set('wv3_swir',os.path.join(dir,'WV3/SWIR'),"*.tif")
ds.add_raster_set('wv3_pan',os.path.join(dir,'WV3/PAN'), "*.tif")
ds.add_raster_set('wv3_msi',os.path.join(dir,'WV3/MSI'), "*.tif")
ds.add_shape_set('buildings', shape_file, 255)

ds.create_tiles('/raid/data/wdixon/output')

