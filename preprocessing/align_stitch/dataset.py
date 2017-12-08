import glob
import logging
import os
import shutil
from osgeo import gdal,osr
import tempfile
from core3d.preprocessing.WVCalKernel import RadiometricCalibrator


logger = logging.getLogger(__name__)


class RasterSet:
    def __init__(self, name, folder, pattern, geo, nodata=0):
        self.name = name
        self.folder = folder
        self.pattern = pattern
        self.geo = geo
        self.nodata = nodata


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
        self.f_nodata = 9999
        self.out_ext = None

    def set_output_extents(self, filename):
        self.out_ext = filename

    def output_extents(self, name, file):
        if self.out_ext:
            ds = gdal.Open(file)
            with open(self.out_ext, 'a+') as f:
                geo = ds.GetGeoTransform()
                ulx = geo[0]
                uly = geo[3]
                lrx = ulx + ds.RasterXSize * geo[1]
                lry = uly + ds.RasterYSize * geo[5]
                min_x = min(ulx, lrx)
                max_x = max(ulx, lrx)
                min_y = min(uly, lry)
                max_y = max(uly, lry)
                # (minx, miny, maxx, maxy)
                line = '{} {} {} {} {} {} {} {} {} {}\n'.format(name, file, ds.RasterXSize, ds.RasterYSize, ds.RasterCount,
                                                           min_x, min_y, max_x, max_y, str(geo))
                f.write(line)

    def output_tile_extents(self, folder, pattern):
        if self.out_ext:
            files = sorted(glob.glob(os.path.join(folder, pattern)))
            for file in files:
                self.output_extents('tile',file)

    def get_te(self):
        # TODO - for different hemispheres we need to reorder this as
        return str(self.ul_x) + ' ' + str(self.lr_y) + ' ' + str(self.lr_x) + ' ' + str(self.ul_y)

    def create_tiles(self, target_folder):
        os.makedirs(target_folder, exist_ok=True)
        self.create_vector_tiles(target_folder)
        self.create_raster_tiles(target_folder)

    def create_raster_tiles(self, target_folder):
        for r in self.rasters:
            # It names the rasters the same name as the vrt - so we will create a temporary dir to be able to name
            # the vrt file without worrying about collision
            v_dir = tempfile.mkdtemp()
            vrt = os.path.join(v_dir, r.name + '.vrt')
            x = self.get_te()
            os.system('gdalbuildvrt {} -tr {} {} -te {} -srcnodata {} -vrtnodata {} {}'.format(
                vrt, self.res_x, self.res_y, self.get_te(), r.nodata, r.nodata, os.path.join(r.folder, r.pattern)))
            os.system('gdal_retile.py -ps {} {} -co "TILED=yes" -co "INTERLEAVE=PIXEL" -targetDir {} {}'.format(
                self.tile_x, self.tile_y, target_folder, vrt))
            shutil.rmtree(v_dir)
        self.output_tile_extents(target_folder, "*.tif")

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
        self.output_tile_extents(target_folder, "*.tif")

    def calibrate_raster_set(self, folder, sub_folder, pattern, out_dir, ext):
        src_folder = os.path.join(folder, sub_folder)
        out_folder = os.path.join(out_dir, sub_folder)
        os.makedirs(out_folder, exist_ok=True)

        files = sorted(glob.glob(os.path.join(src_folder, pattern)))
        for file in files:
            out_path = os.path.join(out_folder, os.path.basename(file) + ext)
            if os.path.exists(out_path):
                logger.info('Skipping calibration for {}, file exists.'.format(out_path))
            else:
                raster = RadiometricCalibrator(file)
                raster.calibrate()
                raster_array, src_ds = raster.get_calibrated_data()

                gdal.Warp(out_path, src_ds, format='GTiff', rpc=True, multithread=True, resampleAlg='cubic',
                          outputType=gdal.GDT_UInt16)
                logger.info('Calibrated {}.'.format(out_path))

    def add_las_set(self, name, folder, pattern):
        tdir = tempfile.mkdtemp()
        jfile = os.path.join(tdir,'laz.json')

        json = '{"pipeline": [ {"type": "readers.las", "filename": "dummy_in" },' + \
               '{"type": "writers.gdal", "data_type": "float", "nodata": '+str(self.f_nodata)+', ' +\
               '"resolution": 0.5, "radius": 1, "filename": "dummy_out" }]}'

        with open(jfile, 'w') as f:
            f.write(json)

        files = sorted(glob.glob(os.path.join(folder, pattern)))
        for file in files:

            tif_file = file + '.tif'
            tif_proj_file = file + '_4326.tif'

            if os.path.exists(tif_proj_file):
                logger.info('Skipping raster projection {}, file exists.'.format(tif_proj_file))
            else:
                if os.path.exists(tif_file):
                    logger.info('Skipping raster {}, file exists.'.format(tif_file))
                else:
                    exe = 'pdal pipeline -i {} --readers.las.filename={} --writers.gdal.filename={}'\
                        .format(jfile, file, tif_file)
                    os.system(exe)

                exe = 'gdalwarp -t_srs EPSG:4326 {} {}'.format(tif_file, tif_proj_file)
                os.system(exe)
                os.unlink(tif_file)

        self.add_raster_set(name, folder, '*4326.tif', self.f_nodata)
        shutil.rmtree(tdir)

    def add_raster_set(self, name, folder, pattern, nodata=0):
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        for file in files:
            self.output_extents(name, file)

        tdir = tempfile.mkdtemp()
        vrt = os.path.join(tdir,name+'.vrt')
        exe = 'gdalbuildvrt {} {}'.format(vrt, os.path.join(folder,pattern))
        os.system(exe)
        ds = gdal.Open(vrt)
        self.size_x = ds.RasterXSize
        self.size_y = ds.RasterYSize
        geo = ds.GetGeoTransform()
        shutil.rmtree(tdir)
        self.rasters.append(RasterSet(name, folder, pattern, geo, nodata))

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


if __name__ == "__main__":
    ds = DataSet()

    out_dir = '/home/sernamlim/data/CORE3D/jacksonville/output/'
    dir = '/home/sernamlim/data/CORE3D/jacksonville/satellite_imagery/'
    shape_file = '/home/sernamlim/data/CORE3D/jacksonville/open_street_map/SHP/ex_FgMKU8FtfzgKJNgmUTE7T3Y5E1cgb_osm_buildings.shp'
    ds.set_output_extents(os.path.join(out_dir,'ext.txt'))

    ds.add_las_set('laz', '/home/sernamlim/data/CORE3D/jacksonville/pc/Vricon_Point_Cloud/data','*.laz')

    ds.calibrate_raster_set(dir,'WV2/MSI', "*.NTF", out_dir, '_cal.tif')
    ds.calibrate_raster_set(dir,'WV2/PAN', "*.NTF", out_dir, '_cal.tif')
    ds.calibrate_raster_set(dir,'WV3/SWIR', "*.NTF", out_dir, '_cal.tif')
    ds.calibrate_raster_set(dir,'WV3/PAN', "*.NTF", out_dir, '_cal.tif')
    ds.calibrate_raster_set(dir,'WV3/MSI', "*.NTF", out_dir, '_cal.tif')
    #
    ds.add_raster_set('wv2_msi',os.path.join(out_dir,'WV2/MSI'), "*.tif")
    ds.add_raster_set('wv2_pan',os.path.join(out_dir,'WV2/PAN'), "*.tif")
    ds.add_raster_set('wv3_swir',os.path.join(out_dir,'WV3/SWIR'),"*.tif")
    ds.add_raster_set('wv3_pan',os.path.join(out_dir,'WV3/PAN'), "*.tif")
    ds.add_raster_set('wv3_msi',os.path.join(out_dir,'WV3/MSI'), "*.tif")
    ds.add_shape_set('buildings', shape_file, 255)

    ds.create_tiles('/home/sernamlim/data/CORE3D/jacksonville/output')

