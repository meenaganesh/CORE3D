import glob
import logging
import os
import shutil
from osgeo import gdal
import tempfile

from preprocessing.align_stitch.RegisterImage import RegisterImage
from preprocessing.align_stitch.TileMaker import TileMaker
from preprocessing.align_stitch.WVCalKernel import RadiometricCalibrator


logger = logging.getLogger(__name__)


class RasterSet:
    def __init__(self, name, folder, pattern, geo, nodata=0):
        self.name = name
        self.folder = folder
        self.pattern = pattern
        self.geo = geo
        self.nodata = nodata


class AOI:
    def __init__(self, out_dir, cache_dir=None):
        self.out_dir = out_dir
        if cache_dir is None:
            cache_dir = os.path.join(out_dir,".cache")
        self.cache_dir = cache_dir
        self.tm = TileMaker(out_dir)
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

    def add_vector(self, file, cat, name, value):
        out_folder = os.path.join(self.cache_dir, cat)
        os.makedirs(out_folder, exist_ok=True)

        tif = os.path.join(out_folder, name + '.tif')
        if os.path.exists(tif):
            logging.info("File {} already exists, skipping rasterization", tif)
        else:
            # ERROR 1: JPEGPreEncode:Strip/tile too large for JPEG
            os.system('gdal_rasterize -burn {} -ot Byte -tr {} {} -a_nodata 0 -co "COMPRESS=LZW" {} {}'.format(value,
                3e-6, 3e-6 , file, tif))

        self.add_raster(tif, cat, False, 'lanczos')
        self.output_tile_extents(tif, "*.tif")

    def calibrate_raster(self, file, out_folder, ext):
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, os.path.splitext(os.path.basename(file))[0] + ext)
        if os.path.exists(out_path):
            logger.info('Skipping calibration for {}, file exists.'.format(out_path))
        else:
            if RadiometricCalibrator.get_cal_params(file) is None:
                # There is no radiometric calibration for this file - just return tin input file
                out_path = file
            else:
                raster = RadiometricCalibrator(file)
                raster.calibrate()
                raster_array, src_ds = raster.get_calibrated_data()

                gdal.Warp(out_path, src_ds, format='GTiff', rpc=True, multithread=True, resampleAlg='cubic',
                          outputType=gdal.GDT_UInt16)
                logger.info('Calibrated {}.'.format(out_path))
        return out_path

    def basename_without_ext(self, file):
        name = os.path.basename(file)
        return os.path.splitext(os.path.basename(name))[0]

    def add_pointcloud(self, file_pattern, cat, name):
        """
        Build a stitched raster for the point cloud data matching the given pattern
        :param file_pattern:
        :param cat:
        :param name:
        :return:
        """
        out = os.path.join(self.cache_dir, cat, name + '.tif')
        if not os.path.exists(out):
            logger.info('Skipping raster of {}, file exists.'.format(out))

            tdir = tempfile.mkdtemp()
            jfile = os.path.join(tdir, 'laz.json')

            json = '{"pipeline": [ {"type": "readers.las", "filename": "dummy_in" },' + \
                   '{"type": "writers.gdal", "data_type": "float", "nodata": '+str(self.f_nodata)+', ' +\
                   '"resolution": 0.5, "radius": 1, "filename": "dummy_out" }]}'

            with open(jfile, 'w') as f:
                f.write(json)

            files = sorted(glob.glob(file_pattern))
            for file in files:
                fn = self.basename_without_ext(file)

                tif_file = os.path.join(self.cache_dir, cat, fn + '_4326.tif')
                tif_file_4326 = os.path.join(self.cache_dir, cat, fn + '_4326.tif')

                if os.path.exists(tif_file_4326):
                    logger.info('Skipping raster projection {}, file exists.'.format(tif_file_4326))
                else:
                    if os.path.exists(tif_file):
                        logger.info('Skipping raster {}, file exists.'.format(tif_file))
                    else:
                        exe = 'pdal pipeline -i {} --readers.las.filename={} --writers.gdal.filename={}'\
                            .format(jfile, file, tif_file)
                        os.system(exe)

                    exe = 'gdalwarp -t_srs EPSG:4326 {} {}'.format(tif_file, tif_file_4326)
                    os.system(exe)
                    os.unlink(tif_file)

            vrt = os.path.join(tdir, name + '.vrt')
            cmd = 'gdalbuildvrt {} {}'.format(vrt, os.path.join(self.cache_dir, cat, '*_4326.tif'))
            os.system(cmd)

            cmd = 'gdal_translate {} {}'.format(vrt, out)
            os.system(cmd)
            shutil.rmtree(tdir)

        self.add_raster(out, cat, False, 'lanczos')

    def add_raster(self, file, cat, calibrate=True, interp=None):
        """
        Adds individual raster to the AOI.
        :param file:
        :param cat:
        :param calibrate: True if to perform radiometric calibration; otherwise false.
        :param interp:
        :return:
        """
        self.output_extents(cat, file)
        rast = file
        if calibrate:
            rast = self.calibrate_raster(file, os.path.join(self.cache_dir, cat), '_cal.tif')
        self.tm.add_raster(rast, cat, interp)

    def add_rasters(self, file_pattern, cat, calibrate=True, interp=None):
        """
        Adds a set of rasters to the AOI.
        :param file_pattern:
        :param cat:
        :param calibrate: True if to perform radiometric calibration; otherwise false.
        :param interp:
        :return:
        """
        files = sorted(glob.glob(file_pattern))
        for file in files:
            self.output_extents(cat, file)
            rast = file
            if calibrate:
                rast = self.calibrate_raster(file, os.path.join(self.cache_dir, cat), '_cal.tif')
            self.tm.add_raster(rast, interp)

    def get_tile_maker(self):
        return self.tm


if __name__ == "__main__":
    # Below is an example of how to use this AOI pipeline

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("aoi.log"),
            logging.StreamHandler()
        ])

    # Construct the AOI object - and set the output directory
    ds = AOI('/raid/data/wdixon/output3')

    # ds.set_output_extents(os.path.join(out_dir,'ext.txt'))

    #ds.add_pointcloud('/raid/data/wdixon/jacksonville/pc/Vricon_Point_Cloud/data/*.laz','PC', 'jacksonville')
    #ds.add_vector('/raid/data/wdixon/jacksonville/open_street_map/newSHP/ex_6Nv2MxW21gh8ifRtwCb75mo8YRTjb_osm_buildings.shp', 'SHP', 'buildings', 255)
    #ds.add_raster('/raid/data/wdixon/jacksonville/pc/vricon_raster_50cm/classification/data/classification_4326.tif', 'CLS', False, None)
    ds.add_rasters('/raid/data/wdixon/jacksonville/satellite_imagery/WV3/MSI/*.NTF', 'WV3/MSI')
    ds.add_rasters('/raid/data/wdixon/jacksonville/satellite_imagery/WV3/PAN/*.NTF', 'WV3/PAN')

    tile_files = ds.get_tile_maker().create_tiles_xy(4476, 6743)

    reg = RegisterImage('/raid/data/wdixon/output3/4476_6743/4476_6743_27JAN15WV031100015JAN27160845-P1BS.tif')
    for tile_in in tile_files:
        tile_out = os.path.join(os.path.dirname(tile_in), os.path.splitext(os.path.basename(tile_in))[0] + '_cv2_reg.tif')
        # reg.register_image(tile_in, tile_out)
        reg.register_image2(tile_in, tile_out)


