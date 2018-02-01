import argparse
import glob
import logging
import os
import shutil
import yaml

from osgeo import gdal
import tempfile

from preprocessing.align_stitch import msi_to_rgb, path_util
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

    def add_vector(self, files, cat, name, value=None):
        """
        Add a vector set to raster.
        :param files: Array of files to combine into a single raster
        :param cat: Category name
        :param name: Name of resulting raster without a path
        :param value: if specified it will be the value burned into the raster, otherwise the id will be used.
        """
        out_folder = os.path.join(self.cache_dir, cat)
        os.makedirs(out_folder, exist_ok=True)

        c_dir = os.path.join(self.cache_dir, cat)
        vrt = os.path.join(c_dir, name + '.vrt')
        idx = 0
        t_files = []
        for f in files:
            t_tif = os.path.join(out_folder, name + '_' + str(idx) + '.tif')
            t_files.append(t_tif)
            idx = idx + 1
            if not os.path.exists(t_tif):
                if value is None:
                    os.system('gdal_rasterize -of GTiff -at -a id -ot UInt32 -tr {} {} -co "COMPRESS=LZW" {} {}'.format(
                               3e-6, 3e-6, f, t_tif))
                else:
                    os.system('gdal_rasterize -burn {} -of GTiff -at -a id -ot Byte -tr {} {} -a_nodata 0 -co "COMPRESS=LZW" {} {}'.format(value,
                               3e-6, 3e-6, f, t_tif))

        t_files_str = " ".join([str(i) for i in t_files])
        os.system('gdalbuildvrt -srcnodata 0 -vrtnodata 0 {} {}'.format(vrt, t_files_str))
        self.add_raster(vrt, cat, calibrate=False, interp=None, register=False)
        # self.output_tile_extents(tif, "*.tif")

    def calibrate_raster(self, file, out_folder, ext):
        os.makedirs(out_folder, exist_ok=True)
        out_path = path_util.derived_path(file, alt_dir=out_folder, alt_ext=ext)
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

    def add_pointcloud(self, file_pattern, cat, name, dimensions, statistics):
        """
        Build a stitched raster for the point cloud data matching the given pattern. The pointclouds are processed
        by PDAL. PDAL will produce rasters for specific dimensions of the PC data. The individual PC data typically
        represents a narrow strip of recon.  Each of the narrow raster strips are then composed into a single raster
        using gdal vrt.

        :param file_pattern:
        :param cat:
        :param name:
        :param dimensions:
        :param statistics:
        :return:
        """
        # The following yml file configuration controls how the rasters are produced. The file allows you to specify
        # the dimension to interpolate over (typically Z). Depending on how complete the PC data is, there may be more
        # dimensions available to choose from. Typical dimensions are X, Y, Z (default), Intensity, Classification, Red,
        # Green, Blue. You can choose the dimension by adding the dimension tag to the json.  For example:
        #
        # "dimension" : "Intensity"
        #
        # By default, a 6 band raster will be produced with the following defintions: (min, max, mean, idx, count,
        # stdev). The bands can be controlled by setting the output_type attribute to a comma separated list of
        # statistics for which to produce raster bands. The supported values are “min”, “max”, “mean”, “idw”,
        # “count”, “stdev” and “all”.individual_str

        for dim in dimensions.split(","):

            out = os.path.join(self.out_dir, cat, name + '_' + dim + '.tif')

            if os.path.exists(out):
                logger.info('Skipping raster of {}, file exists.'.format(out))
            else:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                tdir = tempfile.mkdtemp()
                jfile = os.path.join(tdir, 'laz.json')

                json = '{"pipeline": [ {"type": "readers.las", "filename": "dummy_in" },' + \
                       '{"type": "writers.gdal", "data_type": "float", "nodata": ' + str(self.f_nodata) + ', ' + \
                       '"resolution": 0.5, "radius": 1, "dimension":"' + dim + '", ' + \
                       '"output_type":"' + statistics + '", "filename": "dummy_out" }]}'

                with open(jfile, 'w') as f:
                    f.write(json)

                c_dir = os.path.join(self.cache_dir, cat)
                os.makedirs(c_dir, exist_ok=True)
                files = sorted(glob.glob(file_pattern))
                for file in files:
                    tif_file = path_util.derived_path(file, '_'+dim, alt_dir=c_dir, alt_ext='.tif')
                    tif_file_4326 = path_util.derived_path(file, '_'+dim+'_4326', alt_dir=c_dir, alt_ext='.tif')

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
                cmd = 'gdalbuildvrt {} {}'.format(vrt, os.path.join(self.cache_dir, cat, '*_'+dim+'_4326.tif'))
                os.system(cmd)

                cmd = 'gdal_translate {} {}'.format(vrt, out)
                os.system(cmd)
                shutil.rmtree(tdir)
                shutil.rmtree(c_dir)

            self.add_raster(out, cat, False, 'lanczos')

    def add_raster(self, file, cat, calibrate=True, interp=None, register=True):
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
        self.tm.add_raster(rast, cat, interp, register=register)

    def add_rasters(self, files, cat, calibrate=True, interp=None, register=True, name=None, srs=None):
        """
        Adds a set of rasters to the AOI.
        :param files: list of files to process
        :param cat:
        :param calibrate: True if to perform radiometric calibration; otherwise false.
        :param interp:
        :return:
        """
        individual = []
        for file in files:
            self.output_extents(cat, file)
            rast = file
            if calibrate:
                rast = self.calibrate_raster(file, os.path.join(self.cache_dir, cat), '_cal.tif')
            individual.append(rast)
            if name is None:
                self.tm.add_raster(rast, interp, register=register)
        if name is not None:
            c_dir = os.path.join(self.cache_dir, cat)
            os.makedirs(c_dir, exist_ok=True)

            individual_str = " ".join([str(i) for i in individual])

            rast = vrt = os.path.join(c_dir, name + '.vrt')
            if os.path.exists(vrt):
                logger.info('Skipping vrt {}, file exists.'.format(vrt))
            else:
                cmd = 'gdalbuildvrt {} {}'.format(vrt, os.path.join(self.cache_dir, cat, vrt, individual_str))
                os.system(cmd)

            if srs is not None:
                rast = tif = os.path.join(c_dir, name + '.tif')
                if os.path.exists(tif):
                    logger.info('Skipping warp {}, file exists.'.format(tif))
                else:
                    cmd = 'gdalwarp -t_srs {} {} {}'.format(srs, vrt, tif)
                    os.system(cmd)
            self.tm.add_raster(rast, interp, register=register)

    def filter_msi(self, files):
        result = []
        for f in files:
            if(os.path.exists(f)):
                ds = gdal.Open(f)
                md = ds.GetMetadata()
                del ds
                if 'NITF_IREP' in md and md['NITF_IREP'] == 'MULTI':
                    result.append(f)
        return result

    def filter_pan(self, files):
        result = []
        for f in files:
            if(os.path.exists(f)):
                ds = gdal.Open(f)
                md = ds.GetMetadata()
                del ds
                if 'NITF_IREP' in md and md['NITF_IREP'] != 'MULTI':
                    result.append(f)
        return result

    def pan_sharpen(self, pan, msi_list, pan_sharpened):
        if os.path.exists(pan_sharpened):
            logger.info('Skipping pan sharpen of {}, file exists.'.format(pan_sharpened))
        else:
            cmd = 'gdal_pansharpen.py {} -r lanczos '.format(pan)

            dataset = pan.split("-")[0]
            for f in msi_list:
                if(f.split("-")[0] == dataset):
                    cmd = cmd + '{} '.format(f)
                    cmd = cmd + pan_sharpened
                    os.system(cmd)
                    break

    def build_files_list(self, entry):
        result = []
        if isinstance(entry, list):
            for f in entry:
                result.extend(glob.glob(f))
        else:
            result.extend(glob.glob(entry))
        return result


if __name__ == "__main__":
    # Below is an example of how to use this AOI pipeline

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("aoi.log"),
            logging.StreamHandler()
        ])

    parser = argparse.ArgumentParser(description='AOI data processor.')
    parser.add_argument("-i", dest="input", required=True, type=path_util.arg_exist_file,
                        help="input config file", metavar="FILE")
    args = parser.parse_args()

    with open(args.input, 'r') as stream:
        try:
            cfg = yaml.load(stream)

            aoi = AOI(cfg['output'])

            if 'extents' in cfg:
                ext = cfg['extents']
                if ext == 'None':
                    ext = None
                aoi.set_output_extents(ext)

            if 'point_clouds' in cfg:
                for p in cfg['point_clouds']:
                    aoi.add_pointcloud(p['files'], p['dir'], p['name'], p['dimensions'], p['statistics'])

            if 'vectors' in cfg:
                for v in cfg['vectors']:
                    files = aoi.build_files_list(v['files'])
                    if v.get('type', 'value') == 'id':
                        aoi.add_vector(files, v['dir'], v['name'])
                    else:
                        aoi.add_vector(files, v['dir'], v['name'], v['burn'])

            if 'rasters' in cfg:
                for r in cfg['rasters']:
                    print(r)
                    calibrate = r.get('calibrate', True)
                    interp = r.get('interpolation', 'lanczos')
                    if interp == "None":
                        interp = None

                    register = r.get('register', True)
                    dir = r.get('dir', 'DIR')
                    files = aoi.build_files_list(r['files'])
                    name = r.get('name', None)
                    srs = r.get('srs', None)
                    aoi.add_rasters(files, dir, calibrate=calibrate, interp=interp, register=register, name=name,
                                    srs=srs)

            reg_to_pattern = None
            if 'register' in cfg:
                reg_to_pattern = cfg['register']

            if 'aoi' in cfg:
                x1, y1 = aoi.tm.deg_to_xy(cfg['aoi']['west'], cfg['aoi']['north'])
                x2, y2 = aoi.tm.deg_to_xy(cfg['aoi']['east'], cfg['aoi']['south'])

                for x in range(x1+1, x2):
                    for y in range(y1+1, y2):
                        # create the unregistered tiles - we are done processing these
                        aoi.tm.create_tiles_xy(x, y, False, border=1)

                        tile_files = []
                        tile_files.extend(aoi.tm.create_tiles_xy(x, y, True, border=100))

                        # Now attempt to register the images to a reference image.
                        # TODO: It currently only makes sense to register the
                        # reference image for a single region (x,y or lon, lat).

                        reg_tiles = []
                        for t in tile_files:
                            if reg_to_pattern is not None:
                                reg_to = glob.glob(os.path.join(os.path.dirname(t), reg_to_pattern))
                                if len(reg_to) <= 0:
                                    logger.warning("Registration image not found for {}".format(t))
                                    reg_tiles.append(t)
                                else:
                                    reg = RegisterImage(reg_to[0])
                                    tile_out = path_util.derived_path(t, '_reg')
                                    if reg.register_image(t, tile_out):
                                        reg_tiles.append(tile_out)
                                    else:
                                        reg_tiles.append(t)
                            else:
                                reg_tiles.append(t)

                        cut_list = []
                        for t in reg_tiles:
                            cut_list.extend(aoi.tm.create_tiles_xy2(x, y, t, 'cut', border=1))

                        pan_list = aoi.filter_pan(cut_list)
                        msi_list = aoi.filter_msi(cut_list)

                        for p in pan_list:
                            ps_out = path_util.derived_path(p, '_ps')
                            rgb_out = path_util.derived_path(p, '_ps_rgb')
                            aoi.pan_sharpen(p, msi_list, ps_out)
                            try:
                                msi_to_rgb.msi_to_rgb(ps_out, rgb_out)
                            except AttributeError:
                                logging.error("Not enough info to construct rgb {}".format(rgb_out))

        except yaml.YAMLError as exc:
            print(exc)