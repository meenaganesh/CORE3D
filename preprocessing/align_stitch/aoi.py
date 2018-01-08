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

    def add_vector(self, file, cat, name, value):
        out_folder = os.path.join(self.cache_dir, cat)
        os.makedirs(out_folder, exist_ok=True)

        tif = os.path.join(out_folder, name + '.tif')
        if os.path.exists(tif):
            logging.info("File {} already exists, skipping rasterization".format(tif))
        else:
            # ERROR 1: JPEGPreEncode:Strip/tile too large for JPEG
            os.system('gdal_rasterize -burn {} -ot Byte -tr {} {} -a_nodata 0 -co "COMPRESS=LZW" {} {}'.format(value,
                3e-6, 3e-6 , file, tif))

        self.add_raster(tif, cat, False, 'lanczos')
        self.output_tile_extents(tif, "*.tif")

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
                c_dir = os.path.join(self.cache_dir, cat)
                tif_file = path_util.derived_path(file, alt_dir=c_dir, alt_ext='.tif')
                tif_file_4326 = path_util.derived_path(file, '_4326', alt_dir=c_dir, alt_ext='.tif')

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

            # Setting this will write the extent information for each of the added rasters, vectors and pointcloud data.
            # This file is appended to - so remove if you which to only contain data from this run.
            if 'extents' in cfg:
                ext = cfg['extents']
                if ext == 'None':
                    ext = None
                aoi.set_output_extents(ext)

            if 'point_clouds' in cfg:
                for p in cfg['point_clouds']:
                    aoi.add_pointcloud(p['files'], p['dir'], p['name'])

            if 'vectors' in cfg:
                for v in cfg['vectors']:
                    aoi.add_vector(v['files'], v['dir'], v['name'], v['burn'])

            if 'rasters' in cfg:
                for r in cfg['rasters']:
                    print(r)
                    calibrate = r.get('calibrate', True)
                    interp = r.get('interpolation', 'lanczos')
                    if interp == "None":
                        interp = None
                    aoi.add_rasters(r['files'], r['dir'], calibrate=calibrate, interp=interp)

            if 'aoi' in cfg:
                x1, y1 = aoi.tm.deg_to_xy(cfg['aoi']['west'], cfg['aoi']['north'])
                x2, y2 = aoi.tm.deg_to_xy(cfg['aoi']['east'], cfg['aoi']['south'])

                for x in range(x1 + 2, x2):
                    for y in range(y1 + 1, y2):
                        tile_files = []
                        tile_files.extend(aoi.tm.create_tiles_xy(x, y, border=100))

                        # Now attempt to register the images to a reference image.
                        # It currently only makes sense to register the
                        # reference image for a single region (x,y or lon, lat).

                        reg_tiles = []
                        for t in tile_files:
                            reg_to = glob.glob(os.path.join(os.path.dirname(t),
                                                            '*MANALIGN-P1BS_cal.tif'))  # '*27JAN15WV031100015JAN27160845-P1BS.tif'))
                            if reg_to is not None:
                                reg = RegisterImage(reg_to[0])
                                tile_out = path_util.derived_path(t, '_reg')
                                reg_tiles.append(tile_out)
                                reg.register_image(t, tile_out)

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


    # # Construct the AOI object - and set the output directory.
    # # - By default a .cache subdirectory will be created for intermediate files.  If you wish you can override the
    # # location of the cache by supplying a 2nd argument to the AOI object constructor.
    # # - To ensure you aren't using cached files - you may wish to remove the cache before running.
    # # - Files are generally not overwritten.  If the target file exists, the generation of the target will
    # # be skipped, and the previously existing target will be incorporated into the results.
    # aoi = AOI('/raid/data/wdixon/output3')
    #
    # # Setting this will write the extent information for each of the added rasters, vectors and pointcloud data.
    # # This file is appended to - so remove if you which to only contain data from this run.
    # aoi.set_output_extents('/raid/data/wdixon/output3/ext.txt')
    #
    # # Point cloud data will be merged into a single raster - in the specified sub-directory using the supplied name
    # # for the raster file.
    # aoi.add_pointcloud('/raid/data/wdixon/jacksonville/pc/Vricon_Point_Cloud/data/*.laz', 'PC', 'jacksonville')
    #
    # # A raster mask will be created from the vector data - in the specified sub-directory using the supplied name
    # # for the raster file. The last argument is the numerical value for the encountered shape.
    # aoi.add_vector('/raid/data/wdixon/jacksonville/open_street_map/newSHP/ex_6Nv2MxW21gh8ifRtwCb75mo8YRTjb_osm_buildings.shp', 'SHP', 'buildings', 255)
    #
    # # You can add a single raster that is not radiometrically calibrated or interpolated
    # aoi.add_raster('/raid/data/wdixon/jacksonville/pc/vricon_raster_50cm/classification/data/classification_4326.tif', 'CLS', False, None)
    #
    # aoi.add_raster('/raid/data/wdixon/output3/.cache/WV3/PAN/01MAY15MANALIGN-P1BS_cal.tif', 'WV3/PAN', False)
    #
    # # You can add rasters that will (by default) - be radiometrically calibrated
    # aoi.add_rasters('/raid/data/wdixon/jacksonville/satellite_imagery/WV3/MSI/*.NTF', 'WV3/MSI')
    # aoi.add_rasters('/raid/data/wdixon/jacksonville/satellite_imagery/WV3/PAN/*.NTF', 'WV3/PAN')
    #
    # # Now you can generate the tiles.  You mah choose to generate just a single region designated by its x,y quad tree
    # # reference...  This will generate a separate tile for each input file covering this region.
    # # tile_files = aoi.get_tile_maker().create_all_tiles()
    #
    # # tile_files = aoi.get_tile_maker().create_tiles_deg(-81.640245, 30.30948)
    #
    # x1,y1 = aoi.tm.deg_to_xy(-81.719766, 30.340642)
    # x2,y2 = aoi.tm.deg_to_xy(-81.62187, 30.32112)
    #
    # for x in range(x1+2,x2):
    #     for y in range(y1+1,y2):
    #         tile_files = []
    #         # for x in range(35809, 35810):
    #         #     for y in range(53946, 53948):
    #         #         tile_files.extend(aoi.get_tile_maker().create_tiles_xy(x, y, border=100))
    #
    #         tile_files.extend(aoi.tm.create_tiles_xy(x, y, border=100))
    #
    #
    #         #    4476, 6743
    #
    #
    #         # alternatively you can specify a point in lon, lat
    #         # tile_files = aoi.get_tile_maker().create_tiles_deg(-81.3, 30.0)
    #
    #         # Or you may choose to generate all tiles for all extents based on the supplied input:
    #         # tile_files = aoi.get_tile_maker().create_all_tiles()
    #
    #         # Now attempt to register the images to a reference image. It currently only makes sense to register the
    #         # reference image for a single region (x,y or lon, lat).
    #         #
    #         # Note currently the output images are ~8K x 8K
    #         # After registration, the images will be translated to the best match - in the future we may wish to generate
    #         # the images slightly larger and trim after registration.  But for now we aren't all that concerned with
    #         # stitching
    #         # reg = RegisterImage('/raid/data/wdixon/output3/35810_53946/35810_53946_27JAN15WV031100015JAN27160845-P1BS.tif')
    #
    #         reg_tiles = []
    #         for t in tile_files:
    #             reg_to = glob.glob(os.path.join(os.path.dirname(t), '*MANALIGN-P1BS_cal.tif')) # '*27JAN15WV031100015JAN27160845-P1BS.tif'))
    #             if reg_to is not None:
    #                 reg = RegisterImage(reg_to[0])
    #                 tile_out = path_util.derived_path(t, '_reg')
    #                 reg_tiles.append(tile_out)
    #                 reg.register_image(t, tile_out)
    #
    #         cut_list = []
    #         for t in reg_tiles:
    #             cut_list.extend(aoi.tm.create_tiles_xy2(x, y, t, 'cut', border=1))
    #
    #         pan_list = aoi.filter_pan(cut_list)
    #         msi_list = aoi.filter_msi(cut_list)
    #
    #         for p in pan_list:
    #             ps_out = path_util.derived_path(p, '_ps')
    #             rgb_out = path_util.derived_path(p, '_ps_rgb')
    #             aoi.pan_sharpen(p, msi_list, ps_out)
    #             try:
    #                 msi_to_rgb.msi_to_rgb(ps_out, rgb_out)
    #             except AttributeError:
    #                 logging.error("Not enough info to construct rgb {}".format(rgb_out))
    #
    #
    #



