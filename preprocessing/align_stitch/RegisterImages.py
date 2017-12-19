import logging
import os
import tempfile

import shutil
from osgeo import gdal
from rsgislib import imageregistration
import imreg_dft as ird

logger = logging.getLogger(__name__)


class RegisterImages:
    def __init__(self, reference):
        self.reference = reference

        self.pixel_gap = 1000
        self.threshold = 0.4
        self.window = 100
        self.search = 50
        self.stddevRef = 1      # higher number causes removal for low standard deviation?
        self.stddevFloat = 1
        self.sub_pixel_resolution = 4
        self.metric = imageregistration.METRIC_CORELATION
        self.output_type = imageregistration.TYPE_RSGIS_IMG2MAP
        self.distance_threshold = 50
        self.max_iterations = 10
        self.movement_threshold = 500
        self.smoothness = 2
        self.resolution = 2.5e-6
        self.gdal_format = 'GTiff'
        self.poly_order = 3
        self.interpolation = 'lanczos'

        ds = gdal.Open(self.reference)
        self.reference_band = ds.GetRasterBand(1).ReadAsArray()
        del ds

    def create_wkt(self, file):
        with(open(file,"w")) as f:
            f.write('GEOGCS["WGS 84",')
            f.write('DATUM["WGS_1984",')
            f.write('SPHEROID["WGS 84", 6378137, 298.257223563,')
            f.write('AUTHORITY["EPSG", "7030"]],')
            f.write('AUTHORITY["EPSG", "6326"]],')
            f.write('PRIMEM["Greenwich", 0],')
            f.write('UNIT["degree", 0.0174532925199433],')
            f.write('AUTHORITY["EPSG", "4326"]]')

    def extract_band(self, in_file, band, out_file):
        cmd = 'gdal_translate -b {} {} {}'.format(band, in_file, out_file)
        logging.info(cmd)
        return os.system(cmd)

    def trans_geotiff_tile(self, in_path, out_path, trans):
        '''
        Translates a geotiff tile by amount trans (in pixels)
        and saves to new geotiff
        '''
        shutil.copyfile(in_path, out_path)

        ds = gdal.Open(out_path, gdal.GA_Update)
        # get the geotransform as a tuple of 6
        gt = ds.GetGeoTransform()
        # unpack geotransform into variables
        x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = gt

        # compute shift of 1 pixel RIGHT in X direction
        shift_x = trans[1] * x_res
        # compute shift of 2 pixels UP in Y direction
        # y_res likely negative, because Y decreases with increasing Y index
        shift_y = trans[0] * y_res

        # make new geotransform
        gt_update = (x_tl + shift_x, x_res, dx_dy, y_tl + shift_y, dy_dx, y_res)
        # assign new geotransform to raster
        ds.SetGeoTransform(gt_update)
        # ensure changes are committed
        ds.FlushCache()
        del ds

    def register_image(self, file_in, file_out):
        if file_in == self.reference:
            logging.warning("Asked to register reference image {}, skipping...".format(file_in))
            return
        if os.path.exists(file_out):
            logging.info("File exists {}, skipping...".format(file_out))
            return

        ds = gdal.Open(file_in)
        md = ds.GetMetadata()
        del ds

        v_dir = tempfile.mkdtemp()

        sensor = md['NITF_PIAIMC_SENSNAME']
        type = md['NITF_IREP']

        band = 1
        file_to_register = file_in
        if type is not None and type == 'MULTI':
            file_to_register = os.path.join(v_dir, os.path.splitext(os.path.basename(file_in))[0] + '_mono.tif')
            if sensor == 'WV03':
                band = 6
            elif sensor == 'WV02':
                band = 6
            self.extract_band(file_in, band, file_to_register)

        ds1 = gdal.Open(file_in)
        im1 = ds1.GetRasterBand(band).ReadAsArray()

        result = ird.translation(self.reference_band, im1)
        self.trans_geotiff_tile(file_in, file_out, result['tvec'])
        print("{} translated by {}".format(file_in, result['tvec']))

        # tvec = result['tvec']
        #
        #
        #
        # print("Processing {}".format(file_in))
        # file_gcp = os.path.join(v_dir, os.path.splitext(os.path.basename(file_to_register))[0] + '.gcp')
        #
        # imageregistration.singlelayerregistration(self.reference, file_to_register, self.pixel_gap, self.threshold, self.window,
        #                                           self.search, self.stddevRef, self.stddevFloat, self.sub_pixel_resolution,
        #                                           self.distance_threshold, self.max_iterations, self.movement_threshold,
        #                                           self.smoothness, self.metric, self.output_type, file_gcp)
        #
        # wkt = os.path.join(v_dir, 'wkt.txt')
        # self.create_wkt(wkt)
        #
        # try:
        #     imageregistration.polywarp(file_in, file_gcp, file_out, wkt, self.resolution, self.poly_order, self.gdal_format)
        # except:
        #     logging.error("Unable to warp {}, skipping...", file_in)

        shutil.rmtree(v_dir)