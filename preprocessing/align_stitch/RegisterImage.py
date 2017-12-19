import logging
import os
import shutil
from osgeo import gdal
import imreg_dft as ird

logger = logging.getLogger(__name__)


class RegisterImage:
    def __init__(self, reference):
        self.reference = reference
        ds = gdal.Open(self.reference)
        self.reference_band = ds.GetRasterBand(1).ReadAsArray()
        del ds

    def trans_geotiff_tile(self, in_path, out_path, trans):
        """
        Translates a geotiff tile by amount trans (in pixels)
        and saves to new geotiff
        """
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

        if 'NITF_PIAIMC_SENSNAME' not in md:
            logging.info("Skipping registration of {}, skipping...".format(file_in))
        else:
            sensor = md['NITF_PIAIMC_SENSNAME']
            type = md['NITF_IREP']

            band = 1
            if type is not None and type == 'MULTI':
                if sensor == 'WV03':
                    band = 6
                elif sensor == 'WV02':
                    band = 6

            translate_band = ds.GetRasterBand(band).ReadAsArray()

            result = ird.translation(self.reference_band, translate_band)
            self.trans_geotiff_tile(file_in, file_out, result['tvec'])
            logging.info("{} translated by {}".format(file_in, result['tvec']))
        del ds

