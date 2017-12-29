import logging
import os
import shutil

import cv2 as cv2
import numpy as np
from osgeo import gdal
import imreg_dft as ird

logger = logging.getLogger(__name__)


class RegisterImage:
    def __init__(self, reference):
        self.reference = reference
        ds = gdal.Open(self.reference)
        self.reference_band_f = self.read_stretch_min_max(ds, 1).astype(np.float32)
        self.reference_band = self.reference_band_f.astype(np.uint8)
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

    def read_stretch_min_max(self, ds, band, scale=255):
        arr = ds.GetRasterBand(band).ReadAsArray()
        mip = arr.min()
        mxp = arr.max()
        z = (arr - mip) / (mxp - mip) * scale
        return np.clip(z, 0, scale)

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

            if type == 'MULTI' and (sensor == 'WV03' or sensor == 'WV02'):
                R = self.read_stretch_min_max(ds, 5)
                G = self.read_stretch_min_max(ds, 3)
                B = self.read_stretch_min_max(ds, 2)
                translate_band = R * .299 + G * .587 + B * .114
            else:
                translate_band = self.read_stretch_min_max(ds, 1)

            # Define motion model
            warp_mode = cv2.MOTION_TRANSLATION

            # Set the warp matrix to identity.
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Set the stopping criteria for the algorithm.
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

            # Warp the blue and green channels to the red channel
            # (cc, warp_matrix) = cv2.findTransformECC(self.get_gradient(self.reference_band.astype(np.float32)),
            #                                          self.get_gradient(img.astype(np.float32)),
            #                                          warp_matrix, warp_mode, criteria)
            # Warp the blue and green channels to the red channel
            (cc, warp_matrix) = cv2.findTransformECC(self.reference_band_f, translate_band.astype(np.float32),
                                                     warp_matrix, warp_mode, criteria)
            logging.info("CV2 {} translated by {}".format(file_in, warp_matrix))
            tvec = [-warp_matrix[1, 2], -warp_matrix[0, 2]]  # y,x
            self.trans_geotiff_tile(file_in, file_out, tvec)

            # translate_band = translate_band.astype(np.uint8)
            # result = ird.translation(self.reference_band, translate_band)
            # self.trans_geotiff_tile(file_in, file_out, result['tvec'])
            # logging.info("{} translated by {}".format(file_in, result['tvec']))
        del ds

    def get_gradient(self,im):
        # Calculate the x and y gradients using Sobel operator
        grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

        # Combine the two gradients
        grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
        return grad

    def register_image2(self, file_in, file_out):

        ds = gdal.Open(file_in)
        img = self.read_stretch_min_max(ds,1)

        # Define motion model
        warp_mode = cv2.MOTION_TRANSLATION

        # Set the warp matrix to identity.
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Set the stopping criteria for the algorithm.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

        # Warp the blue and green channels to the red channel
        # (cc, warp_matrix) = cv2.findTransformECC(self.get_gradient(self.reference_band.astype(np.float32)),
        #                                          self.get_gradient(img.astype(np.float32)),
        #                                          warp_matrix, warp_mode, criteria)
        # Warp the blue and green channels to the red channel
        (cc, warp_matrix) = cv2.findTransformECC(self.reference_band_f, img.astype(np.float32),
                                                 warp_matrix, warp_mode, criteria)
        logging.info("CV2 {} translated by {}".format(file_in, warp_matrix))
        tvec = [-warp_matrix[1,2], -warp_matrix[0,2]]  # y,x
        self.trans_geotiff_tile(file_in, file_out, tvec)





