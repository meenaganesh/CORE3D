import os
from cv2 import cv2

import numpy as np
from osgeo import gdal
import argparse

from osgeo.gdalconst import GDT_Byte


def read_stretch_min_max(ds, band, scale=255):
    arr = ds.GetRasterBand(band).ReadAsArray()
    mn = arr.min()
    mx = arr.max()
    z = (arr - mn) / (mx - mn) * scale
    return np.clip(z, 0, scale)


def read_as_rgb(ds):
    r = read_stretch_min_max(ds, 5)
    g = read_stretch_min_max(ds, 3)
    b = read_stretch_min_max(ds, 2)
    return [r, g, b]


def extant_file(x):
    """
    for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def create_output_image(ds, outfile, rgb):
    """
    Translates a geotiff tile by amount trans (in pixels)
    and saves to new geotiff
    """

    driver = ds.GetDriver()
    out = driver.Create(outfile, ds.RasterYSize, ds.RasterXSize, 3, GDT_Byte)

    for b in range(0, 3):
        band = out.GetRasterBand(b + 1)
        band.WriteArray(rgb[b])
        band.FlushCache()
        # band.SetNoDatavalue(0)

    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())

    # ensure changes are committed
    out.FlushCache()
    del out


def msi_to_rgb(in_file, out_file):
    ds = gdal.Open(in_file)
    rgb = read_as_rgb(ds)
    # create_output_image(ds, out_file, rgb)

    x = np.dstack(rgb).astype(np.uint8)
    H, S, V = cv2.split(cv2.cvtColor(x, cv2.COLOR_RGB2HSV))
    eq_V = cv2.equalizeHist(V)
    img_output = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)

    # x = np.dstack(rgb).astype(np.uint8)
    #
    # img_yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
    # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    #
    r = img_output[:,:,0]
    g = img_output[:,:,1]
    b = img_output[:,:,2]

    zzz = [r,g,b]


    create_output_image(ds, out_file, zzz)
    del ds


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Output pixels in reference coordinate system.')
    parser.add_argument("-i", dest="input", required=True, type=extant_file,
                        help="input image file", metavar="FILE")
    parser.add_argument("-o", dest="output", required=True,
                        help="output image file", metavar="FILE")

    args = parser.parse_args()
    msi_to_rgb(args.input, args.output)
    ds = gdal.Open(args.input)

