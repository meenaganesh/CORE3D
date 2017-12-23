import os
import numpy as np
from osgeo import gdal
import argparse

from osgeo.gdalconst import GDT_Byte


def read_stretch_min_max(ds, band, scale=255):
    arr = ds.GetRasterBand(band).ReadAsArray()
    mip = arr.min()
    mxp = arr.max()
    z = (arr - mip) / (mxp - mip) * scale
    return np.clip(z, 0, scale)


def read_as_gray(ds):
    r = read_stretch_min_max(ds, 5)
    g = read_stretch_min_max(ds, 3)
    b = read_stretch_min_max(ds, 2)
    return r * .299 + g * .587 + b * .114


def extant_file(x):
    """
    for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def create_output_image(ds, outfile, img):
    """
    Translates a geotiff tile by amount trans (in pixels)
    and saves to new geotiff
    """

    driver = ds.GetDriver()
    out = driver.Create(outfile, ds.RasterYSize, ds.RasterXSize, 1, GDT_Byte)

    for b in range(0, 1):
        band = out.GetRasterBand(b + 1)
        band.WriteArray(img, 0, 0)
        band.FlushCache()
        # band.SetNoDatavalue(0)

    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())

    # ensure changes are committed
    out.FlushCache()
    del out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Output pixels in reference coordinate system.')
    parser.add_argument("-i", dest="input", required=True, type=extant_file,
                        help="input image file", metavar="FILE")
    parser.add_argument("-o", dest="output", required=True,
                        help="output image file", metavar="FILE")

    args = parser.parse_args()

    ds = gdal.Open(args.input)
    o = read_as_gray(ds)
    create_output_image(ds, args.output, o)

