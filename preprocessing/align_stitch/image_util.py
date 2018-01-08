import os
import numpy as np
from osgeo import gdal, gdal_array


def read_raster(file, np_type=None, bands=None):
    ds = gdal.Open(file)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    meta = ds.GetMetadata()

    if np_type is None:
        # load type based on raster
        np_type = gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)

    # Allocate our numpy array
    image = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np_type)

    if bands is None:
        bands = range(ds.RasterCount)

    # Loop over desired bands (zero based)
    for b in bands:
        band = ds.GetRasterBand(b + 1)  # bands are indexed from 1
        image[:, :, b] = band.ReadAsArray()

    del ds
    return image, geo, proj, meta


def get_driver(file):
    f_ext = os.path.splitext(file)[1]
    for i in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(i)
        if driver.GetMetadataItem(gdal.DCAP_RASTER):
            d_ext_str = driver.GetMetadataItem(gdal.DMD_EXTENSIONS)
            if d_ext_str is not None:
                for d_ext in d_ext_str.split(' '):
                    if f_ext == '.'+d_ext:
                        return driver
    return None


def write_raster(image, out_file, geo=None, proj=None, meta=None, np_type=None, bands=None):
    driver = get_driver(out_file)
    if np_type is None:
        np_type = np.float32

    if bands is None:
        bands = range(image.shape[2])

    out = driver.Create(out_file, image.shape[1], image.shape[0], len(bands),
                        gdal_array.NumericTypeCodeToGDALTypeCode(np_type))

    for b in bands:
        band = out.GetRasterBand(b + 1)
        band.WriteArray(image[...,b].astype(np_type), 0, 0)
        band.FlushCache()

    if geo is not None:
        out.SetGeoTransform(geo)

    if proj is not None:
        out.SetProjection(proj)

    if meta is not None:
        out.SetMetadata(meta)

    del out

if __name__ == "__main__":

    img, geo, proj, meta = read_raster('/raid/data/wdixon/output3/35792_53940/21JAN15WV031100015JAN21161243-M1BS_reg_cut.tif')
    write_raster(img, '/tmp/tmp.tif', geo, proj, meta, np_type=np.uint8)
