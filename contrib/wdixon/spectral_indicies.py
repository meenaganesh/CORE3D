from cv2 import cv2

from osgeo import gdal

import numpy as np
from matplotlib import pyplot as plt
from osgeo.gdalconst import *

def test(self, file, outfile):
    NIR = 7 - 1
    NIR2 = 8 - 1
    RED = 5 - 1
    GREEN = 3 - 1
    BLUE = 2 - 1

    ds = gdal.Open(file, gdal.GA_ReadOnly)
    image = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    #                 dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))

    # Loop over all bands in dataset
    for b in range(ds.RasterCount):
        # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
        band = ds.GetRasterBand(b + 1)

        # Read in the band's data into the third dimension of our array
        image[:, :, b] = band.ReadAsArray()

    ndvi = (image[:, :, NIR] - image[:, :, RED]) / (image[:, :, NIR] + image[:, :, RED])

    driver = ds.GetDriver()
    out = driver.Create(outfile, ds.RasterYSize, ds.RasterXSize, 1, GDT_Float32)

    for b in range(0, 1):
        band = out.GetRasterBand(b + 1)
        band.WriteArray(ndvi, 0, 0)
        band.FlushCache()
        # band.SetNoDatavalue(0)

    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    del out
    del ds


# ds = gdal.Open('/raid/data/wdixon/output/jacksonville/WV3/MSI/01MAY15WV031200015MAY01160357-M1BS-500648062030_01_P001_________GA_E0AAAAAAKAAK0.NTF_cal.tif_s.tif', gdal.GA_ReadOnly)

ds = gdal.Open('/raid/data/wdixon/output3/35792_53940/30OCT14WV031100014OCT30155732-M1BS_reg_cut.tif', gdal.GA_ReadOnly)

# Allocate our array using the first band's datatype
image_datatype = ds.GetRasterBand(1).DataType

image = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount),dtype=np.float32)
#                 dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))

# Loop over all bands in dataset
for b in range(ds.RasterCount):
    # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
    band = ds.GetRasterBand(b + 1)

    # Read in the band's data into the third dimension of our array
    image[:, :, b] = band.ReadAsArray()

# WV3
# 8 Multispectral:
# Coastal: 400 - 450 nm
# Blue: 450 - 510 nm
# Green: 510 - 580 nm
# Yellow: 585 - 625 nm
# Red: 630 - 690 nm
# Red Edge: 705 - 745 nm
# Near-IR1: 770 - 895 nm
# Near-IR2: 860 - 1040 nm

NIR = 7-1
NIR2 = 8-1
RED = 5-1
GREEN = 3-1
YELLOW = 4-1
BLUE = 2-1

NDVI = (image[:, :, NIR] - image[:, :, RED]) / (image[:, :, NIR] + image[:, :, RED])

MNDWI = (image[:, :, GREEN] - image[:, :, NIR]) / (image[:, :, GREEN] + image[:, :, NIR])

REI = (image[:, :, NIR2] - image[:, :, BLUE]) / (image[:, :, NIR2] + image[:, :, BLUE]) # * image[:, :, NIR2]
BSI2 = (image[:, :, YELLOW] - 2*image[:, :, NIR]) / (image[:, :, YELLOW] + 2*image[:, :, NIR])


BAI = (image[:, :, BLUE] - image[:, :, NIR]) / (image[:, :, BLUE] + image[:, :, NIR]) # * image[:, :, NIR2]


x = np.sqrt(image[:, :, RED] * image[:, :, RED] + image[:, :, BLUE] * image[:, :, BLUE] + image[:, :, GREEN] * image[:, :, GREEN])
xx = (image[:, :, RED] - x) / (image[:, :, RED] + x)

#th2 = cv2.adaptiveThreshold((MNDWI*128+128).astype(np.uint8),128,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,4)
# ret,th2 = cv2.threshold((MNDWI*128+128).astype(np.uint8),140,255,cv2.THRESH_BINARY)
ret,th2 = cv2.threshold((REI*-128+128).astype(np.uint8),150,255,cv2.THRESH_BINARY)
ret,th2 = cv2.threshold((MNDWI*-128+128).astype(np.uint8),255,255,cv2.THRESH_OTSU)
ret,th2 = cv2.threshold((BAI*128+128).astype(np.uint8),255,255,cv2.THRESH_OTSU)  # http://iopscience.iop.org/article/10.1088/1755-1315/37/1/012006/pdf




#MNDWI = (Green - SWIR) / (Green + SWIR)

# plt.imshow(NDVI, cmap='gray_r')
#
# plt.colorbar()
# plt.show()
#
plt.imshow(REI, cmap='gray_r')
plt.colorbar()
plt.show()
#
# plt.imshow(MNDWI, cmap='gray_r')
# plt.colorbar()
# plt.show()

plt.imshow(th2, cmap='gray')
plt.colorbar()
plt.show()

print('hello world')