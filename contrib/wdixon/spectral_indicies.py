from osgeo import gdal
from osgeo import gdal_array
import numpy as np
import matplotlib.pyplot as plt


ds = gdal.Open('/raid/data/wdixon/output/jacksonville/WV3/MSI/01MAY15WV031200015MAY01160357-M1BS-500648062030_01_P001_________GA_E0AAAAAAKAAK0.NTF_cal.tif_s.tif', gdal.GA_ReadOnly)

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

NIR = 7-1
NIR2 = 8-1
RED = 5-1
GREEN = 3-1
BLUE = 2-1

ndvi = (image[:, :, NIR] - image[:, :, RED]) / (image[:, :, NIR] + image[:, :, RED])

MNDWI = (image[:, :, GREEN] - image[:, :, NIR]) / (image[:, :, GREEN] + image[:, :, NIR])

REI = (image[:, :, NIR2] - image[:, :, BLUE]) / (image[:, :, NIR2] + image[:, :, BLUE]) # * image[:, :, NIR2]

#MNDWI = (Green - SWIR) / (Green + SWIR)


plt.imshow(REI, cmap='gray_r')

plt.colorbar()
plt.show()

plt.imshow(MNDWI, cmap='gray')
plt.colorbar()
plt.show()

print('hello world')