from tkinter import Image

from osgeo import gdal
from osgeo import gdal_array
import numpy as np
import matplotlib.pyplot as plt

dir = '/raid/data/wdixon/output2/4476_6743'


def norm(a,mip,mxp,n):
    #mip = a.min()
    #mxp = a.max()
    z = (a - mip) / (mxp - mip) * n
    return np.clip(z,0,255)


ds = gdal.Open(dir +'/'+'4476_6743_01MAY15WV031200015MAY01160357-P1BS.tif')
p = np.zeros((ds.RasterYSize, ds.RasterXSize, 3),dtype=np.float32)
pan = norm(ds.GetRasterBand(1).ReadAsArray(), 140,795,1)
# mip = pan.min()
# mxp = pan.max()
# pan = (pan-mip)/(mxp-mip)
# p[:,:,0] = pan
# p[:,:,1] = pan
# p[:,:,2] = pan


# mip = min(pan.all())
# mxp = min(pan.all())



ds = gdal.Open(dir +'/'+'4476_6743_01MAY15WV031200015MAY01160357-M1BS.tif')
msi = np.zeros((ds.RasterYSize, ds.RasterXSize, 3),dtype=np.float32)
r = norm(ds.GetRasterBand(1).ReadAsArray(),252,459,255) * pan # red
g = norm(ds.GetRasterBand(2).ReadAsArray(),195,513,255) * pan # green
b = norm(ds.GetRasterBand(3).ReadAsArray(),176,720,255) * pan # blue

rgb = np.zeros((ds.RasterYSize, ds.RasterXSize, 3),dtype=np.float32)
# rgb[:, :, 0] = pan * msi[:, :, 0]
# rgb[:, :, 1] = pan * msi[:, :, 1]
# rgb[:, :, 2] = pan * msi[:, :, 2]
# rgb[:, :, 0] = msi[:, :, 0]
# rgb[:, :, 1] = msi[:, :, 1]
# rgb[:, :, 2] = msi[:, :, 2]

# plt.imshow(r, interpolation=None)
# plt.show()
#
# plt.imshow(g, interpolation=None)
# plt.show()
#
# plt.imshow(b, interpolation=None)
# plt.show()

rgb[..., 0] = r
rgb[..., :, 1] = g
rgb[..., 2] = b

img = np.dstack((r.astype(np.uint8),g.astype(np.uint8),b.astype(np.uint8)))

plt.imshow(img, interpolation=None)
plt.show()

print('hello world')