import glob
import os

import mercantile as mercantile
from osgeo import gdal

def output_res(pattern):
    files = sorted(glob.glob(pattern))
    for file in files:
        ds = gdal.Open(file)
        geo = ds.GetGeoTransform()
        print("{} {}".format(geo[1],geo[5]))
        lng, lat, zoom = geo[0], geo[3], 19-4
        t = mercantile.tile(lng, lat, zoom)
        print(t)
        bb = mercantile.bounds(t)
        print(bb)
        ulx = geo[0]
        uly = geo[3]
        lrx = ulx + ds.RasterXSize * geo[1]
        lry = uly + ds.RasterYSize * geo[5]
        print('{} {} {} {}'.format(ulx, uly, lrx, lry))



def perform_shift(file, dx, dy, ext):
    out_path = os.path.join(os.path.dirname(file), os.path.basename(file) + ext)

    ds = gdal.Open(file)
    geo = ds.GetGeoTransform()
    ulx = geo[0] + dx
    uly = geo[3] + dy
    lrx = ulx + ds.RasterXSize * geo[1] + dx
    lry = uly + ds.RasterYSize * geo[5] + dy

    cmd = 'gdal_translate -a_ullr {} {} {} {} {} {}'.format(ulx, uly, lrx, lry, file, out_path)
    print(cmd)
    os.system(cmd)
# 0.00007601, 0.00005469,


if __name__ == "__main__":
    wv3p = '/raid/data/wdixon/output/jacksonville/WV3/PAN'
    wv3m = '/raid/data/wdixon/output/jacksonville/WV3/MSI'
    wv3s = '/raid/data/wdixon/output/jacksonville/WV3/SWIR'

    output_res('/raid/data/wdixon/output/jacksonville/WV3/PAN/*.tif')
    # neg x moves img right, pos y moves img up
    # perform_shift(os.path.join(wv3p, '02MAY15WV031100015MAY02161943-P1BS-500648061030_01_P001_________AAE_0AAAAABPABR0.NTF_cal.tif'), 0.00004878, 0.00003953, '_s.tif')
    #perform_shift(os.path.join(wv3m, '01MAY15WV031200015MAY01160357-M1BS-500648062030_01_P001_________GA_E0AAAAAAKAAK0.NTF_cal.tif'), -0.00008991, 0.00001324, '_s.tif')

    #perform_shift(os.path.join(wv3s, '01NOV15WV031100015NOV01162032-A1BS-500900785020_01_P001_________SW_U0AAAAAACAAC0.NTF_cal.tif'), -0.00002953, 0.00005412, '_s.tif')


