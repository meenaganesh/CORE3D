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



def perform_shift(file, out_path, dx, dy):
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
   # output_res('/raid/data/wdixon/output/jacksonville/WV3/PAN/*.tif')
    # neg x moves img right, pos y moves img up
    # perform_shift(os.path.join(wv3p, '02MAY15WV031100015MAY02161943-P1BS-500648061030_01_P001_________AAE_0AAAAABPABR0.NTF_cal.tif'), 0.00004878, 0.00003953, '_s.tif')
    #perform_shift(os.path.join(wv3m, '01MAY15WV031200015MAY01160357-M1BS-500648062030_01_P001_________GA_E0AAAAAAKAAK0.NTF_cal.tif'), -0.00008991, 0.00001324, '_s.tif')

    fin = '/raid/data/wdixon/output3/.cache/WV3/PAN/01MAY15WV031200015MAY01160357-P1BS-500648062030_01_P001_________AAE_0AAAAABPABQ0_cal.tif'
    fout = '/raid/data/wdixon/output3/.cache/WV3/PAN/01MAY15WV031200015MAY01160357-P1BS-500648062030_01_P001_________AAE_0AAAAABPABQ0_cal_s.tif'

    perform_shift(fin, fout, -0.0001191085*.8, 0.000007)


