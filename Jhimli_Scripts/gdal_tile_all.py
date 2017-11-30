import os,sys
import glob
from osgeo import gdal,ogr

WV2MSIfolderPath=sys.argv[1]
SHPfolderPath=sys.argv[3]
WV2PANfolderPath=sys.argv[2]
#WV3MSIfolderPath=sys.argv[3]
#WV3PANfolderPath=sys.argv[4]
#WV3SWIRfolderPath=sys.argv[5]
OutputFolder=sys.argv[4]
os.system('gdalbuildvrt'+' '+OutputFolder+'WV2MSI.vrt'+' '+WV2MSIfolderPath+'*_Calibrated.tif')
os.system('gdalbuildvrt'+' '+OutputFolder+'WV2PAN.vrt'+' '+WV2PANfolderPath+'*_Calibrated.tif')

src_ds=gdal.Open(OutputFolder+'WV2PAN.vrt')
ulx,xres,xskew,uly,yskew,yres=src_ds.GetGeoTransform()
print(xres,yres)
lrx=ulx+(src_ds.RasterXSize*xres)
lry=uly+(src_ds.RasterYSize*yres)

os.system('gdalbuildvrt'+' '+OutputFolder+'WV2PAN.vrt'+' '+'-te'+' '+str(ulx)+' '+str(lry)+' '+str(lrx)+' '+str(uly)+' -srcnodata "0" -vrtnodata "0" '+WV2PANfolderPath+'*_Calibrated.tif')

os.system('gdal_retile.py'+ ' -ps '+'4096'+' '+'4096'+' -co '+'"TILED=yes"'+' -co "INTERLEAVE=PIXEL" -targetDir'+' '+OutputFolder+' '+OutputFolder+'WV2PAN.vrt')


os.system('gdalbuildvrt'+' '+OutputFolder+'WV2MSI.vrt'+' '+'-te'+' '+str(ulx)+' '+str(lry)+' '+str(lrx)+' '+str(uly)+' -tr '+str(xres)+' '+str(abs(yres))+' -srcnodata "0 0 0 0 0 0 0 0" -vrtnodata "0 0 0 0 0 0 0 0" '+WV2MSIfolderPath+'*_Calibrated.tif')

os.system('gdal_retile.py'+ ' -ps '+'4096'+' '+'4096'+' -co '+'"TILED=yes"'+' -co "INTERLEAVE=PIXEL" -targetDir'+' '+OutputFolder+' '+OutputFolder+'WV2MSI.vrt')

os.system('gdal_rasterize -burn 255 -ot Byte -tr '+str(xres)+' '+str(yres)+' -co "COMPRESS=JPEG" '+SHPfolderPath+'*buildings.shp'+' '+OutputFolder+'buildings.tif')

os.system('gdalbuildvrt '+OutputFolder+'buildings.vrt'+' -te '+str(ulx)+' '+str(lry)+' '+str(lrx)+' '+str(uly)+' -tr '+str(xres)+' '+str(abs(yres))+' '+OutputFolder+'buildings.tif')

os.system('gdal_retile.py'+' -ps '+'4096 4096 -co "TILED=yes" -co "COMPRESS=JPEG" -targetDir '+OutputFolder+' '+OutputFolder+'buildings.vrt')

