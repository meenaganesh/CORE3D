import os,sys,shutil
import glob
from osgeo import gdal,ogr

def get_res(vrt_file,xreslist,yreslist,ulxlist,ulylist,lrxlist,lrylist):
	src_ds=gdal.Open(vrt_file)
	ulx,xres,xskew,uly,yskew,yres=src_ds.GetGeoTransform()
	xreslist.append(xres)
	yreslist.append(yres)
	lrx=ulx+(src_ds.RasterXSize*xres)
	lry=uly+(src_ds.RasterYSize*yres)
        ulxlist.append(ulx)
	ulylist.append(uly)
	lrxlist.append(lrx)
	lrylist.append(lry)
	src_ds=None
	return xreslist,yreslist,ulxlist,ulylist,lrxlist,lrylist



	
#WV2MSIfolderPath=sys.argv[1]
#SHPfolderPath=sys.argv[6]
#WV2PANfolderPath=sys.argv[2]
#WV3MSIfolderPath=sys.argv[3]
#WV3PANfolderPath=sys.argv[4]
#WV3SWIRfolderPath=sys.argv[5]
OutputFolder=sys.argv[7]

RasterList=['WV2MSI','WV2PAN','WV3MSI','WV3PAN','WV3SWIR']
Raster_SHP_LAZ_FolderPath=[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]]#Add LAZ as sys.argv[7] and change OutputFilder to sys.argv[8]
OutputFolder=sys.argv[7]
if os.path.exists(OutputFolder):
	shutil.rmtree(OutputFolder)
	os.mkdir(OutputFolder)
	os.mkdir(OutputFolder+'Temp')

xreslist=[]
yreslist=[]
ulxlist=[]
ulylist=[]
lrxlist=[]
lrylist=[]

for i in range(len(RasterList)):
	print(RasterList[i])
	os.system('gdalbuildvrt'+' '+OutputFolder+'Temp/'+RasterList[i]+'.vrt'+' '+Raster_SHP_LAZ_FolderPath[i]+'*_Calibrated.tif')
        xreslist,yreslist,ulxlist,ulylist,lrxlist,lrylist=get_res(OutputFolder+'Temp/'+RasterList[i]+'.vrt',xreslist,yreslist,ulxlist,ulylist,lrxlist,lrylist)
	#os.system('gdalbuildvrt'+' '+OutputFolder+'WV2PAN.vrt'+' '+WV2PANfolderPath+'*_Calibrated.tif')
	#os.system('gdalbuildvrt'+' '+OutputFolder+'WV3MSI.vrt'+' '+WV3MSIfolderPath+'*_Calibrated.tif')
	#os.system('gdalbuildvrt'+' '+OutputFolder+'WV3PAN.vrt'+' '+WV3PANfolderPath+'*_Calibrated.tif')
	#os.system('gdalbuildvrt'+' '+OutputFolder+'WV3SWIR.vrt'+' '+WV3SWIRfolderPath+'*_Calibrated.tif')
#xreslist=[]
#yreslist=[]

xres_min_index=np.argmin(xreslist)
yres_min_index=np.argmin(yreslist)
if xres_min_index<=yres_min_index:
	ulx=ulxlist[xres_min_index]
	uly=ulylist[xres_min_index]
	lrx=lrxlist[xres_min_index]
	lry=lrylist[xres_min_index]
	xres=xreslist[xres_min_index]
	yres=yreslist[xres_min_index]
else:
	ulx=ulxlist[yres_min_index]
        uly=ulylist[yres_min_index]
        lrx=lrxlist[yres_min_index]
        lry=lrylist[xres_min_index]
        xres=xreslist[yres_min_index]
        yres=yreslist[yres_min_index]


for i in range(len(RasterList)):
	if 'PAN' in RasterList[i]:
		NO_DATA="0"
	else:
		NO_DATA="0 0 0 0 0 0 0 0"
	os.system('gdalbuildvrt'+' '+OutputFolder+'Temp/'+RasterList[i]+'.vrt'+' '+'-te'+' '+str(ulx)+' '+str(lry)+' '+str(lrx)+' '+str(uly)+' -srcnodata '+NO_DATA+' -vrtnodata '+NO_DATA+' '+Raster_SHP_LAZ_FolderPath+'*_Calibrated.tif')

	os.system('gdal_retile.py'+ ' -ps '+'4096'+' '+'4096'+' -co '+'"TILED=yes"'+' -co "INTERLEAVE=PIXEL" -ot UInt16 -targetDir'+' '+OutputFolder+' '+OutputFolder+'Temp/'+RasterList[i]+'.vrt')


#os.system('gdalbuildvrt'+' '+OutputFolder+'WV2MSI.vrt'+' '+'-te'+' '+str(ulx)+' '+str(lry)+' '+str(lrx)+' '+str(uly)+' -tr '+str(xres)+' '+str(abs(yres))+' -srcnodata "0 0 0 0 0 0 0 0" -vrtnodata "0 0 0 0 0 0 0 0" '+WV2MSIfolderPath+'*_Calibrated.tif')

#os.system('gdal_retile.py'+ ' -ps '+'4096'+' '+'4096'+' -co '+'"TILED=yes"'+' -co "INTERLEAVE=PIXEL" -ot UInt16 -targetDir'+' '+OutputFolder+' '+OutputFolder+'WV2MSI.vrt')

os.system('gdal_rasterize -burn 255 -ot Byte -tr '+str(xres)+' '+str(yres)+' -co "COMPRESS=JPEG" '+Raster_SHP_LAZ_FolderPath[5]+'*buildings.shp'+' '+OutputFolder+'Temp/'+'buildings.tif')

os.system('gdalbuildvrt '+OutputFolder+'Temp/'+'buildings.vrt'+' -te '+str(ulx)+' '+str(lry)+' '+str(lrx)+' '+str(uly)+' -tr '+str(xres)+' '+str(abs(yres))+' '+OutputFolder+'Temp/'+'buildings.tif')

os.system('gdal_retile.py'+' -ps '+'4096 4096 -co "TILED=yes" -co "COMPRESS=JPEG" -targetDir '+OutputFolder+' '+OutputFolder+'Temp/'+'buildings.vrt')

