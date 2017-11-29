import os, sys,glob
from PIL import Image, ImageDraw
import numpy as np
from osgeo import gdal,gdalnumeric
from osgeo import ogr
import cv2
import WVCalKernel
from WVCalKernel import *

import pdb



def create_polygon(coords):
    ring=ogr.Geometry(ogr.wkbLinearRing)
    for coord in range(shape.coords[0]):
        ring.AddPoint(float(coords[coord][0]), float(coords[coord][1]))
    newpoly=ogr.Geometry(ogr.wkbPolygon)
    newpoly.AddGeometry(ring)
    return newpoly.ExportToWkt()

def imageToArray(i):
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a=gdalnumeric.fromstring(i.tobytes(),'b')
    a.shape=i.im.size[1], i.im.size[0]
    return a

def arrayToImage(a):
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i=Image.frombytes('L',(a.shape[1],a.shape[0]),
            (a.astype('b')).tostring())
    return i

def world2pixel(geoMatrix, x, y):
  """
  Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
  the pixel location of a geospatial coordinate
  """
  ulX = geoMatrix[0] #These are hard coded as GeoTransform values seem to be incorrect, 3 following lines.
  ulY = geoMatrix[3]
  xDist =geoMatrix[1]
  yDist = geoMatrix[5]
  rtnX = geoMatrix[2]
  rtnY = geoMatrix[4]
  pixel = abs(int((x - ulX) / xDist))
  line = abs(int((ulY - y) / yDist))
  return (pixel, line)

def world2pixelshape(geoMatrix, x, y):
  """
  Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
  the pixel location of a geospatial coordinate
  """
  ulX = geoMatrix[0]
  ulY = geoMatrix[3]
  xDist = 0.000025379#geoMatrix[1] # 2 lines hardcoded values
  yDist = 0.000014183#geoMatrix[5] 
  rtnX = geoMatrix[2]
  rtnY = geoMatrix[4]
  pixel = abs(int((x - ulX) / xDist))
  line = abs(int((ulY - y) / yDist))
  return (pixel, line)


def stretch(a):
  """
  Performs a histogram stretch on a gdalnumeric array image.
  """
  hist = histogram(a)
  im = arrayToImage(a)
  lut = []
  for b in range(0, len(hist), 256):
    # step size
    step = reduce(operator.add, hist[b:b+256]) / 255
    # create equalization lookup table
    n = 0
    for i in range(256):
      lut.append(n / step)
      n = n + hist[i+b]
  im = im.point(lut)
  return imageToArray(im)

def histogram(a, bins=range(0,256)):
  """
  Histogram function for multi-dimensional array.
  a = array
  bins = range of numbers to match
  """
  fa = a.flat
  n = gdalnumeric.searchsorted(gdalnumeric.sort(fa), bins)
  n = gdalnumeric.concatenate([n, [len(fa)]])
  hist = n[1:]-n[:-1]
  return hist

#Main arguments argv[1]: Path to raster file, argv[2]: Path to shape file, argv[3]: Output folder path
def main():
        InputFolderPath=sys.argv[1]

        WVType=sys.argv[2]
        OutputFolder=sys.argv[3]
        NTF_list=sorted(glob.glob(os.path.join(InputFolderPath+'*.NTF')))
        print NTF_list
        for i in range(len(NTF_list)):
        	RasterFilePath=NTF_list[i]
		RasterfileBase=os.path.basename(RasterFilePath)
        	RasterFile=os.path.splitext(RasterfileBase)[0]

                if WVType=='WV2':
			
        	#Raster input as MSI
        		sat_params=WV2params()
               	else:
			sat_params=WV3params()
                print sat_params
        	raster=RadiometricCalibrator(RasterFilePath,sat_params)
        	raster.calibrate()
        	raster_array,src_ds=raster.get_calibrated_data()
        	#driver=gdal.GetDriverByName('GTiff')
        	print(raster_array.shape)
       
        	for i in range(raster_array.shape[2]):
			band=src_ds.GetRasterBand(i+1)
                	ar=np.zeros((raster_array.shape[0],raster_array.shape[1]))
			ar=raster_array[:,:,i]
			band.WriteArray(ar)
        
                outputFileName=OutputFolder+RasterFile+'_Calibrated.tif'
        	gdal.Warp(outputFileName,src_ds,format='GTiff',rpc=True,multithread=True,resampleAlg='cubic',outputType=gdal.GDT_UInt16)
		#ds=gdal.Open(OutputFolder+RasterFile+'_Calibrated.tif')
		#data = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount))
		#for rasterCounter in range(1,ds.RasterCount+1):
		#	data[:,:,rasterCounter-1]=ds.GetRasterBand(rasterCounter).ReadAsArray()
		#	print(np.amin(data),np.amax(data))


def rasterize_shape(ShapeFilePath,baseRaster_XSize,baseRaster_YSize,baseRaster_pixelwidth,baseRaster_pixelheight,baseRaster_xmin,baseRaster_ymax):	
	ShapefileBase=os.path.basename(ShapeFilePath)
        ShapeFile=os.path.splitext(ShapefileBase)[0]
        output = os.path.join(sys.argv[3],'overlay.tif')
        #gdal_raster_cmd='/projects/bialvm/Jhimli/anaconda2/envs/py27_mpl_gdal/bin/gdal_rasterize'+' -a'+' value'+' -ot'+' Int16'+' -at'+' -ts '+str(XSize)+' '+str(YSize)+' -l'+' '+ShapeFile+' '+sys.argv[2]+' '+output
        #print gdal_raster_cmd
        #os.system(gdal_raster_cmd)
        
	SHPFILE_LandUsage=ogr.Open(ShapeFilePath)
	layer = SHPFILE_LandUsage.GetLayer(0)
            
	output = os.path.join(sys.argv[3],'overlay.tif')
	target_ds = gdal.GetDriverByName('GTiff').Create(output, baseRaster_XSize, baseRaster_YSize, 1, gdal.GDT_Int16)
	target_ds.SetGeoTransform((baseRaster_xmin, baseRaster_pixelwidth, 0, baseRaster_ymax, 0, baseRaster_pixelheight))
        gdal.RasterizeLayer(target_ds,[1],layer,options=["ALL_TOUCHED=TRUE","ATTRIBUTE=value"])
        return target_ds

if __name__ == "__main__":
   	main()
