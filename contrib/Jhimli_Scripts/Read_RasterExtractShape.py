import os, sys
from PIL import Image, ImageDraw
import numpy as np
from osgeo import gdal,gdalnumeric
from osgeo import ogr
import cv2

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

def world2pixelraster(geoMatrix, x, y):
  """
  Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
  the pixel location of a geospatial coordinate
  """
  ulX = -81.690277778# geoMatrix[0] #These are hard coded as GeoTransform values seem to be incorrect, 3 following lines.
  ulY = 30.400555556#geoMatrix[3]
  xDist = 0.000025379#geoMatrix[1]
  yDist = 0.000014183#geoMatrix[5]
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
        RasterFilePath=sys.argv[1]
        #Raster input as MSI
	MSIImage  = gdal.Open(RasterFilePath)
        
	infotext=os.system('gdalinfo'+' '+ RasterFilePath)
        print infotext
	XSize=MSIImage.RasterXSize
	YSize=MSIImage.RasterYSize
        MSIGeoTrans=MSIImage.GetGeoTransform()
        print MSIGeoTrans
        pixelSizeX=MSIGeoTrans[1]
        pixelSizeY=MSIGeoTrans[5]
        RasterBand_Count=MSIImage.RasterCount
	print XSize,YSize,RasterBand_Count,pixelSizeX,pixelSizeY
	for i in range(RasterBand_Count):
		bandarray=np.array(MSIImage.GetRasterBand(i+1).ReadAsArray())
		print bandarray.shape
		#print bandarray
        #Shapefile for landusage
        ShapeFilePath=sys.argv[2]
	SHPFILE_LandUsage=ogr.Open(ShapeFilePath)
	layer = SHPFILE_LandUsage.GetLayer(0)
        minX,maxX,minY,maxY=layer.GetExtent()
        raster_minX=-81.690277778 #Hardcoded with following 3 lines
        raster_minY=30.298888889 
        raster_maxX=-81.456388889
        raster_maxY=30.400555556
        if (minX<raster_minX):
		X_translate=abs(minX-raster_minX)
	if (minY<raster_minY):
		Y_translate=abs(minY-raster_minY)
        xDist = 0.000025379#geoMatrix[1] #Hardcoded 2 lines
        yDist = 0.000014183#geoMatrix[5]
        X_translate_pixels=int(X_translate/xDist)
        Y_translate_pixels=int(Y_translate/yDist)
        print minX,raster_minX,minY,raster_minY,maxX,raster_maxX,maxY,raster_maxY
        ulX,ulY=world2pixelraster(MSIGeoTrans,raster_minX,raster_maxY)
        lrX,lrY=world2pixelraster(MSIGeoTrans,raster_maxX,raster_minY)
        s_ulX,s_ulY=world2pixelraster(MSIGeoTrans,minX,maxY)
        s_lrX,s_lrY=world2pixelraster(MSIGeoTrans,maxX,minY)
 
	#ulX=ulX+X_translate_pixels
        #ulY=ulY+Y_translate_pixels
        #lrX=lrX+X_translate_pixels
        #lrY=lrY+Y_translate_pixels

        print('Pixel Extent',ulX,lrX,ulY,lrY)
        pxWidth=int(lrX-ulX)
       
        pxHeight=int(lrY-ulY)
        srcArray=np.zeros((8,YSize,XSize),dtype=float)
        #srcArray=gdalnumeric.LoadFile('/projects/bialvm/Jhimli/MSI_JacksonVille_WV2/05SEP16WV021200016SEP05162552-M1BS-500881026010_01_P006_________GA_E0AAAAAAIAAG0.NTF')#(np.zeros((8,YSize,XSize),dtype=float)
        for i in range(RasterBand_Count):
                bandarray=np.array(MSIImage.GetRasterBand(i+1).ReadAsArray().astype(np.float32))
                srcArray[i,:,:]=bandarray
        cliparray=srcArray[:,ulY:lrY,ulX:lrX]
        newclip=np.zeros((pxHeight,pxWidth,8),dtype=float)
        for i in range(0,8):
            newclip[:,:,i]=cliparray[i,:,:]
        OutputFolderPath=sys.argv[3]
	cv2.imwrite(os.path.join(OutputFolderPath+'clipraster.png'),newclip[:,:,1])
        #pdb.set_trace()
	xoffset=s_ulX
        yoffset=s_ulY
        print(lrY,s_lrY)
        scale_Y=float(lrY)/float(s_lrY)
        scale_X=float(ulX)/float(s_ulX)
        print ('YScale:',scale_Y)
        print('XScale:',scale_X)
        print('Xoffset',xoffset,'Yoffset',yoffset)
        geoTrans = list(MSIGeoTrans)
    	geoTrans[0] = minX
    	geoTrans[3] = maxY

        print geoTrans[0],geoTrans[3]    
        #ds=gdal.GetDriverByName('gtiff').Create('/projects/bialvm/Jhimli/wood_raster.tif',XSize,YSize,1)
        #ds.SetGeoTransform([ulx,1,0,uly,0,-1])
        #print layer.GetSpatialRef().ExportToWkt()
        #ds.SetProjection(layer.GetSpatialRef().ExportToWkt())

        #Extracting geometry by material type from shape file landuasages
	multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
	layer.SetAttributeFilter("type='stadium'")
        #print layer
        #new_band=ds.GetRasterBand(1)
        #new_band.SetNoDataValue(0)
        #gdal.RasterizeLayer(ds,[1],layer,None, None, [1], ['ALL_TOUCHED=TRUE'])
        allpoly=[]
	for feature in layer:
    		type_name = feature.GetField("type")
    		geometry = feature.GetGeometryRef()
    		rcnt=0
    		for ring in geometry:
    			rcnt+=1
    			points = ring.GetPointCount()
        		pcnt=0
                        poly_coords=[]
                        poly_pixels=[]
    			for p in xrange(points):
      				pcnt+=1
				lon, lat, z = ring.GetPoint(p)
                		#print lon,lat,z
				poly_coords.append([lon,lat])
                        for p in poly_coords:
				px_X,px_Y=world2pixelshape(geoTrans,p[0],p[1])
				poly_pixels.append((px_X-xoffset,int(px_Y*scale_Y)))
		
		allpoly.append(poly_pixels)
        print allpoly
	rasterPoly=Image.new("L",(pxWidth,pxHeight),1)
        rasterize=ImageDraw.Draw(rasterPoly)
        for i in range(len(allpoly)):
                print allpoly[i]
	       	rasterize.polygon(allpoly[i],0)
        print(pxWidth,pxHeight)
	mask=imageToArray(rasterPoly)
        
        mask=mask.astype(gdalnumeric.uint8)
        maskFileName=os.path.join(OutputFolderPath+'mask.jpg')
        gdalnumeric.SaveArray(mask*255, maskFileName, format="JPEG")

        for i in range(8):
		cliparray[i,:,:]=cliparray[i,:,:]
        # Save as an 8-bit jpeg for an easy, quick preview
        clipImage=np.zeros((pxHeight,pxWidth,3),dtype=np.uint8)
        for kk in range(0,3):
		clipImage[:,:,kk]=cliparray[kk,:,:].astype(gdalnumeric.uint8)
        #mask=mask.astype(gdalnumeric.uint8)
        new_mask=np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        new_mask[:,:,0]=255*mask
        new_mask[:,:,1]=mask
        new_mask[:,:,2]=mask
       
	#pdb.set_trace()              
        result_img=cv2.addWeighted(new_mask,0.5,clipImage,0.5,0)
        ShapeMaskOnRasterFile=os.path.join(OutputFolderPath+'OUTPUT.png')
        cv2.imwrite(ShapeMaskOnRasterFile,result_img)

        #gdalnumeric.SaveArray(clipImage, "/projects/bialvm/Jhimli/OUTPUT.jpg", format="JPEG")
        '''
        gtiffDriver = gdal.GetDriverByName( 'GTiff' )
    	if gtiffDriver is None:
        	raise ValueError("Can't find GeoTiff Driver")
    	gtiffDriver.CreateCopy( "/projects/bialvm/Jhimli/OUTPUT.tif",OpenArray(clipImage, prototype_ds=raster_path, xoff=xoffset, yoff=yoffset )
    )
        '''


		#new_layer_poly=create_polygon(poly_coords)
			#print new_layer_poly
    		#print rcnt,pcnt,geometry.GetGeometryName()
        
        #array=new_band.ReadAsArray()
        #print np.sum(array)
        #fig=plt.figure()
        #im1=plt.imshow(array)
        #plt.show()

if __name__ == "__main__":
    main()

#print band
#try:        
#   band=MSIImage.GetRasterd_num)
   
#except RuntimeError, e:
#   print "Band (%i) not found",band_num
#   print e
#   sys.exit(1)
