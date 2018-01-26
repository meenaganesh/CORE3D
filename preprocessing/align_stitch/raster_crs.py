from osgeo import gdal
from osgeo.osr import SpatialReference, CoordinateTransformation
from preprocessing.align_stitch.image_util import read_raster

"""
rs = RasterCrs("in.tif", alt_crs=3857) 
v = rs.get_value(-9095700.917, 3546372.663)
print(v)
"""


class RasterCrs:

    def __init__(self, file, alt_crs=None, bands=None):
        """
        Create a RasterCrs object that allows retrieval of data from the raster using an alternative CRS

        :param file: raster to load
        :param alt_crs: alternative CRS as a number
        :param bands: bands to load. This will be the all the values returned for the given index.
        """
        self.image, self.geo, self.proj, self.meta = read_raster(file, bands=bands)
        self.r2p = RasterToPixel(geo=self.geo)

        self.idx_crs = SpatialReference()
        self.idx_crs.ImportFromEPSG(alt_crs)

        self.dst_crs = SpatialReference(self.proj)
        if alt_crs == None:
            self.idx_crs = self.dst_crs
        else:
            self.idx_crs = SpatialReference()
            self.idx_crs.ImportFromEPSG(alt_crs)

        self.alt_to_dst = CoordinateTransformation(self.idx_crs, self.dst_crs)
        self.dst_to_alt = CoordinateTransformation(self.dst_crs, self.idx_crs )

    def get_value_alt(self, alt_x, alt_y):
        """
        get values for bands at specified coordinates using the alternate crs specified.
        :param alt_x:
        :param alt_y:
        :except IndexError when specifying bounds outside the image
        :return: values from all specified bands as list
        """
        px, py = self.get_pixel_alt_crs(alt_x, alt_y)
        return self.image[px, py, :]

    def get_value(self, x, y):
        """
        get values for bands at specified coordinates using the source crs.
        :param x:
        :param y:
        :except IndexError when specifying bounds outside the image
        :return: values from all specified bands as list
        """
        px, py = self.get_pixel_crs(x,y)
        return self.image[px, py, :]

    def get_pixel_crs(self, x, y):
        """
        get pixel coordinates for source CRS
        :param x: source x coordinate
        :param y: source y coordinate
        :except IndexError when specifying bounds outside the image
        :return: pixel coordinate of source image
        """
        return self.r2p.to_pixel(x, y)

    def get_pixel_alt_crs(self, alt_x, alt_y):
        """
        get pixel coordinates for source CRS
        :param alt_x: alt csr x coordinate
        :param alt_y: alt crs y coordinate
        :except IndexError when specifying bounds outside the image
        :return: pixel coordinate of source image
        """
        dest_x, dest_y, dest_z = self.alt_to_dst.TransformPoint(alt_x, alt_y)
        return self.r2p.to_pixel(dest_x, dest_y)


class RasterToPixel:
    def __init__(self, **kwargs):
        if 'file' in kwargs:
            ds = gdal.Open(kwargs['file'])
            geo = ds.GetGeoTransform()
        elif 'geo' in kwargs:
            geo = kwargs['geo']
        self.x0 = geo[0]
        self.y0 = geo[3]
        self.inv_pixel_width = 1.0 / geo[1]
        self.inv_pixel_height = 1.0 / -geo[5]

    def to_pixel(self, x, y):
        px = round((x - self.x0) * self.inv_pixel_width)
        py = round((self.y0 - y) * self.inv_pixel_height)
        return px, py