import glob
import logging
import re

import mercantile

from gdal_retile import *

logger = logging.getLogger(__name__)


class TileMaker:
    def __init__(self, out_folder):
        self.zoom = 14  # 15 gives us approx 2.682209014892578125e-6 deg/pixel using 4096x4096
        self.no_data = 0
        self.tile_x = 8192
        self.tile_y = 8192
        self.resample = 'lanczos'  # nearest (default),bilinear,cubic,cubicspline,lanczos,average,mode
        self.out_folder = out_folder
        self.rasters = []

    def add_rasters(self, pattern):
        files = sorted(glob.glob(pattern))
        self.rasters.extend(files)

    def add_raster(self, raster):
        self.rasters.append(raster)

    def create_all_tiles(self):
        for file in self.rasters:
            self._create_tiles_file_extent(file, postfix=self._postfix_file(file))

    def create_tiles_xy(self, x, y):
        result = []
        tile = mercantile.Tile(x, y, self.zoom)
        for file in self.rasters:
            result.append(self._create_single_tile(tile, file, postfix=self._postfix_file(file)))
        return result

    def create_tiles_deg(self, lng, lat):
        result = []
        tile = mercantile.tile(lng=lng, lat=lat, zoom=self.zoom)
        for file in self.rasters:
            result.append(self._create_single_tile(tile, file, postfix=self._postfix_file(file)))
        return result

    def _create_single_tile(self, tile, in_file, prefix='', postfix=''):
        name = "{}_{}".format(tile.x, tile.y)

        tile_folder = os.path.join(self.out_folder, name)
        os.makedirs(tile_folder, exist_ok=True)

        out_base_name = prefix + name + postfix + '.tif'
        bb = mercantile.bounds(tile)
        out_file = os.path.join(tile_folder, out_base_name)

        if os.path.exists(out_file):
            logger.info('Skipping tile for {}, file exists.'.format(out_file))
        else:
            x_res = abs(bb.west - bb.east) / self.tile_x    # should be fixed
            y_res = abs(bb.north - bb.south) / self.tile_y  # changes with lat

            # When we select the projection - we will grow by a pixel in all directions to handle partial pixel
            cmd = 'gdal_translate -r {} -tr {} {} -projwin {} {} {} {} -a_nodata {} -co COMPRESS=LZW {} {}'.format(
                self.resample, x_res, y_res, bb.west - x_res, bb.north + y_res, bb.east + x_res,
                bb.south - y_res, self.no_data, in_file, out_file)
            logging.info(cmd)
            os.system(cmd)
        return out_file

    def _create_tiles_file_extent(self, in_file, prefix='', postfix=''):
        ds = gdal.Open(in_file)
        geo = ds.GetGeoTransform()
        ds.close()

        ul_lon = geo[0]
        ul_lat = geo[3]
        lr_lon = ul_lon + ds.RasterXSize * geo[1]
        lr_lat = ul_lat + ds.RasterYSize * geo[5]

        bb = mercantile.LngLatBbox(ul_lon, lr_lat, lr_lon, ul_lat)
        tiles = mercantile.tiles(bb.west, bb.south, bb.east, bb.north, [self.zoom])

        result = []
        for t in tiles:
            result.append(self._create_single_tile(t, in_file, prefix, postfix))
        return result

    @staticmethod
    def _postfix_file(file):
        name = os.path.basename(file)
        postfix = ''
        m = re.search('([^-]+-[^-]+)', name)
        if m is not None:
            postfix = '_' + m.group(1)
        return postfix


if __name__ == "__main__":
    wv3p = '/raid/data/wdixon/output/jacksonville/WV3/PAN'
    wv3m = '/raid/data/wdixon/output/jacksonville/WV3/MSI'
    wv3s = '/raid/data/wdixon/output/jacksonville/WV3/SWIR'

    rt = TileMaker('/raid/data/wdixon/output2')
    # rt.add_rasters('/raid/data/wdixon/output/jacksonville/buildings.tif')
    # rt.add_rasters('/raid/data/wdixon/output/jacksonville/WV3/PAN/*.tif')
    # rt.add_rasters('/raid/data/wdixon/output/jacksonville/WV3/MSI/*.tif')
    rt.create_tiles_xy(4476, 6743)