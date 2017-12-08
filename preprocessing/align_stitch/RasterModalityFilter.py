import os
import re

import numpy as np
import rasterio
import shapely as shapely
from rasterio.mask import mask
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union


class RasterModalityFilter:
    def __init__(self):
        self.tiles_bb = {}  # key = row_col as string, value = bounding box
        self.tiles_files = {}  # key = row_col as string, value = list of tiles at row_col
        self.mod_union = {}  # key = modality name, value is the union of all shapes

    def load_extents(self, filename):
        with open(filename) as f:
            for line in f:
                data = line.split(' ')
                bb = shapely.geometry.geo.box(float(data[5]), float(data[6]), float(data[7]), float(data[8]))
                if data[0] not in self.mod_union:
                    self.mod_union[data[0]] = Polygon(bb)
                else:
                    self.mod_union[data[0]] = cascaded_union([bb, self.mod_union[data[0]]])

                if data[0] == 'tile':
                    m = re.search('([0-9]+_[0-9]+).tif', data[1])
                    row_col = m.group(1)
                    if row_col not in self.tiles_bb:
                        self.tiles_bb[row_col] = bb
                        self.tiles_files[row_col] = []
                    self.tiles_files[row_col].append(data[1])

    def generate_raster_cuts(self, cut_name, required_modalities):
        for row_col, bb in self.tiles_bb.items():
            region = Polygon(bb)
            for r in required_modalities:
                if r in self.mod_union:
                    poly = self.mod_union[r]
                    region = region.intersection(poly)
                else:
                    region = None  # modality did not exist

            if region is not None and not region.is_empty:
                print('intersection {} {}'.format(region.area, region))
                for tf in self.tiles_files[row_col]:
                    self.cut_raster(tf, os.path.splitext(tf)[0] + '_' + cut_name + '.tif', region)
            else:
                print('empty')

    def cut_raster(self, in_file, out_file, intersection):
        region = np.vstack(intersection.boundary.coords.xy)
        polynomial = list(zip(region[0], region[1]))
        # Setup the polynomial cut:
        geoms = [{'type': 'Polygon', 'coordinates': [polynomial]}]
        # Perform cut
        with rasterio.open(in_file) as f:
            output_image, output_transform = mask(f, geoms, crop=True, invert=False)
        out_meta = f.meta.copy()

        # Save Output
        out_meta.update({'driver': 'GTiff', 'height': output_image.shape[1],
                         'width': output_image.shape[2],
                         'transform': output_transform})
        with rasterio.open(out_file, 'w', **out_meta) as f:
            f.write(output_image)


if __name__ == '__main__':
    ii = RasterModalityFilter()
    ii.load_extents('/home/sernamlim/data/CORE3D/jacksonville/output/ext.txt')
    ii.generate_raster_cuts('mycut', ['wv3_msi', 'laz'])
