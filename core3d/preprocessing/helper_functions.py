# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:30:35 2017

@author: 200021424
"""

import numpy as np
from shapely.geometry.polygon import Polygon
import rasterio
from rasterio.tools.mask import mask



def completeTiff(filename, outputFilename, boundingBoxCollection):

    # Find the intersection of polygons
    polygon = Polygon(boundingBoxCollection[0])
    for boundsIndex in range(1, boundingBoxCollection.shape[0]):
        polygon = polygon.intersection(Polygon(boundingBoxCollection[boundsIndex ]))
    region = np.vstack(polygon.boundary.coords.xy)
    polynomial = list(zip(region[0],region[1]))
    # Setup the polynomial cut:
    geoms = [{'type': 'Polygon', 'coordinates': [polynomial]}];
    # Perform cut
    with rasterio.open(filename) as inputSource:
        output_image, output_transform = mask(inputSource, geoms, crop=True)
    out_meta = inputSource.meta.copy()
    
    # Save Output
    out_meta.update({'driver' : 'GTiff','height': output_image.shape[1],
                     'width': output_image.shape[2],
                     'transform': output_transform})
    with rasterio.open(outputFilename, 'w', **out_meta) as dest:
        dest.write(output_image)
    

if __name__ == '__main__':
    print("run test")