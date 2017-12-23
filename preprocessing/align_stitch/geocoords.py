import os
import csv
import affine
from osgeo import gdal
import argparse


def output_geo(img_file, pix_file, linenum):
    ds = gdal.Open(img_file)
    geo = ds.GetGeoTransform()
    a = affine.Affine.from_gdal(*geo)

    line = 1
    with(open(pix_file)) as p:
        reader = csv.reader(p, delimiter='\t')
        for row in reader:
            x, y = a * (float(row[0]), float(row[1]))
            if linenum:
                print("{}\t{}\t{}".format(line,x,y))
            else:
                print("{}\t{}".format(x,y))
            line = line + 1


if __name__ == "__main__":

    def extant_file(x):
        """
        for argparse - checks that file exists but does not open.
        """
        if not os.path.exists(x):
            # error: argument input: x does not exist
            raise argparse.ArgumentTypeError("{0} does not exist".format(x))
        return x


    parser = argparse.ArgumentParser(description='Output pixels in reference coordinate system.')
    parser.add_argument("-i", dest="image", required=True, type=extant_file,
                        help="input image file", metavar="FILE")
    parser.add_argument("-p", dest="pix", required=True, type=extant_file,
                        help="pixel coordinates", metavar="FILE")
    parser.add_argument('-l', dest="linenum", action='store_true',  help = 'output line numbers')

    args = parser.parse_args()
    output_geo(args.image, args.pix, args.linenum)
