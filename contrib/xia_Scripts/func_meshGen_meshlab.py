### Mesh generation using meshlabserver

import os, sys
from subprocess import call
import numpy as np
#from plyfile import PlyData, PlyElement
import time

def func_meshGen(inDir, filename):

    # check if files exist
    tifFile = os.path.join(inDir, filename)
    if not os.path.exists(tifFile):
        print('Input file does not exist.')
        return

    mlx_normalEst_file = '/raid/data/xia/bin/normalEst.mlx'
    mlx_meshGen_file = '/raid/data/xia/bin/poissonRec.mlx'

    
    # generate filenames        
    prefixStr = filename.split('.')[0]
    xyzFile = os.path.join(inDir, prefixStr + '_meshlab.xyz')    
    xyzNormFile = os.path.join(inDir, prefixStr + '_norm_meshlab.xyz')
    plyFile = os.path.join(inDir, prefixStr + '_meshlab.ply') 

    tStart = time.time()

    # Step 1: convert geotiff to xyz
    print('Converting geotiff to xyz... '),
    cmd = 'gdal_translate -of XYZ {} {}'.format(tifFile, xyzFile)
    call(cmd, shell=True)
    print('Done.')

    # Step 2: normal estimation
    print('Estimating normals... '),
    cmd = 'meshlabserver -i {} -o {} -s {} -om vn'.format(xyzFile, xyzNormFile, mlx_normalEst_file)
    call(cmd, shell=True)
    print('Done.')

    # Step 3: mesh generation, also save vertex normal
    print('Mesh generation ... '),
    cmd = 'meshlabserver -i {} -o {} -s {} -om vn'.format( xyzNormFile, plyFile, mlx_meshGen_file)
    call(cmd, shell=True)
    print('Done.')
    
    print('Running time: %.2f minutes. ' % ((float(time.time()) - float(tStart))/60 ))
 

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python func_meshGen_meshlab.py <input Dir> <filename>')
        sys.exit()
        
    func_meshGen(sys.argv[1], sys.argv[2])
