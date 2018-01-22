### Poisson surface reconstruction from http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version9.011/


import os, sys
from subprocess import call
import numpy as np
import time


def func_poissonRecon(inDir, filename):

    normalEstBin = '/raid/data/xia/bin/normalEst'
    poissonBin = '/raid/data/xia/bin/PoissonRecon'

    # check if the input file exists
    tifFile = os.path.join(inDir, filename)
    if not os.path.exists(tifFile):
        print('Input file does not exist.')
        return

    # generate filenames        
    prefixStr = filename.split('.')[0]
    xyzFile = os.path.join(inDir, prefixStr + '.xyz')
    xyzDemeanFile = os.path.join(inDir, prefixStr + '_demean.xyz')
    xyzNormFile = os.path.join(inDir, prefixStr + '_norm.xyz')
    plyFile = os.path.join(inDir, prefixStr + '.ply') 


    tStart = time.time()

    # Step 1: convert geotiff to xyz
    print('Converting geotiff to xyz... '),
    cmd = 'gdal_translate -of XYZ {} {}'.format(tifFile, xyzFile)
    call(cmd, shell=True)
    print('Done.')

    # Step 2: demean xyz
    data = np.loadtxt(xyzFile)
    mXYZ = np.expand_dims(np.mean(data,axis = 0),1)
    mdata = np.repeat(mXYZ.T,data.shape[0],axis=0)
    data_demean = data - mdata
    np.savetxt(xyzDemeanFile,data_demean, fmt='%.8f')

    # Step 3: normal estimation
    print('Estimating normals... '),
    cmd = '{} {} {}'.format(normalEstBin, xyzDemeanFile, xyzNormFile)
    call(cmd, shell=True)
    print('Done.')

    # Step 4: put mean back
    data = np.loadtxt(xyzNormFile)
    data[:,:3] = data[:,:3] + mdata
    np.savetxt(xyzNormFile,data)

    # Step 5: Poisson surface reconstruction
    print('Poisson surface reconstruction... '),
    cmd = '{} --in {} --out {} --depth 10 '.format(poissonBin, xyzNormFile, plyFile)
    call(cmd, shell=True)
    print('Done.')

    print('Running time: %.2f minutes. ' % ((float(time.time()) - float(tStart))/60 ))
 

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python func_poissonRecon.py <input Dir> <filename>')
        sys.exit()
        
    func_poissonRecon(sys.argv[1], sys.argv[2])
