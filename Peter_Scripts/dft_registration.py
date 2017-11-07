# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:29:45 2017

@author: 212420318

"""

import os
import scipy as sp
#import scipy.misc
import imreg_dft as ird

# the template
im0 = sp.misc.imread('img_test_0.png', True)
# the image to be transformed
im1 = sp.misc.imread('img_test_1.png', True)
result = ird.similarity(im0, im1, numiter=10)

assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    ird.imshow(im0, im1, result['timg'])
    plt.show()