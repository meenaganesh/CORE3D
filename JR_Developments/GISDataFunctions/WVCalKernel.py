import numpy as np
import tarfile
import gdal
import re
import warnings


def get_band_info(filename):
    """
    Extract values required in the calibration formula from the .IMD file in a .tar directory.
    
    NOTE: DEPENDS ON PYTHON VERSION. See line comments.
    
    :param filename: name of .tar file corresponding to .NTF to calibrate
    :return: tuple of band labels, absCalFactors, and effectiveBandwidths
    """
    tar_data = tarfile.open(filename)
    member_names = tar_data.getnames()
    imd_loc = [i for i, fname in enumerate(member_names) if '.IMD' in fname]

    # PYTHON VERSION 2.7
    imd_data = [line.strip() for line in tar_data.extractfile(member_names[imd_loc[0]]).read().split('\n')]  # Takes first IMD file found for now. I do not know if there is ever more than one.
    # PYTHON VERSION >= 3.3 comment out above line, uncomment below
    # imd_data = [line.strip() for line in tar_data.extractfile(member_names[imd_loc[0]]).read().decode('utf-8').split('\n')]

    band_labels = []
    abs_cal_factors = {}
    effective_bandwidths = {}
    i = 0
    while i < len(imd_data):
        if 'BEGIN_GROUP' in imd_data[i] and 'BAND' in imd_data[i]:
            band_labels.append(imd_data[i].split('_')[-1])
            i += 1
            while 'END_GROUP' not in imd_data[i]:
                if 'absCalFactor' in imd_data[i]:
                    abs_cal_factors[band_labels[-1]] = float(re.findall('\d+\.\d+', imd_data[i])[0])
                if 'effectiveBandwidth' in imd_data[i]:
                    effective_bandwidths[band_labels[-1]] = float(re.findall('\d+\.\d+', imd_data[i])[0])
                i += 1
        else:
            i += 1
    return band_labels, abs_cal_factors, effective_bandwidths


# copied directly from Johan's helpers code.
def loadRasters(filename):
    ds = gdal.Open(filename)
    xSize = ds.RasterXSize
    ySize = ds.RasterYSize
    rasterCount = ds.RasterCount
    data = np.zeros((ySize, xSize, rasterCount))
    for rasterCounter in range(1, rasterCount+1):
        data[:, :, rasterCounter-1] = ds.GetRasterBand(rasterCounter).ReadAsArray()
    return data


class WV3params:

    def __init__(self):

        self.band_lookup = ["P", "C", "B", "G", "Y", "R", "RE", "N", "N2", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]

        self.gain = {"P": 0.950,
                     "C": 0.905,
                     "B": 0.940,
                     "G": 0.938,
                     "Y": 0.962,
                     "R": 0.964,
                     "RE": 1.000,
                     "N": 0.961,
                     "N2": 0.978,
                     "S1": 1.200,
                     "S2": 1.227,
                     "S3": 1.199,
                     "S4": 1.196,
                     "S5": 1.262,
                     "S6": 1.314,
                     "S7": 1.346,
                     "S8": 1.376}

        self.offset = {"P": -3.629,
                       "C": -8.604,
                       "B": -5.809,
                       "G": -4.996,
                       "Y": -3.649,
                       "R": -3.021,
                       "RE": -4.521,
                       "N": -5.522,
                       "N2": -2.992,
                       "S1": -5.546,
                       "S2": -2.600,
                       "S3": -2.309,
                       "S4": -1.676,
                       "S5": -0.705,
                       "S6": -0.669,
                       "S7": -0.512,
                       "S8": -0.372}


class WV2params:

    def __init__(self):

        self.band_idx = ["P", "C", "B", "G", "Y", "R", "RE", "N", "N2"]

        self.gain = {"P": 0.942,
                     "C": 1.151,
                     "B": 0.988,
                     "G": 0.936,
                     "Y": 0.949,
                     "R": 0.952,
                     "RE": 0.974,
                     "N": 0.961,
                     "N2": 1.002}

        self.offset = {"P": -2.704,
                       "C": -7.478,
                       "B": -5.736,
                       "G": -3.546,
                       "Y": -3.564,
                       "R": -2.512,
                       "RE": -4.120,
                       "N": -3.300,
                       "N2": -2.891}


class RadiometricCalibrator:

    def __init__(self, raster_file, satellite_params, sub_array=None):
        """
        :param raster_file: file name with full path to .NTF file to calibrate
        :param satellite_params: one of the above WV satellite objects corresponding to satellite of the .NTF
        :param sub_array: optional. numpy ndarray respresentation of a raster
        """
        self.raster_file = raster_file
        self.tar_file = raster_file[:-4] + '.tar'
        self.gains = satellite_params.gain
        self.offset = satellite_params.offset
        try:
            self.band_labels, self.abs_cal_factors, self.effective_bandwidths = get_band_info(self.tar_file)
        except IOError:
            warnings.warn(".tar not found. band_labels, abs_cal_factors, and effective_bandwidths not set. Set tar file manually with set_tar() before calibrating.")
        if sub_array is None:
            self.raster_array = loadRasters(raster_file)
        else:
            self.raster_array = sub_array
        self.calibrated_array = np.zeros(self.raster_array.shape)

    def set_tar(self, tar_file):
        self.tar_file = tar_file
        try:
            self.band_labels, self.abs_cal_factors, self.effective_bandwidths = get_band_info(self.tar_file)
        except IOError:
            warnings.warn(".tar not found. band_labels, abs_cal_factors, and effective_bandwidths not set.")

    def calibrate(self):
        for i in range(self.raster_array.shape[2]):
            band_id = self.band_labels[i]
            self.calibrated_array[:, :, i] = self.raster_array[:, :, i] * self.gains[band_id] * (self.abs_cal_factors[band_id] / self.effective_bandwidths[band_id]) + self.offset[band_id]

    def get_calibrated_data(self):
        return self.calibrated_array
