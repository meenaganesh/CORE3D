# How to use this file?
# python generateConfig.py [inputFolder] [outputFolder]
# Replace [inputFolder] by the folder that contains images in .NTF
# [outputFolder] is optional, by fault it is inputFolder/testoutput

import glob
import os
import sys
import json
import gdal
import argparse
from time import gmtime, strftime

# The name of the file to keep all the s2p commands
script_file_name = "s2p_commands.txt"

parser = argparse.ArgumentParser()
parser.add_argument("-e","--entire", help="If -e flag is used, it will generate configs for all subfolders( including WV2/MSI,WV2/PAN,WV3/MSI,WV3/PAN) of input_dir. \nOtherwise, if -e flag is not used, this script works on a single directory that contains all the images and rpc files.",\
	                action="store_true")
parser.add_argument("input_dir", help="The input directory")
parser.add_argument("-o","--out_dir", help="The output directory to keep the result. By default is input_dir/testoutput/")
parser.add_argument("-s","--s2p_dir", help="The folder that contains s2p.py. If no argument, it would use ~/s2p/")
parser.add_argument("-c","--criterion", help="Criterion for grouping the images. 1/ date (by default). 2/ month. 3/ season")
args = parser.parse_args()

if args.input_dir[-1] != '/':
	args.input_dir += '/'

if not args.s2p_dir:
	# Use default s2p path
	args.s2p_dir = "~/s2p/"
else:
	args.s2p_dir = os.path.abspath(args.s2p_dir) + "/"

if not args.out_dir:
	# Use default output folder -- 'inputFolder/testoutput'
	args.out_dir = "./testoutput"
elif args.out_dir[-1] != "/":
	args.out_dir += "/"


# Reverse date, whether the date in NTF and XML file are reversed. By default it is True. But
# for some folder this is actually False. Weird thing.
reverse_date = True

def criterion_same_date(imagePath1,imagePath2):
		'''
		The criterion to be used to group images.
		Return true if two images should be grouped together.
		This version return true when two images are taken on same date
		'''
		date1, date2 = parseImageName(imagePath1)[0], parseImageName(imagePath2)[0]
		return date1 == date2

def criterion_same_month(imagePath1,imagePath2):
		'''
		The criterion to be used to group images.
		Return true if two images should be grouped together.
		This version return true when two images are taken on same month
		'''
		month1, month2 = parseImageName(imagePath1)[0][0], parseImageName(imagePath2)[0][0]
		return month2 == month1

def criterion_same_season(imagePath1,imagePath2):
		'''
		The criterion to be used to group images.
		Return true if two images should be grouped together.
		This version return true when two images are taken on same season
		Month 3 - 10 is regular season, Month 11 - 2 is winter season
		'''
		regular = ["MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT"]
		month1, month2 = parseImageName(imagePath1)[0][0], parseImageName(imagePath2)[0][0]
		if month1 in regular:
			return month2 in regular
		else:
			return month2 not in regular

# This variable determines the criterion to use when running the script
if not args.criterion or "date" in args.criterion:
	criterion = criterion_same_date
elif "month" in args.criterion:
	criterion = criterion_same_month
else:
	criterion = criterion_same_season


def getXMLs(folderName):
	'''
	Given a folder extracted by tar,
	return a list that contains all the absolute paths of its xml files
	'''
	xml_list = []
	dvdDirs = [ dvdDir for dvdDir in glob.glob(folderName + "DVD*/")]
	if len(dvdDirs) == 0:
		print("No DVD folder in " + folderName)
		return []
	for dir in dvdDirs:
		subDir = [ subdir for subdir in glob.glob(dir + "*/")][0]
		subsubDir = subDir.split("/")[-2]
		xmlDir = [ xmldir for xmldir in glob.glob(subDir + subsubDir[:-1] + "*/")][0]
		xmlPaths = [ xmlpath for xmlpath in glob.glob(xmlDir + "*.XML")]
		if len(xmlPaths) == 0:
			print(" No xml file in directory -- " + xmlDir)
			print(" Please double check to make sure the folder structure is correct")
			exit(0)
		xml_list.append(xmlPaths[0])
	return xml_list


def parseImageName(imagePath):
	'''
	Given an absolute image path (with .NTF suffix),
	return a tuple - (date, folderName, fourDigitCode)
	For example, 21JAN15WV031100015JAN21161308-M1BS-501504474040_01_P001_________GA_E0AAAAAAKAAI0.NTF
	will return ( ('JAN',21, 15), '501504474040_01', 'P001') )
	'''
	imageName = imagePath.split('/')[-1]
	date = (imageName[2:5], int(imageName[:2]), int(imageName[5:7]))
	imageInfor = imageName.split('-')[2].split('_',3)
	folderName = imageInfor[0] + '_' + imageInfor[1]
	if reverse_date:
		fourDigitCode = imageInfor[2]
	else:
		fourDigitCode = imageInfor[2][:4]
	return (date, folderName, fourDigitCode)


def parseXML(xmlPath):
	'''
	Given an absolute XML path,
	return a tuple - (date, folderName, fourDigitCode)
	Note that the date is reversed for xml files, so for example,
	15NOV01161954-P1BS-500648062080_01_P001.XML
	will return ( ('NOV', 01, 15), '00648062080_01', 'P001')
	'''
	xmlName = xmlPath.split('/')[-1]
	if reverse_date:
		date = (xmlName[2:5], int(xmlName[5:7]), int(xmlName[:2]))
	else:
		date = (xmlName[2:5], int(xmlName[:2]), int(xmlName[5:7]))
	try:
		xmlInfor = xmlName.split('-')[2].split('_',3)
	except:
		print(xmlName)
		exit(0)
	folderName = xmlInfor[0] + '_' + xmlInfor[1]
	fourDigitCode = xmlInfor[2][:4]
	return (date, folderName, fourDigitCode)

def pairUpImageAndXML(imagesList, xmlList):
	'''
	Given the set of image paths and the set of xml paths,
	pair them up and return as a list of tuple (imagePath:ImageInfor, xmlPath:xmlInfor)
	'''
	result = {}
	imagesDict = { imagePath : parseImageName(imagePath) for imagePath in imagesList}
	xmlDict    = { xmlPath   : parseXML(xmlPath)         for xmlPath   in xmlList}
	for iPath,imageInfor in imagesDict.items():
		for xPath,xmlInfor in xmlDict.items():
			# if "/home/zl279/aerial/ucsd/satellite_imagery/WV2/PAN/" in iPath:
			# 	print(imageInfor)
			# 	print(xmlInfor)
			if imageInfor == xmlInfor:
				result[iPath] = (imageInfor, xPath, xmlInfor)
				break
	return result

def groupByCriterion(imagesDict):
	'''
	Criterion is a function that takes two images and determine
	whether they should be grouped together.
	Returning a list of lists, each sublist represents a set of image group
	'''
	# TODO: Right now I assume the criterion to have 'transitivity' property
	# but in fact it may not be true. Change this function such that it could
	# handle non-transitive criterion

	# BTW, each item only appears in one set of resulting group, that is assuming
	# the transitivity
	result = []
	for iP, (iI, xP, xI) in imagesDict.items():
		if len(result) == 0:
			result.append([{"img" : iP, "rpc" : xP}])
		else:
			for group in result:
				if criterion(iP,group[0]["img"]):
					group.append({"img" : iP, "rpc" : xP})
					break
			# Create a new set
			# print("A new set for " + str(iI[0][0]) + str(iI[0][1]) + str(iI[0][2]))
			result.append([{"img" : iP, "rpc" : xP}])
	return [ set for set in result if len(set) > 1] # Return sets that have >= 2 images

def getRasterSizeFromGroup(group):
	'''
	Return the smallest w and h value from the images of group
	'''
	first_image = group[0]["img"]
	first_g = gdal.Open(first_image)
	smallest_w = first_g.RasterXSize
	smallest_h = first_g.RasterYSize
	for imageDict in group:
		g = gdal.Open(imageDict["img"])
		w = g.RasterXSize
		h = g.RasterYSize
		if smallest_w > w:
			smallest_w = w
		if smallest_h > h:
			smallest_h = h
	return smallest_w, smallest_h


def generateConfigs(folderName,imagesGroup):
	'''
	Generate config for an entire image group under folderName
	Return a list that contains all the s2p command to run.
	'''
	with open(script_file_name,"a") as script_file:
		config_template = {}
		config_template['out_dir'] = args.out_dir
		config_template["full_img"                    ] = 'false'
		config_template["matching_algorithm"          ] = "mgm"
		config_template["horizontal_margin"           ] = 20
		config_template["vertical_margin"             ] = 5
		config_template["sift_match_thresh"           ] = 0.6
		config_template["disp_range_extra_margin"     ] = 0.2
		config_template["n_gcp_per_axis"              ] = 5
		config_template["epipolar_thresh"             ] = 0.5
		config_template["temporary_dir"               ] = "/tmp"
		config_template["tile_size"                   ] = 1024
		config_template["clean_tmp"                   ] = 'true'
		config_template["clean_intermediate"          ] = 'false'
		config_template["fusion_thresh"               ] = 3
		config_template["disable_srtm"                ] = 'true'
		config_template["disp_range_method"           ] = "sift"
		config_template["disp_range_srtm_high_margin" ] = 50
		config_template["disp_range_srtm_low_margin"  ] = -20
		config_template["skip_existing"               ] = 'false'
		config_template["msk_erosion"                 ] = 0
		config_template["dsm_resolution"              ] = 0.5
		config_template["omp_num_threads"             ] = 8
		config_template["debug"                       ] = 'true'
		config_template["max_processes"               ] = 4
		with open(folderName + "generate_config_log.txt", "w+") as log_file:
			group_num = 1
			for group in imagesGroup:
				new_config = config_template.copy()
				new_config['images'] = group
				w, h = getRasterSizeFromGroup(group)
				new_config['roi'] = { "x" : 0, "y" : 0, "w" : w, "h" : h}
				if new_config['out_dir'][-1] != '/':
					new_config['out_dir'] += '/'
				group_date = group[0]['img'].split('/')[-1][:7] # Name out_dir by the date of first image of a group
				if criterion == criterion_same_date:
					group_date += "_BY_DATE"
				elif criterion == criterion_same_month:
					group_date += "_BY_MONTH"
				else:
					group_date += "_BY_SEASON"
				new_config['out_dir'] += group_date
				json_str = json.dumps(new_config, indent=4, separators=(',', ': '))
				if criterion == criterion_same_season:
					json_file_name = folderName + "config_season_"+group_date+".json"
				elif criterion == criterion_same_month:
					json_file_name = folderName + "config_month_"+group_date[2:5]+".json"
				else:
					json_file_name = folderName + "config_date_"+group_date+".json"

				#Generate one config file
				with open(json_file_name, "w+") as json_file:
					json_file.write(json_str)

				# Append to script file
				script_file.write("python " + args.s2p_dir + "s2p.py " + os.path.abspath(json_file_name) + "\n")

				log_file.write(json_file_name + " \n")
				for image in group:
					log_file.write(image["img"] + "\n")
				log_file.write("\n")
				group_num += 1



def processOneFolder(folderName):
	global reverse_date
	if folderName[-1] != "/":
		folderName += "/"
	print("Generating config for directory: " + folderName)
	imagesList  = [ imagePath for imagePath in glob.glob(folderName + '/*.NTF')]

	xmlList     = [ xmlPath for tar_dir in glob.glob(folderName + '/*/') \
	                        for xmlPath in getXMLs(tar_dir)]

	# print(str(imagesList))
	# print(str(xmlList))

	imagesDict_reverse_date   = pairUpImageAndXML(imagesList, xmlList)
	if len(imagesDict_reverse_date) < 5:
		print("WARNING: I suspect that files in " + folderName + " may be ill-formatted because only "+ str(len(imagesDict_reverse_date)) +" pairings are available. \n" +
			  "The reason is most likely that: Image and Xml file has non-reversed date ( For example, 16JAN21*.NTF weirdly corresponds to 16JAN21*.XML).\n" +
			  "Now reverse the date of all xml files under this folder in order for a better pairing result.\n" +
			  "But please double check if this is indeed the case.\n")
		reverse_date = False
		imagesDict_no_reverse_date = pairUpImageAndXML(imagesList, xmlList)
		reverse_date = True

		if len(imagesDict_reverse_date) < len(imagesDict_no_reverse_date):
			print("RESULT: Not reversing date yields better result\n")
			imagesDict = imagesDict_no_reverse_date
		else:
			print("RESULT: Reversing date yields better result\n")
			imagesDict = imagesDict_reverse_date
	else:
		imagesDict = imagesDict_reverse_date
	# print(str(len(imagesList)) + " " + str(len(xmlList)) + " " + str(len(imagesDict)) + " for " + folderName)

	# Print for debugging
	# for iP in imagesList:
	# 	print(iP)
	# for tar_dir in glob.glob(folderName + '/*/'):
	# 	print(tar_dir)
	# for xml in xmlList:
	# 	print(xml)
	# for iP in imagesDict:
	# 	imageInfor, xP, xmlInfor = imagesDict[iP]
	# 	print(iP + " ***is paired with*** " + xP)

	imagesGroup  = groupByCriterion(imagesDict) # A list of list of dictionaries that represent images


	generateConfigs(folderName, imagesGroup)


def processEntireFolder(folderName):
	if folderName[-1] != "/":
		folderName += "/"
	if len(glob.glob(folderName + "satellite_imagery/")) > 0:
		folderName += "satellite_imagery/"
	subfolders = glob.glob(folderName + 'WV*/')
	if len(subfolders) == 0:
		print("No WV2 or WV3 folder under " + folderName)
	imgfolders = []
	for wvfolder in subfolders:
		imgfolders += glob.glob(wvfolder + "MSI/")
		imgfolders += glob.glob(wvfolder + "PAN/")
	if len(imgfolders) == 0:
		print("No MSI or PAN folder is found")
		exit(0)
	for imgfolder in imgfolders:
		processOneFolder(imgfolder)




if __name__ == "__main__":
	print("Start generating config files..")
	with open(script_file_name,"a") as script_file:
		script_file.write("\n\n----------New scripts generated on " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "----------\n")
	if not args.entire:
		processOneFolder(args.input_dir)
	else:
		print("Generating config for all subfolders under directory: " + args.input_dir)
		processEntireFolder(args.input_dir)
	print("Successfully generated all config files! \nPlease use commands in 's2p_commands.txt' to start processing!")
