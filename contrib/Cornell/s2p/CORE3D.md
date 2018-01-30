This README has two parts: testing the script on CORE3D data, and
running our script for generating config.json files from a CORE3D
directory.

# Testing the pipeline on CORE3D data

To test out the pipeline on actual CORE3D data, try the following:

 1. Image data. Copy these two images to the test_jacksonville
    subdirectory:

    - 05OCT14WV031100014OCT05160138-P1BS-500648062040_01_P001_________AAE_0AAAAABPABL0.NTF
    - 05OCT14WV031100014OCT05160149-P1BS-500648061080_01_P001_________AAE_0AAAAABPABL0.NTF

    These two files are in:

    - 170623-CORE3D-Performer-Data-Package/performer_data/performer_source_data/jacksonville/satellite_imagery/WV3/PAN/05OCT14WV031100014OCT05160138-P1BS-500648062040_01_P001_________AAE_0AAAAABPABL0.NTF

    and

    - 170623-CORE3D-Performer-Data-Package/performer_data/performer_source_data/jacksonville/satellite_imagery/WV3/PAN/05OCT14WV031100014OCT05160149-P1BS-500648061080_01_P001_________AAE_0AAAAABPABL0.NTF

    respectively.

 2. RPC metadata. Copy the tar files corresponding to these to image 
    to the test_jacksonville subdirectory. These tar files are named:

    - 05OCT14WV031100014OCT05160138-P1BS-500648062040_01_P001_________AAE_0AAAAABPABL0.tar
    - 05OCT14WV031100014OCT05160149-P1BS-500648061080_01_P001_________AAE_0AAAAABPABL0.tar

    and are located at:

    - 170623-CORE3D-Performer-Data-Package/performer_data/performer_source_data/jacksonville/satellite_imagery/WV3/PAN/05OCT14WV031100014OCT05160138-P1BS-500648062040_01_P001_________AAE_0AAAAABPABL0.tar
    - 170623-CORE3D-Performer-Data-Package/performer_data/performer_source_data/jacksonville/satellite_imagery/WV3/PAN/05OCT14WV031100014OCT05160149-P1BS-500648061080_01_P001_________AAE_0AAAAABPABL0.tar

    Untar these tar files. The RPC data will be found in the following
    extracted XML files (mentioned in the config8K.json file):

    - 500648062040_01/DVD_VOL_1/500648062040_01/500648062040_01_P001_PAN/14OCT05160138-P1BS-500648062040_01_P001.XML
    - 500648061080_01/DVD_VOL_1/500648061080_01/500648061080_01_P001_PAN/14OCT05160149-P1BS-500648061080_01_P001.XML

 3. Run the script, capturing any output errors to a log file:

```
    > python s2p.py test_jacksonville/config8K.json > output.log 2>&1
```

# Running the script to generate config files

To run the s2p.py script on a CORE3D data directory, you can use the
configGenerator.py script in the root repository directory. You can do 

```
    > python configGenerator.py --help
```

to see the usage for this script. Here is a summary of how it works:

  1. The only required argument is the input directory. For example,
     for this script to process a single folder
     "ucsd/satellite_images/WV2/PAN", just type:

     > python configGenerator.py ucsd/satellite_images/WV2/PAN

  2. However, the script also outputs a "s2p_commands.txt" that
     contains the list of Python commands to run, so it has to know
     where your s2p folder is. By default, it assumes the s2p folder
     is ~/s2p/, but if you want to modify the path, then use "-s" flag
     with a supported argument( it may be more convenient to make a
     symbolic link for your s2p folder, so you won't need this flag):

     > python configGenerator.py ucsd/satellite_images/WV2/PAN -s [your
       own s2p directory path]

  3. Say you want the script to process all subfolders under 'ucsd/',
     then use "-e" flag which stands for "--entire":

     > python configGenerator.py ucsd -e

  4. By default, the json files this script generates will have the
     'out_dir' option set to './testoutput/XXX'. If you want to modify the
     out_dir option in these config files, then use "-o" flag with an
     argument which stands for "out_dir":

     > python configGenerator.py ucsd -e -o [Your desired output
       directory]

  5. Finally, the current criterion that I used is to group images by
     same date. If you want to change this, then use "-c" flag, which
     stands for "--criterion", with an supported argument, which could
     be one of ['date','month','season']:

     > python configGenerator.py ucsd -e -c month    
