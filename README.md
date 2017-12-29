# CORE3D

TO generate stitched and aligned tiles, go to preprocessing/align_stitch
------------------------------------------------------------------------

Step 1:
Please get the core3d docker image from docker hub 
```bash
docker pull core3d/core3d:latest
```

Step 2:
Run the shell script "dev.sh"
```bash
./dev.sh
```

Step 3:
Tiles are now generated using a quad tree referencing scheme. Each tile that is generated will belong
to a particular row and column based on a given geo reference.

Once you are in the container, run "python aoi.py":
 - if python complains missing path, please simply "export PYTHONPATH=path_to_CORE3d_on_your_host"
 - please note that input and output paths in aoi.py needs to be changed
 - See inline documentation at the end of aoi.py to control what input is processed, and what regions tiles are 
   generated for.
 - for 3D laz, we only generate the Z values into a raster image, but a laz contains a lot of information, and you can            
   modify line 131 "type" to generate raster of the 3D laz with more information than Z, such as intensity 

This will generate aligned tiles among the different modalities, e.g., wv2_pan_0_0.tif and wv2_msi_0_0.tif that are 
generated in the output dir are all aligned tiles.

Step 4:
Still in the container, run "python RasterModalityFilter.py":
 - this will generate patches from each set of corresponding tiles that contains all the modalities that you want
