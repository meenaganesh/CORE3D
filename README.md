# CORE3D

Generate aligned tiles
------------------------------------------------------------------------

## Step 1: Pull the image
Please get the core3d docker image from docker hub. Note the docker image with the latest
tag is updated with some frequency, so please be sure to periodically repeat the docker
pull command: 
```bash
docker pull core3d/core3d:latest
```

## Step 2: Launch the container
Run the shell script "dev.sh". This will launch the a bash shell inside the dev container.
You may wish to adjust this script if you would like to make additional host directories
available to the container.  By default it maps in your home directory for the host.
```bash
./dev.sh
```

## Step 3: Generate the tiles
Tiles are now generated using a quad tree referencing scheme. The quad tree maps all geographic regions of the 
wold to to a corresponding row, column and zoom based, given a geo reference. For the purpose of this tile set
the zoom has been fixed to a zoom of 17, which yields 1024x1024 tiles with a resolution of 
2.68e-6 deg. lat and lon per pixel.  All images will be up-sampled to that resolution so that coordinates in
pixel space will align between different perspectives and modalities. 

The current output directory structure for this quad tree is a relatively flat structure, with a directory for
each ROW_COLUMN the tiles were generated for.

Once you are in the container, run:
```bash
python aoi.py -i your_config.yml
```

The YAML configuration file provides the tiling routines with the set of files to process.  An example YAML file
has been provided (aoi.yml), which includes additional comments on the various configuration parameters.

 - if python complains missing path, please simply "export PYTHONPATH=path_to_CORE3d_on_your_host"
 - for 3D laz, we are presently only generating the Z values into statistical bands (min, max, mean, idx, count, stdev).
   Soon the configuration will be updated to all you to specify the desired dimension and bands.

## Examine the output:

In your output directory - you will see a set of folders. Corresponding to the rows and columns of
quad tree. For example:

```bash
35787_53938  35793_53936  35799_53934  35804_53939  35810_53937  35816_53935
35787_53939  35793_53937  35799_53935  35804_53940  35810_53938  35816_53936
35787_53940  35793_53938  35799_53936  35805_53934  35810_53939  35816_53937
```

Inside each ROW_COL folder, there will be a set of raster images that represent the tiles cut from
the source images that intersected with the bounding box of the quad tree cell.

### Naming conventions:
The directory will contain files who's names are derived from the source image name. Along the way, various
intermediate iles will be produced with additional tags added to the filename.

- name.tif - (intermediate file) slightly larger than desired. The oversized tiles are intermediate files that allow the
image to be translated during registration - before being croped to the desired size.
- name_reg.tif - (intermediate file) oversized image that has been registered to a reference image
- name_reg_cut.tif - output tile - that has been registed and trimed to desired size (1024x1024).
- name_reg_cut_ps.tif - output tile - pan sharpened image.
- name_reg_cut_ps_rgb.tif - output tile - pan sharpened image - with just rgb bands.


