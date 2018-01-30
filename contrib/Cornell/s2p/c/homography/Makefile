CXXFLAGS ?= -march=native -O3
GDAL_CONFIG=/phoenix/S2/snavely/local/bin/gdal-config
override CXXFLAGS := $(CXXFLAGS) `${GDAL_CONFIG} --cflags`
override CXXLIBS  := $(CXXLIBS) `${GDAL_CONFIG} --libs`

OBJ = main.o \
      LibImages/LibImages.o \
      Utilities/Utilities.o \
      Utilities/Memory.o \
      Utilities/Parameters.o \
      Utilities/Time.o \
      LibHomography/Homography.o \
      LibHomography/Splines.o

homography: $(OBJ)
	$(CXX) -o $@ $^ $(CXXLIBS)

clean:
	rm -f homography $(OBJ)
