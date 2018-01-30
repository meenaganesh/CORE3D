CXXFLAGS ?= -march=native -O3
GDAL_CONFIG=/phoenix/S2/snavely/local/bin/gdal-config
override CXXFLAGS := $(CXXFLAGS) `${GDAL_CONFIG} --cflags`
override CXXLIBS  := $(CXXLIBS) `${GDAL_CONFIG} --libs`
override LDLIBS   := $(LDLIBS) -lm

OBJ = sift_roi.o \
      LibImages/LibImages.o \
      LibSSE/LibSSE.o \
      LibSift/KeyPoint.o \
      LibSift/LibSift.o \
      LibSift/Octave.o \
      LibSift/ScaleSpace.o \
      Utilities/Memory.o \
      Utilities/Parameters.o \
      Utilities/Time.o \
      Utilities/Utilities.o \

all: sift_roi matching

sift_roi: $(OBJ)
	$(CXX) -o $@ $^ $(CXXLIBS)

clean:
	rm -f sift_roi matching $(OBJ)
