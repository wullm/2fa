#Compiler options
GCC = mpicc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3 -lfftw3_omp -lfftw3_mpi
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -I/usr/include/hdf5/openmpi

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES)
CFLAGS = -Wall -Wshadow=global -fopenmp -march=native -O4
LDFLAGS =

ELEMENTS = input cosmology_tables perturb_data titles primordial fluid_equations
PROGRAMS = 3fa

OBJECTS=$(patsubst %, lib/%.o, $(ELEMENTS))
OBJECT_SOURCES=$(patsubst %, src/%.c, $(ELEMENTS))

$(PROGRAMS): % : src/%.c $(OBJECTS) $(INI_PARSER) include/*.h
	$(GCC) src/$@.c -o $@ $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

lib/%.o: src/%.c include/*.h
	$(GCC) $< -c -o $@ $(INCLUDES) $(CFLAGS)

all: minIni $(PROGRAMS)
	
minIni:
	cd parser && make

clean:
	rm -f lib/*.o
	rm -f 3fa
	rm -f parser/minIni.o
