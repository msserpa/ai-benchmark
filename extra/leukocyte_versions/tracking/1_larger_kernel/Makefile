CC = gcc

INCDIR = meschach_lib/
LIBDIR = meschach_lib/

FLAGS = -g -O3 -Wall -I$(INCDIR)
NVCC_FLAGS = -I/usr/local/cuda/include -I../../common/inc

# make dbg=1 enables nvcc debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

# make emu=1 compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif


detect_test: detect_main.o avilib.o find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o
	$(CC) $(FLAGS) -lm avilib.o find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o detect_main.o -o detect_test -lm $(LIBDIR)meschach.a -L/usr/local/cuda/lib -L../../common/lib -lm -lcuda -lcudart

%.o: %.[ch]
	$(CC) $(FLAGS) $< -c

detect_main.o: detect_main.c find_ellipse.h track_ellipse.h avilib.h 
	$(CC) $(FLAGS) detect_main.c -c -I/usr/local/cuda/include

find_ellipse_kernel.o: find_ellipse_kernel.cu find_ellipse_kernel.h
	nvcc $(NVCC_FLAGS) -c find_ellipse_kernel.cu

track_ellipse_kernel.o: track_ellipse_kernel.cu track_ellipse_kernel.h
	nvcc $(NVCC_FLAGS) -I$(INCDIR) -c track_ellipse_kernel.cu

find_ellipse.o: avilib.h
track_ellipse.o: find_ellipse.h avilib.h

clean:
	rm -f *.o *~ detect_test





