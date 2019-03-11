CC=gcc

INCDIR = meschach_lib
LIBDIR = meschach_lib

FLAGS = -g -O3 -I$(INCDIR)

# make dbg=1 enables nvcc debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g
endif

# make emu=1 compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

detect_test: detect_main.o avilib.o find_ellipse.o misc_math.o find_ellipse_kernel.o
	$(CC) $(FLAGS) avilib.o find_ellipse_kernel.o find_ellipse.o misc_math.o detect_main.o -o detect_test $(LIBDIR)/meschach.a -L/usr/local/cuda/lib -L../../common/lib -lm -lcuda -lcudart

detect_main.o: detect_main.c
	$(CC) $(FLAGS) detect_main.c -c -I/usr/local/cuda/include

avilib.o: avilib.c
	$(CC) $(FLAGS) avilib.c -c

find_ellipse.o: find_ellipse.c find_ellipse.h
	$(CC) $(FLAGS) find_ellipse.c -c

find_ellipse_kernel.o: find_ellipse_kernel.cu find_ellipse_kernel.h
	nvcc $(NVCC_FLAGS) -O3 -I/usr/local/cuda/include -I../../common/inc -c find_ellipse_kernel.cu

misc_math.o: misc_math.c
	$(CC) $(FLAGS) misc_math.c -c

clean:
	rm -f *.o *~ detect_test test_avi_stuff test_gradient





