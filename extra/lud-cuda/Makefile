NVCC := nvcc

DEFS += \
				$(SPACE)

NVCCFLAGS += \
						 -O3 \
						 -use_fast_math \
						 -arch=sm_13 \
						 -lm \
						 $(SPACE)


# Add source files here
EXEC  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES := lu.cu lud_kernel_1.cu lud_kernel_2.cu lud_kernel_3.cu lud_kernel_4.cu lud_kernel_5.cu lud_kernel_6.cu 
OBJS := lu.o lud_kernel_1.o lud_kernel_2.o lud_kernel_3.o lud_kernel_4.o lud_kernel_5.o lud_kernel_6.o 


.PHONY: all clean 
all : $(EXEC)


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(DEFS) -o $@ -c $<


$(EXEC) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@  $^

clean:
	rm -f $(EXEC) $(OBJS) *.linkinfo
