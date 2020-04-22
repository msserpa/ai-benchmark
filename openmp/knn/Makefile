ifndef COMPILER
$(warning Try one of these compiler options: )
$(info make COMPILER=gcc)
$(info make COMPILER=icc)
$(info make COMPILER=ncc)
$(error COMPILER is not set)
endif

LDFLAGS= -lm
CFLAGS=-Wall

ifeq ($(COMPILER), gcc)
	CC=gcc
	CFLAGS+=-O2 -Wextra -Wno-unused-result -fopenmp -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a
endif

ifeq ($(COMPILER), icc)
	CC=icc
	CFLAGS+=-Wextra -qopenmp -ipo -xCORE-AVX512 -qopt-zmm-usage=high -qopt-report=4 -qopt-report-phase ipo
endif

ifeq ($(COMPILER), ncc)
	CC=/opt/nec/ve/bin/ncc
	CFLAGS+=-fopenmp
endif

all:
	$(CC) $(CFLAGS) knn.c $(LDFLAGS) -o ../../bin/knn.$(COMPILER)