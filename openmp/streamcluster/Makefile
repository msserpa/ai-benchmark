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
	CC=g++
	CFLAGS+=-O2 -Wextra -Wno-unused-result -fopenmp -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a
endif

ifeq ($(COMPILER), icc)
	CC=icpc
	CFLAGS+=-Wextra -qopenmp -ipo -xCORE-AVX512 -qopt-zmm-usage=high -qopt-report=4 -qopt-report-phase ipo
endif

ifeq ($(COMPILER), ncc)
	CC=/opt/nec/ve/bin/n++
	CFLAGS+=-fopenmp
endif

all:
	$(CC) $(CFLAGS) streamcluster.cpp $(LDFLAGS) -o ../../bin/streamcluster.$(COMPILER)