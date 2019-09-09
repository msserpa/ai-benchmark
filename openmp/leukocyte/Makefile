all:
	gcc -o a.out meschach_lib/meschach.a avilib.c find_ellipse.c track_ellipse.c misc_math.c detect_main.c -Imeschach_lib  -O2 -lm -fopenmp -lpthread -Wall -Wextra -Wno-unused-result

meschach_lib/meschach.a:
	cd meschach_lib; ./configure --with-all; make all;