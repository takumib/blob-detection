#all:  blackout cuda_blackout webcam_display webcam_frame security video_edge edge window

#window:
#	gcc window.c -o ./bin/window `pkg-config --cflags --libs opencv`

#security:
#	gcc opencv_security.c -o ./bin/security `pkg-config --cflags --libs opencv`

#blackout: 
#	gcc blackout.c -o ./bin/blackout `pkg-config --cflags --libs opencv`

#cuda_blackout:
	#nvcc  cuda_blackout.cu -o ./bin/cuda_blackout `pkg-config --cflags --libs opencv`

SOURCES := \
	opencv_webcam_display.c \
	harris_cpu.cpp 

CUDASOURCES := \
	../src/harris_gpu.cu \
	../src/harris_device.cu \

EXTRADEPS := \
	harris_cpu.hpp \
	#../src/harris_host.hpp \

webcam_display:
	g++ -c opencv_webcam_display.c harris_cpu.cpp -I/opt/cuda/sdk/C/common/inc -L/opt/cuda/sdk/C/common/lib -L/usr/lib/nvidia-current `pkg-config --cflags --libs opencv` `pkg-config --cflags --libs gimpui-2.0`
	nvcc opencv_webcam_display.o harris_cpu.o harris_gpu.cu harris_gpu_op.cu harris_gpu_shared_op.cu -o ./bin/webcam_display `pkg-config --cflags --libs opencv`
#webcam_frame:
#	gcc opencv_webcam_frame.c  -o ./bin/webcam_frame `pkg-config --cflags --libs opencv` 

#canny_naive:
#	gcc opencv_canny_naive.c -o ./bin/opencv_canny_canny `pkg-config --cflags --libs opencv`

#video_edge:
#	gcc opencv_video_edge_test.c -o ./bin/video_edge `pkg-config --cflags --libs opencv` 

#edge:
#	gcc opencv-edge.c -o edge `pkg-config --cflags --libs opencv`

#test:
#	make clean
#	make 
#	./bin/window /scratch/images/lenna.png /scratch/images/Lenna_Query.png

#clean:
#	rm -rf ./bin
#	mkdir bin
