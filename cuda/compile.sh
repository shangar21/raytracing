nvcc -c camera.cu -o camera.o
g++ -o main main.cpp camera.o -L/usr/local/cuda/lib64 -lcudart `pkg-config --cflags --libs opencv4`
./main
