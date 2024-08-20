nvcc -I./third_party/glm/glm -c render.cu -o camera.o
g++ -o main render.cpp camera.o -L/usr/local/cuda/lib64 -lcudart `pkg-config --cflags --libs opencv4`
./main
