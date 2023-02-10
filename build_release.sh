echo "Configuring and building ..."

mkdir release
cd release 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc
make -j8