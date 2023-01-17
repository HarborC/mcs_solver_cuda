echo "Configuring and building ..."

mkdir release
cd release 
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8