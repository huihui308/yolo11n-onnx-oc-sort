



https://github.com/microsoft/onnxruntime/releases 下载：onnxruntime-linux-aarch64-1.24.4.tgz


### Install package

```shell
tar zxvf onnxruntime-linux-aarch64-1.24.4.tgz

sudo mv onnxruntime-linux-aarch64-1.24.4 /opt/onnxruntime

sudo apt install ninja-build

sudo apt install libeigen3-dev
```


--------------------------


### Compile
```shell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

cmake --build build -j4
```


--------------------------


### Running
```shell
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

./build/live_ocsort_tracker
or
./build/live_ocsort_tracker ./video/04150947.mp4-new.mp4 ./model/yolo11n_640.onnx
```


--------------------------


### Reference

https://github.com/Vinayak07Kushwaha/OC-Sort


