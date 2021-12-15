# Installing aihwkit
## Use excisiting virtual environment
Use the working (most of the times) python virtual environment at Users/zyu/virtual_env/aihwkit_env

```
source /Users/zyu/virtual_env/aihwkit_env/bin/activate
```

## Compile and install aihwkit
### 1.Make virtual envirment
```
python3 -m venv aihwkit
cd aihwkit
source bin/activate
```
### 2.Download library from source
```
git clone https://github.com/IBM/aihwkit.git
cd aihwkit
```
### 3.Install pytorch 1.8.1
This is required for aihwkit 0.4.0 pre-compiled wheel.
```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
### 4.Install compile dependencies
On our server, this first step is already installed.
```
sudo apt-get install python3-dev libopenblas-dev
```
Only the second step needed for the virtual_env.
```
pip install cmake scikit-build torch pybind11
```
### 5.Test compilation
For compiling and installing with CUDA support on RTX30X0s(sm86):
```
python setup.py build_ext --inplace -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES=86
```
or if using cmake directly:
```
build$ cmake -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES=86 ..
```
In our server system, this should end with an error complaining about "Couldn't find CUDA library root".
This is because that the CUDA compiler installation was not in a derectory that cmake thought.
Edit the CMakeDetermineCUDACompiler.cmake file.
```
vim ${virtual_env_dir}/aihwkit/lib/python3.8/site-packages/cmake/data/share/cmake-3.22/Modules/CMakeDetermineCUDACompiler.cmake
```
Change line 200 from
```
199   #We require the path to end in `/nvvm/libdevice'
200   if(_CUDA_NVVMIR_LIBRARY_DIR MATCHES "nvvm/libdevice$")
201     get_filename_component(_CUDA_NVVMIR_LIBRARY_DIR "${_CUDA_NVVMIR_LIBRARY_DIR}/../.." ABSOLUTE)
202     set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR "${_CUDA_NVVMIR_LIBRARY_DIR}")
203   endif()
```
to
```
199   #We require the path to end in `/nvvm/libdevice'
200   if(_CUDA_NVVMIR_LIBRARY_DIR MATCHES "libdevice$")
201     get_filename_component(_CUDA_NVVMIR_LIBRARY_DIR "${_CUDA_NVVMIR_LIBRARY_DIR}/../.." ABSOLUTE)
202     set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR "${_CUDA_NVVMIR_LIBRARY_DIR}")
203   endif()
```
by deleting "nvvm/"
### 6.Install aihwkit pre-compiled wheel
This is required for pip to figure out relevent dependencies.
```
pip install -v aihwkit
```
### 7.Uninstall aihwkit
```
pip uninstall aihwkit
```
### 8.Install again by compiling with CUDA support
```
pip install -v aihwkit --install-option="-DUSE_CUDA=ON" --install-option="-DRPU_CUDA_ARCHITECTURES=86"
```
If the comand does not show similar prograss report as in cmake, something is wrong and it probablly won't end with success.
```
[  1%] Building CXX object CMakeFiles/RPU_CPU.dir/src/rpucuda/dense_bit_line_maker.cpp.o
[  3%] Building CXX object CMakeFiles/RPU_CPU.dir/src/rpucuda/math_util.cpp.o
[  5%] Building CXX object CMakeFiles/RPU_CPU.dir/src/rpucuda/rng.cpp.o
[  6%] Building CXX object CMakeFiles/RPU_CPU.dir/src/rpucuda/rpu.cpp.o
[  8%] Building CXX object CMakeFiles/RPU_CPU.dir/src/rpucuda/rpu_buffered_transfer_device.cpp.o
```
### 9.Install decolle and have fun
The hard part is finished.
## License

This project is licensed under the GPLv3 License - see the [LICENSE.txt](LICENSE.txt) file for details
