SOM(Self Organizing Maps)
=====
This repository is about the som algorithm, which is implemented by pure python and only opencv-depended c++.

Introduction
----
The principal goal of an SOM is to transform an incoming signal pattern of arbitrary dimension into a one or two dimensional discrete map, 
and to perform this transformation adaptively in a topologically ordered fashion.

![Visualization of RGB color space by SOM algorithm](https://github.com/fanser/SOM/raw/master/cpp/bin/som_c.jpg)
  
Visualization of RGB color space by SOM algorithm

<div align=left>
How to use and build
----
I only implement the 2D-SOM algorithm. Because the 1D-SOM can be easlily transformed by setting the height of 2D map to 1. 

#### For python
The python codes are stored in `./python` directory.
> cd ./python <br>
from som.som_2d import SOM2D

The example code is shown in file `./python/som/som_2d.py`

#### For C++
The C++ project is in `./cpp` directory, and built by `cmake` and `make` commands. You can build it like following:
> cd cpp <br>
mkdir build && cd build <br>
cmake .. <br>
make <br>

The usage can be found in example code `./cpp/src/train_som.cc` <br>
* Note: You MUST make sure the opencv library has been installed, and add the opencv dependencies to `./cpp/CMakeLists.txt` file (`include_directories` and `link_directories`)
<br>

## Reference
1. [Self Organizing Maps: Fundamentals](http://www.cs.bham.ac.uk/~jxb/NN/l16.pdf) : A very easily understanding introduction.
2. [Kohonen's Self Organizing Feature Maps](http://www.ai-junkie.com/ann/som/som1.html): A good implementation which tells you how to set some super-parameters.
