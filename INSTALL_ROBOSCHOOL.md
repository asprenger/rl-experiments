
# Roboschool

## Install Roboschool on OS X 10.11.6

Install Ronoschool in a conda environment with Python 3.

### Create conda environment

    conda create -n roboschool python=3.5.2
    source activate roboschool

TODO check if we need to run in Python 3.5.2, the reason was to avoid [ROBOSCHOOL-79](https://github.com/openai/roboschool/issues/79)

    
### Install gym

    brew install cmake boost boost-python sdl2 swig wget
    git clone https://github.com/openai/gym 
    cd gym 
    pip install -e .

TODO check if we need to install from source or could just use the package


### Prepare Roboschool

    git clone https://github.com/openai/roboschool 
    cd roboschool 
    ROBOSCHOOL_PATH=`pwd` 


### Compile and install bullet

    git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision 
    cd bullet3 
    mkdir build 
    cd build 
    cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF .. 
    make -j4 
    make install 
    cd ../..

### Install Roboschool

    brew install cmake tinyxml assimp ffmpeg 
    brew install boost-python --without-python --with-python3 --build-from-source 
    brew install qt

NOTE: on Xing notebook this does not work, use instead:

    brew install boost-python3 --without-python --build-from-source

Set the pkg-config path

    export PKG_CONFIG_PATH=$(dirname $(dirname $(which python)))/lib/pkgconfig

Now the pkg-config --cflags python-3.5 should show you the local python include directory. 
Also pkg-config --cflags Qt5Widgets Qt5OpenGL should show the homebrew installation paths of Qt.

TODO: add /usr/local/opt/qt/lib/pkgconfig to PKG_CONFIG_PATH, WHY???


    pip install -e .

The installation failed because boost_python3 could not be found. I had to fix the Makefile like this:

    ifeq ($(PYTHON),2.7)
        BOOST_PYTHON = -lboost_python
    else
        #BOOST_PYTHON = -lboost_python$(BOOST_PYTHON3_POSTFIX)
        BOOST_PYTHON = -L/usr/local/Cellar/boost-python3/1.67.0/lib -lboost_python36
    endif

Verify that installation is working:

    python -c "import roboschool"

Try run one of the examples:

    python agent_zoo/RoboschoolHopper_v0_2017may.py


Also I need to remove the asserts in:

 * roboschool/roboschool/cpp-household/render-glwidget.cpp
 * roboschool/roboschool/cpp-household/render-simple.cpp

## Some notes

Find out where pkg-config looks for .pc files:

    pkg-config --variable pc_path pkg-config

brew installs packages in:

    /usr/local/Cellar

brew has an 'info' command that shows where lib and .pc files for packages are located. Example:

    brew info qt
    ...
    For compilers to find this software you may need to set:
        LDFLAGS:  -L/usr/local/opt/qt/lib
        CPPFLAGS: -I/usr/local/opt/qt/include
    For pkg-config to find this software you may need to set:
        PKG_CONFIG_PATH: /usr/local/opt/qt/lib/pkgconfig


