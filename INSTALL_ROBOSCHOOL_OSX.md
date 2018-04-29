
# Install Roboschool on OSX

This are some instructions how to install Roboschool in a conda environment with Python 3 on OSX 10.11.6. 
The installation on OSX turned out to be difficult and only worked for me after patching the makefile and 
doing modifications to the source code.

## Installation

### Create conda environment

Check your current conda environments:

    conda info --envs

Delete existing 'roboschool' environment if neccessary:

    conda env remove -n roboschool

Create conda environment 'roboschool' and activate it:

    conda create -n roboschool python=3.5.2
    source activate roboschool

The environment is created with Python 3.5.2, to avoid the problems described in [ROBOSCHOOL-79](https://github.com/openai/roboschool/issues/79)

    
### Install gym

    pip install gym

If the gym installation should support Atari games install the following package:

    pip install gym[atari]


### Prepare Roboschool

    git clone https://github.com/openai/roboschool 
    cd roboschool 
    ROBOSCHOOL_PATH=`pwd` 


### Compile and install bullet

It is important to clone Bullet inside the Roboschool directory.

    git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision 
    cd bullet3 
    mkdir build 
    cd build 
    cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF .. 
    make -j4 
    make install 
    cd ../..

When these steps have been executed successfully the Bullet libraries and include files are located in $ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install.

### Install Roboschool

Install Roboschool dependencies:

    brew install cmake tinyxml assimp ffmpeg 
    brew install boost-python3 --without-python --build-from-source
    brew install qt

Set the pkg-config path:

    export PKG_CONFIG_PATH=$(dirname $(dirname $(which python)))/lib/pkgconfig
    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/opt/qt/lib/pkgconfig

Check that 'pkg-config --cflags python-3.5' shows the local python include directory. 
Check that 'pkg-config --cflags Qt5Widgets Qt5OpenGL' shows the installation paths of Qt.

Build and install Roboschool:

    pip install -e .

On my machine this failed because boost_python3 could not be found. To fix this I had to patch the lib location and name 
in the Makefile:

    ifeq ($(PYTHON),2.7)
        BOOST_PYTHON = -lboost_python
    else
        #BOOST_PYTHON = -lboost_python$(BOOST_PYTHON3_POSTFIX) 
        BOOST_PYTHON = -L/usr/local/Cellar/boost-python3/1.67.0/lib -lboost_python36
    endif

After this patch Roboschool could be build and installed.

Verify that installation is working:

    python -c "import roboschool"

Nothing spectacular happens here, it should just not raise any Python errors or segmentation faults.

Try run one of the examples:

    python agent_zoo/RoboschoolHopper_v0_2017may.py

Unfortunately this failed assertions on my installation. To fix this I had to remove (!) the assertion 'CHECK_GL_ERROR;'
from the following source files:

 * roboschool/roboschool/cpp-household/render-glwidget.cpp
 * roboschool/roboschool/cpp-household/render-simple.cpp

After that uninstall Roboschool and install it again:

    pip uninstall roboschool
    pip install -e .


## Some random notes

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
