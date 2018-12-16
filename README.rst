NEAT NEURON extension
=====================

Introduction
------------

This is an extension of the NEAT toolbox (https://github.com/WillemWybo/NEAT) that allows for the simulation of neuron models while using the typical NEAT interface.

Installation
------------

Note: The following instructions are for Linux and Max OSX systems and only use command line tools. Please follow the appropriate manuals for Windows systems or tools with graphical interfaces.

Check out the git repository and install using :code:`setup.py`
::
    git clone https://github.com/WillemWybo/NEAT
    cd NEAT
::

Then go to the :code:`install.sh` script and change the line

::
    python setup.py install --prefix=~/.local
::

to your preferred python installation. Then change the following line accordingly

::
    cd ~/.local/lib/python2.7/site-packages/neatneuron/
::

To test the installation (requires :code:`pytest`)
::
    sh run_tests.sh
::
