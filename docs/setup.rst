
Installation
=============

Installing dependencies
------------------------

SGNMT depends on the following libraries:

* `Blocks <http://blocks.readthedocs.io/en/latest/>`_ for neural machine translation support (>=0.1)
* `OpenFST <http://openfst.org/>`_ for reading and writing FSTs (e.g. translation lattices) (>=1.5.2)
* `srilm-swig <https://github.com/desilinguist/swig-srilm>`_ for reading ARPA language model files
* `NPLM <http://nlg.isi.edu/software/nplm/>`_ for using feed-forward neural language models (>=0.3)

SGNMT does not work without blocks, but the other dependencies can be bypassed by
commenting out the predictor imports in ``cam.sgnmt.blocks.decode.py``.

Installing Blocks
**********************

Follow the `instructions <http://blocks.readthedocs.io/en/latest/setup.html>`_ in 
the blocks documentation to install the blocks framework and all its dependencies.

.. note:: 

      It might be necessary to install the HDF5 development files before installing blocks, e.g. with::

         $ sudo apt-get install libhdf5-dev 

Installing OpenFST
**********************

We recommend to install the most recent `OpenFST version <http://openfst.org/twiki/bin/view/FST/FstDownload>`_. Make
sure to enable the Python support when compiling OpenFST::

    $ ./configure --enable-far --enable-python
    $ make
    $ make install

SGNMT requires OpenFST 1.5.2 because it uses the extended Python support added in this version. For more information
see the documentation for the `OpenFST Python extension <http://www.openfst.org/twiki/bin/view/FST/PythonExtension>`_.

Installing SRILM
************************

First, install the version 1.7.1 of the `SRI language model toolkit <http://www.speech.sri.com/projects/srilm/>`_ if 
you don't already have an installation. 

.. note:: 

        According to the `documentation <https://github.com/desilinguist/swig-srilm>`_, swig-srilm requires that SRILM 
        is compiled as position independent code using ``MAKE_PIC=yes``::

          $ make World MAKE_PIC=yes

Then, checkout the srilm-swig project::

    $ git clone https://github.com/desilinguist/swig-srilm.git

Modify the Makefile as explained in the `installation instructions <https://github.com/desilinguist/swig-srilm>`_. For Ubuntu, the head of the make file should look like this::

    SRILM_LIBS=/path/to/srilm/lib/i686-m64
    SRILM_INC=/path/to/srilm/include
    PYTHON_INC=/usr/include/python2.7

Build the Python module::

    $ make python


Installing NPLM
************************

Download NPLM from the `project homepage <http://nlg.isi.edu/software/nplm/>`_ and install it. You can also
use the `UCAM NPLM fork <https://github.com/ucam-smt/nplm>`_ from Gonzalo Iglesias for threadsafety and efficiency.

Setting up SGNMT
------------------

Update your environment variables to reflect the locations of OpenFST, SRILM, and NPLM. On Ubuntu, it might be necessary to
add */usr/local/lib* to your ``LD_LIBRARY_PATH`` (default location for OpenFST)::

    $ export LD_LIBRARY_PATH=/usr/local/lib/:/path/to/swig-srilm/:/path/to/nplm/src/python:$LD_LIBRARY_PATH
    $ export PYTHONPATH=/path/to/swig-srilm/:/path/to/nplm/python/:$PYTHONPATH

Clone the GIT repository and try to start ``decode.py`` and ``train.py``::

    $ git clone https://github.com/ucam-smt/sgnmt.git
    $ cd sgnmt
    $ python train.py --help
    $ python decode.py --help

If you see the help texts for both commands, you are ready for the :ref:`tutorial-label`. If both commands fail, there is probably
something wrong with your blocks installation. If only ``decode.py`` fails, double-check that OpenFST, SRILM, and NPLM are
properly installed as described above, and that they are added to your ``PYTHONPATH``.
