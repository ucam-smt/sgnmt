Sphinx extensions::

    pip install sphinxcontrib-napoleon
    pip install sphinx\_rtd\_theme
    pip install sphinx-argparse

How to generate the docs::

    make clean
    sphinx-apidoc -f -o . .. ../cam/sgnmt/blocks/machine_translation
    make html
