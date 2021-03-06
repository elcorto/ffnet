#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.distutils.core import Extension

ext1 = Extension(name = 'ffnet.fortran._ffnet',
                 sources = ['src/fortran/ffnet.f'])

ext2 = Extension(name = 'ffnet.fortran._pikaia',
                 sources = ['src/fortran/pikaia.f'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name              = 'ffnet',
          version           = '0.7.2-elcorto',
          description       = 'Feed-forward neural network solution for python',
          long_description  = '"ffnet" is a fast and easy-to-use feed-forward \
neural network training solution for python. Many nice features are \
implemented: arbitrary network connectivity, automatic data \
normalization, very efficient training tools, support for multicore \
systems, network export to fortran code. This version was changed \
slightly to let the pikaia GA version have better monitoring \
output, see ctrl(12) in src/fortran/pikaia.f and `verbostiy` in \
the python wrapper.',
          author            = 'Marek Wojciechowski, Steve Schmerler',
          author_email      = 'mwojc@p.lodz.pl, steve.schmerler@physik.tu-freiberg.de',
          url               = 'ffnet.sourceforge.net, bitbucket.org/elcorto/ffnet',
          license           = 'GPL',
          platforms         = 'Posix',
          package_dir       = {'ffnet': 'src', 
                               'ffnet.fortran': 'src/fortran'},
          py_modules        = ['ffnet.ffnet', 
                               'ffnet.pikaia', 
                               'ffnet.tools', 
                               'ffnet._mpprop',
                               'ffnet._tests', 
                               'ffnet._version',
                               'ffnet._py2f',
                               'ffnet.fortran.__init__'],
          ext_modules       = [ext1, ext2],
          data_files        = [('ffnet/examples',       ['examples/xor.py',
                                                         'examples/ocr.py',
                                                         'examples/sin.py',
                                                         'examples/stock.py',
                                                         'examples/mptrain.py']),
                               ('ffnet/examples/data',  ['examples/data/ocr.dat',
                                                         'examples/data/black-scholes.dat']),
                               ('ffnet/fortran',        ['src/fortran/ffnet.f', 'src/fortran/pikaia.f'])]
          )
