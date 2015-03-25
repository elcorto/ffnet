"""
-------------
ffnet package
-------------
"""

from _version import version
import fortran

# elcorto 2015-03: Don't need this, eh?
##import ffnet as ffnetmodule

# elcorto 2015-03: Why?? This kind of aliasing makes people go nuts when they
# try to understand a new package. ffnet.ffnet.foo == ffnet.foo, what?? Better
# separate functions into modules such as
#   ffnet.graph.mlgraph
#   ffnet.graph.tmlgraph
#   ffnet.graph.imlgraph
#   ffnet.io.savenet
#   ffnet.io.loadnet
#   ffnet.io.exportnet
#   ffnet.io.readdata
from ffnet import ffnet, \
                  mlgraph, \
                  tmlgraph, \
                  imlgraph, \
                  savenet, \
                  loadnet, \
                  exportnet, \
                  readdata
                  
import tools
from pikaia import pikaia
import _tests
