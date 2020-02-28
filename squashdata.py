 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import fnmatch
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.ticker import FormatStrFormatter

from xbout import open_boutdataset
from boutdata.squashoutput import squashoutput
import animatplot as amp
from xbout.plotting.animate import animate_poloidal, animate_pcolormesh, animate_line

from boutdata.collect import collect
from boututils.datafile import DataFile
from boututils.showdata import showdata
from boutdata.griddata import gridcontourf
from boututils.plotdata import plotdata
from boututils.boutarray import BoutArray
import colorsys
from inspect import getsource as GS
import argparse

from functools import reduce


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1)
                       if n % i == 0)))


if __name__ == "__main__":
    dir = '/users/hm1234/scratch/slabTCV/2020runs/gauss-14-02-20_153923/0/2-hyper/'
    os.chdir(dir)
    squashoutput(outputname='squashed.nc', compress=True, complevel=1)
