#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
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

from functools import reduce

COMMAND = "ls"
HOST = 'viking.york.ac.uk'

ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
                   shell=False,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
result = ssh.stdout.readlines().split('\n')
if result == []:
    error = ssh.stderr.readlines()
    print(sys.stderr, "ERROR: %s" % error)
else:
    print(result)
