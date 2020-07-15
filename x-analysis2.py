#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style

from xbout import open_boutdataset
from boutdata.squashoutput import squashoutput
import animatplot as amp
from xbout.plotting.animate import animate_poloidal, animate_pcolormesh, animate_line
import xarray as xr

from boutdata.collect import collect
from boututils.datafile import DataFile
from boututils.showdata import showdata
from boutdata.griddata import gridcontourf
from boututils.plotdata import plotdata
from boututils.boutarray import BoutArray
import colorsys
from functools import reduce
from functools import wraps


def factors(n):
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
        )
    )


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    colors = [
        colorsys.hsv_to_rgb(huePartition * value, 1.0, 1.0) for value in range(0, n)
    ]
    return colors


def listDuplicates(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def concatenate(data, axis=0):
    lst = []
    for i in data:
        lst.append(i)
    lst = tuple(lst)
    return np.concatenate(lst, axis=axis)


def find_closest(data, v):
    return (np.abs(data - v)).argmin()


def find_line(filename, lookup):
    # finds line in a file
    line_num = "blah"
    with open(filename) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                line_num = num
    if line_num == "blah":
        sys.exit('could not find "{}" in "{}"'.format(lookup, filename))
    return line_num


def read_line(filename, lookup):
    lines = open(filename, "r").readlines()
    line_num = find_line(filename, lookup) - 1
    tmp = lines[line_num].split(": ")[1]
    try:
        tmp = eval(tmp)
    except (NameError, SyntaxError):
        tmp = tmp.strip()
    return tmp


class SimData:
    def read_log_file(read_line):
        wraps(read_line)

        def read_log(self, *args, **kwargs):
            value = read_line(self.log_file, *args, **kwargs)
            return value

    def __init__(self, data_dir, log_file="log.txt"):
        self.date_dir = data_dir
        os.chdir(data_dir)
        self.grid_file = read_line(log_file, "grid_file")
        self.scan_params = read_line(log_file, "scan_params")
        self.title = read_line(log_file,)
        self.scan_num = len(self.scan_num)
        self.sub_dirs = []
        for i in range(self.scan_num):
            a = next(os.walk("{}/{}".format(data_dir, i)))[1]
            a.sort()
            a.insert(0, "")
            self.sub_dirs.append(a)
