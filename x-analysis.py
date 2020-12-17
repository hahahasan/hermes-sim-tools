#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:16:33 2019

@author: hm1234

exec(open('/work/e281/e281/hm1234/test7/analysis/analysis7.py').read())
"""
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
# from boutdata.squashoutput import squashoutput
from boutdata2.squashoutput import squashoutput

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
from inspect import getsource as GS

from functools import reduce


def factors(n):
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
        )
    )


def getSource(obj):
    lines = GS(obj)
    print(lines)


def funcReqs(obj):
    lines = GS(obj).partition(":")[0]
    print(lines)


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


class squashData:
    def __init__(self, dataDir, logFile="log.txt", dataDirName="data"):
        self.dataDir = dataDir
        os.chdir(dataDir)
        self.gridFile = read_line(logFile, "grid_file")
        self.scanParams = read_line(logFile, "scan_params")
        if self.scanParams is not None:
            self.scanNum = len(self.scanParams)
        else:
            self.scanNum = 1
            self.scanParams = ["foo"]
        self.subDirs = []
        for i in range(self.scanNum):
            # print('{}/{}'.format(dataDir, i))
            a = next(os.walk("{}/{}".format(dataDir, i)))[1]
            a.sort()
            a.insert(0, "")
            self.subDirs.append(a)

    def saveData(self, subDir=[], scanIds=[]):
        """
        quant, subDirs: will always assume populated base directory
        """
        if len(scanIds) == 0:
            scanIds = range(self.scanNum)
        if len(subDir) == 0:
            subDirs = self.subDirs
        else:
            subDirs = []
            for i in scanIds:
                subDirs.append(subDir)
        print(self.dataDir)
        for i in scanIds:
            print("################# squashing scanParam {}".format(self.scanParams[i]))
            for j in range(0, len(subDirs[i])):
                title = subDirs[i][j]
                print("############## collecting for {}".format(title))
                os.chdir("{}/{}/{}".format(self.dataDir, i, title))
                if os.path.isfile("squashed.nc") is True:
                    print("already squashed data".format())
                    continue
                try:
                    # os.system('rm BOUT.dmp.nc')
                    squashoutput(outputname="squashed.nc", compress=True, complevel=1, quiet=True, tind_auto=True)
                except (OSError, ValueError):
                    print("could not squash {}-{}".format(i, title))
                    continue
                print("squashed {}".format(title))


class analyse:
    def __init__(self, outDir, analysisDir="analysis", logFile="log.txt"):
        self.outDir = outDir
        self.analysisDir = "{}/{}".format(outDir, analysisDir)
        self.scanParams = read_line("{}/{}".format(outDir, logFile), "scanParams")
        self.title = read_line("{}/{}".format(outDir, logFile), "title")
        self.gridFile = read_line("{}/{}".format(outDir, logFile), "gridFile")
        if type(self.title) is not str:
            self.title = "grid"
        self.scanNum = len(self.scanParams)
        if os.path.isdir(self.analysisDir) is not True:
            os.mkdir(self.analysisDir)
        if self.gridFile is not None:
            self.gridData()
        self.subDirs = []
        for i in range(self.scanNum):
            # print('{}/{}'.format(dataDir, i))
            a = next(os.walk("{}/{}".format(outDir, i)))[1]
            a.sort()
            a.insert(0, "")
            self.subDirs.append(a)

    def listKeys(self, simIndex=0, simType=""):
        os.chdir("{}/{}/{}".format(self.outDir, simIndex, simType))
        datFile = DataFile("BOUT.dmp.0.nc")
        self.datFile = datFile
        return datFile.keys()

    def gridData(self, simIndex=0, simType=""):
        os.chdir("{}/{}/{}".format(self.outDir, simIndex, simType))
        self.gridFile = fnmatch.filter(next(os.walk("./"))[2], "*profile*")[0]
        grid_dat = DataFile(self.gridFile)
        if os.path.isfile("squashed.nc") is False:
            print("{}-{} not squashed. squashing now".format(simIndex, simType))
            squashoutput(
                outputname="squashed.nc", compress=True, complevel=1, quiet=True
            )
        self.datFile = DataFile("squashed.nc")
        self.grid_dat = grid_dat
        self.j11 = int(grid_dat["jyseps1_1"])
        self.j12 = int(grid_dat["jyseps1_2"])
        self.j21 = int(grid_dat["jyseps2_1"])
        self.j22 = int(grid_dat["jyseps2_2"])
        self.ix1 = int(grid_dat["ixseps1"])
        self.ix2 = int(grid_dat["ixseps2"])
        try:
            self.nin = int(grid_dat["ny_inner"])
        except (KeyError):
            self.nin = self.j12
        self.nx = int(grid_dat["nx"])
        self.ny = int(grid_dat["ny"])
        self.R = grid_dat["Rxy"]
        self.Z = grid_dat["Zxy"]
        R2 = self.R[:, self.j12 : self.j22]
        self.outMid_idx = self.j12 + np.where(R2 == np.amax(R2))[1][0]

    def gridPlot(self, quant2D, simIndex=0, simType=""):
        self.gridData(simIndex, simType)
        try:
            quant2D = self.grid_dat[quant2D]
        except (KeyError):
            quant2D = self.datFile[quant2D]
        gridcontourf(self.grid_dat, quant2D)

    def collectData(self, quant="all", simIndex=0, simType=""):
        os.chdir("{}/{}/{}".format(self.outDir, simIndex, simType))
        if os.path.isfile("squashed.nc") is False:
            print("{}-{} not squashed. squashing now".format(simIndex, simType))
            squashoutput(
                outputname="squashed.nc", compress=True, complevel=1, quiet=True
            )
        if self.gridFile is not None:
            gridFile = fnmatch.filter(next(os.walk("./"))[2], "*profile*")[0]
        else:
            gridFile = None
        print(gridFile)
        ds = open_boutdataset(
            "squashed.nc",
            gridfilepath=gridFile,
            coordinates={"x": "psi_pol", "y": "theta", "z": "zeta"},
            geometry="toroidal",
        )
        if quant == "all":
            quant2 = ds
        else:
            try:
                quant2 = ds[quant]
            except (KeyError):
                quant2 = ds["t_array"].metadata[quant]
        return np.squeeze(quant2)

    def scanCollect(self, quant, simType="3-addC", subDirs=[]):
        x = []
        if len(subDirs) == 0:
            subDirs = range(self.scanNum)
        else:
            subDirs = subDirs
        if quant == "t_array":
            for i in subDirs:
                tempTime = (
                    self.collectData("t_array", i, simType)
                    / int(self.collectData("Omega_ci", i, simType))
                ) * 1e3
                # print(tempTime[int(len(tempTime)/2)])
                x.append(tempTime)
        else:
            for i in subDirs:
                x.append(self.collectData(quant, i, simType))
        return x

    def concatenateData(self, subDirs=None, simIndex=0):
        if subDirs is None:
            subDirs = self.subDirs[simIndex]

        a = []
        for i in subDirs:
            a.append(self.collectData(simIndex=simIndex, simType=i))

        c = xr.combine_nested(a, concat_dim="t")

        for i in list(c.data_vars):
            c[i].attrs["metadata"] = a[0]["t_array"].attrs["metadata"]

        return c

    def showScanData(
        self, quant, simType, interval=5, filename="blah", movie=0, fps=5, dpi=300
    ):
        """
        can only be used on ypi laptop with ffmpeg installed
        gotta make sure path to dataDir is the laptop one
        """
        if filename == "blah":
            filename = "{}-{}.mp4".format(quant, simType)
        else:
            filename = filename
        dataDir = self.dataDir.split("/")[1:-1]  # [4:]
        newDir = ""
        for i in dataDir:
            newDir += "/" + i

        quant = self.scanCollect(quant, simType)
        interval = slice(0, -1, interval)
        titles = self.scanParams

        tme = self.scanCollect("t_array", simType)
        times = []
        for i in tme:
            times.append(i.shape[0])
        uniqTimes = list(set(times))
        uniqTimesIdx = []
        for i in uniqTimes:
            uniqTimesIdx.append(listDuplicates(times, i))
        quantIdx = max(uniqTimesIdx, key=len)

        newQuant = []
        newTitles = []
        for i in quantIdx:
            newQuant.append(quant[i][interval])
            newTitles.append(titles[i])
        newTme = tme[quantIdx[0]][interval]

        # print(newQuant[0].shape)
        # print(newTme.shape)
        # print(newTitles)

        if movie == 1:
            vidDir = newDir + "/analysis/vid"
            if os.path.isdir(vidDir) is not True:
                os.mkdir(vidDir)
            os.chdir(vidDir)

        # return newQuant, newTitles, newTme

        showdata(
            newQuant,
            titles=newTitles,
            t_array=newTme,
            movie=movie,
            fps=fps,
            dpi=dpi,
            cmap="plasma",
        )

        if movie == 1:
            os.system("mv animation.mp4 {}".format(filename))

    def plotGridContoursX(self, simIndex=0, yind=[], labels=[]):
        self.gridData(simIndex)
        if (len(labels) == 0) and (len(yind) == 0):
            yind = [self.outMid_idx, -1]
            labels = ["Out Mid", "Target"]
        for i in range(self.nx):
            plt.plot(self.R[:, i], self.Z[:, i], linewidth=1, color="k", alpha=0.5)
        for i, j in enumerate(yind):
            plt.plot(self.R[:, j], self.Z[:, j], linewidth=4, label=labels[i])
        split_idx = [self.j11, self.j12, self.j21, self.j22]
        for i, j in enumerate(split_idx):
            if i < len(split_idx) - 1:
                plt.plot(self.R[:, j], self.Z[:, j], linewidth=2, color="k")
            elif i == len(split_idx) - 1:
                plt.plot(
                    self.R[:, j], self.Z[:, j], linewidth=2, color="k", label="Splits"
                )

        plt.xlabel("R (m)")
        plt.ylabel("Z (m)")
        plt.grid(False)
        plt.legend(bbox_to_anchor=[1, 0.5])
        plt.axis("scaled")
        plt.tight_layout()
        plt.show()

    def plotGridContoursY(self, simIndex=0, xind=[], labels=[]):
        self.gridData(simIndex)
        if (len(labels) == 0) and (len(xind) == 0):
            xind = [self.ix1]
            labels = ["Seperatrix"]
        for i in range(self.ny):
            plt.plot(self.R[i, :], self.Z[i, :], linewidth=1, color="k", alpha=0.2)
        for i, j in enumerate(xind):
            plt.plot(self.R[j, :], self.Z[j, :], linewidth=2, label=labels[i])
        plt.xlabel("R (m)")
        plt.ylabel("Z (m)")
        plt.grid(False)
        plt.legend(bbox_to_anchor=[1, 0.5])
        plt.axis("scaled")
        plt.tight_layout()
        plt.show()

    def noRxScan(
        self, simType, quant, yind, tind=-1, norms=None, qlabels=None, ylabels=None
    ):

        # style.use('seaborn-whitegrid')
        qlen = len(quant)
        ylen = len(yind)

        if norms is None:
            norms = np.ones(qlen)

        if ylabels is None:
            ylabels = []
            for i in range(ylen):
                ylabels.append(yind[i])

        if qlabels is None:
            qlabels = []
            for i in range(qlen):
                qlabels.append(quant[i])

        fig, axs = plt.subplots(qlen, ylen, figsize=(10, 10))
        colors = getDistinctColors(len(self.scanParams))

        quants = []
        for i, j in enumerate(quant):
            tmp = self.scanCollect(j, simType)
            for k in range(len(tmp)):
                tmp[k] = norms[i] * tmp[k]
            # tmp *= norms[i]*tmp
            quants.append(tmp)

        ix1 = self.ix1

        for qNum in range(qlen):
            for yNum, y in enumerate(yind):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval("axs[qNum, yNum]")
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval("axs[yNum]")
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval("axs[qNum]")
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval("axs")
                for i, q in enumerate(quants[qNum]):
                    # print(qNum, yNum)
                    a.plot(q[tind, 2:-2, y], color=colors[i], label=self.scanParams[i])
                    a.axvline(ix1, color="k", linestyle="--")
                    # a.set_xlim([np.amin(np.array(Ry[yNum])),
                    #             np.amax(np.array(Ry[yNum]))])
                    a.yaxis.set_major_formatter(FormatStrFormatter("%g"))

        for i in range(ylen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval("axs[-1, i]")
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval("axs[i]")
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval("axs[-1]")
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval("axs")
            a.set_xlabel(ylabels[i])

        for j in range(0, qlen - 1):
            for i in range(ylen):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval("axs[j, i]")
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval("axs[i]")
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval("axs[j]")
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval("axs")
                a.xaxis.set_ticklabels([])
                # axs[j, i].xaxis.set_ticklabels([])

        for i in range(qlen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval("axs[i, 0]")
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval("axs[0]")
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval("axs[i]")
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval("axs")
            a.set_ylabel(qlabels[i])
            # axs[i, 0].set_ylabel(qlabels[i])

        fig
        plt.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=[0.5, 1],
            bbox_transform=plt.gcf().transFigure,
            # shadow=True,
            fancybox=True,
            title=self.title,
        )
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.04)  # wspace=0

        # plt.suptitle(self.title)
        # plt.subplot_tool()
        plt.show()

        # os.chdir('/users/hm1234/scratch/hermes-sim-tools/test3')
        # plt.savefig(str(tind).zfill(4), bbox_inches='tight')
        # plt.close()
        # plt.cla()

    def quantXScan(
        self, simType, quant, yind, tind=-1, norms=None, qlabels=None, ylabels=None
    ):

        # style.use('seaborn-whitegrid')
        qlen = len(quant)
        ylen = len(yind)

        if norms is None:
            norms = np.ones(qlen)

        if ylabels is None:
            ylabels = []
            for i in range(ylen):
                ylabels.append(yind[i])

        if qlabels is None:
            qlabels = []
            for i in range(qlen):
                qlabels.append(quant[i])

        Ry = []
        for i in yind:
            Ry.append(self.R[:, i])

        fig, axs = plt.subplots(qlen, ylen, figsize=(10, 10))
        colors = getDistinctColors(len(self.scanParams))

        quants = []
        for i, j in enumerate(quant):
            tmp = self.scanCollect(j, simType)
            for k in range(len(tmp)):
                tmp[k] = norms[i] * tmp[k]
            # tmp *= norms[i]*tmp
            quants.append(tmp)

        ix1 = self.ix1

        for qNum in range(qlen):
            for yNum, y in enumerate(yind):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval("axs[qNum, yNum]")
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval("axs[yNum]")
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval("axs[qNum]")
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval("axs")
                for i, q in enumerate(quants[qNum]):
                    # print(qNum, yNum)
                    a.plot(
                        Ry[yNum],
                        q[tind, :, y],
                        color=colors[i],
                        label=self.scanParams[i],
                    )
                    a.axvline(Ry[yNum][ix1], color="k", linestyle="--")
                    a.set_xlim(
                        [np.amin(np.array(Ry[yNum])), np.amax(np.array(Ry[yNum]))]
                    )
                    a.yaxis.set_major_formatter(FormatStrFormatter("%g"))

        for i in range(ylen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval("axs[-1, i]")
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval("axs[i]")
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval("axs[-1]")
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval("axs")
            a.set_xlabel(ylabels[i])

        for j in range(0, qlen - 1):
            for i in range(ylen):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval("axs[j, i]")
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval("axs[i]")
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval("axs[j]")
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval("axs")
                a.xaxis.set_ticklabels([])
                # axs[j, i].xaxis.set_ticklabels([])

        for i in range(qlen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval("axs[i, 0]")
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval("axs[0]")
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval("axs[i]")
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval("axs")
            a.set_ylabel(qlabels[i])
            # axs[i, 0].set_ylabel(qlabels[i])

        fig
        plt.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=[0.5, 1],
            bbox_transform=plt.gcf().transFigure,
            # shadow=True,
            fancybox=True,
            title=self.title,
        )
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.04)  # wspace=0

        # plt.suptitle(self.title)
        # plt.subplot_tool()
        plt.show()

        # os.chdir('/users/hm1234/scratch/hermes-sim-tools/test3')
        # plt.savefig(str(tind).zfill(4), bbox_inches='tight')
        # plt.close()
        # plt.cla()

    def quantYScan(
        self, simType, quant, xind, tind=-1, norms=None, qlabels=None, xlabels=None
    ):
        """
        simType: typically either 1-base, 2-AddN, 3-AddC, 4-addT
        quant: list of quantities to plot
        xind: list of x-indices
        """
        style.use("seaborn-whitegrid")
        qlen = len(quant)
        xlen = len(xind)

        if norms is None:
            norms = np.ones(qlen)

        if xlabels is None:
            xlabels = []
            for i in range(xlen):
                xlabels.append(xind[i])

        if qlabels is None:
            qlabels = []
            for i in range(qlen):
                qlabels.append(quant[i])

        Rx = []
        for i in xind:
            Rx.append(self.R[i, :])

        fig, axs = plt.subplots(qlen, xlen, figsize=(10, 10))
        colors = getDistinctColors(len(self.scanParams))

        quants = []
        for i, j in enumerate(quant):
            tmp = self.scanCollect(j, simType)
            for k in range(len(tmp)):
                tmp[k] = norms[i] * tmp[k]
            # tmp *= norms[i]*tmp
            quants.append(tmp)

        for qNum in range(qlen):
            for xNum, x in enumerate(xind):
                if np.logical_and(qlen > 1, xlen > 1):
                    a = eval("axs[qNum, xNum]")
                elif np.logical_and(qlen == 1, xlen > 1):
                    a = eval("axs[xNum]")
                elif np.logical_and(qlen > 1, xlen == 1):
                    a = eval("axs[qNum]")
                elif np.logical_and(qlen == 1, xlen == 1):
                    a = eval("axs")
                for i, q in enumerate(quants[qNum]):
                    # print(qNum, yNum)
                    # a.plot(Rx[xNum], q[tind, x, :],
                    #        color=colors[i],
                    #        label=self.scanParams[i])
                    # a.set_xlim([np.amin(np.array(Rx[xNum])),
                    #             np.amax(np.array(Rx[xNum]))])
                    # a.yaxis.set_major_formatter(
                    #     FormatStrFormatter('%g'))
                    a.plot(q[tind, x, :], color=colors[i], label=self.scanParams[i])
                    a.axvline(self.j12, color="k", linestyle="--")
                    a.axvline(self.j11, color="k", linestyle="--")
                    a.axvline(self.j22, color="k", linestyle="--")
                    a.yaxis.set_major_formatter(FormatStrFormatter("%g"))

        for i in range(xlen):
            if np.logical_and(qlen > 1, xlen > 1):
                a = eval("axs[-1, i]")
            elif np.logical_and(qlen == 1, xlen > 1):
                a = eval("axs[i]")
            elif np.logical_and(qlen > 1, xlen == 1):
                a = eval("axs[-1]")
            elif np.logical_and(qlen == 1, xlen == 1):
                a = eval("axs")
            a.set_xlabel(xlabels[i])

        for j in range(0, qlen - 1):
            for i in range(xlen):
                if np.logical_and(qlen > 1, xlen > 1):
                    a = eval("axs[j, i]")
                elif np.logical_and(qlen == 1, xlen > 1):
                    a = eval("axs[i]")
                elif np.logical_and(qlen > 1, xlen == 1):
                    a = eval("axs[j]")
                elif np.logical_and(qlen == 1, xlen == 1):
                    a = eval("axs")
                a.xaxis.set_ticklabels([])
                # axs[j, i].xaxis.set_ticklabels([])

        for i in range(qlen):
            if np.logical_and(qlen > 1, xlen > 1):
                a = eval("axs[i, 0]")
            elif np.logical_and(qlen == 1, xlen > 1):
                a = eval("axs[0]")
            elif np.logical_and(qlen > 1, xlen == 1):
                a = eval("axs[i]")
            elif np.logical_and(qlen == 1, xlen == 1):
                a = eval("axs")
            a.set_ylabel(qlabels[i])
            # axs[i, 0].set_ylabel(qlabels[i])

        fig
        plt.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=[0.5, 1],
            bbox_transform=plt.gcf().transFigure,
            # shadow=True,
            fancybox=True,
            title=self.title,
        )
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.04)  # wspace=0

        # plt.suptitle(self.title)
        # plt.subplot_tool()
        plt.show()

        # os.chdir('/users/hm1234/scratch/hermes-sim-tools/test3')
        # plt.savefig(str(tind).zfill(4), bbox_inches='tight')
        # plt.close()
        # plt.cla()

    def neConv(self, simIndex, subDir=[], split=False):
        try:
            os.chdir("{}/{}".format(self.outDir, simIndex))
            test = "data"
        except (FileNotFoundError):
            os.chdir("{}/{}".format(self.outDir, simIndex))
            test = "blah"
        if len(subDir) == 0:
            if test == "blah":
                subDirs = ["1-base"]
                tmp = next(os.walk("./"))[1]
            elif test == "data":
                subDirs = []
                tmp = next(os.walk("./"))[1]
            tmp.sort()
            for i in tmp:
                subDirs.append(i)
        else:
            subDirs = subDir

        fig = plt.figure()
        if split is True:
            grid = plt.GridSpec(2, len(subDirs))
        elif split is False:
            grid = plt.GridSpec(1, len(subDirs))
        ne = []
        tme = []
        for i in subDirs:
            ne.append(self.collectData("Ne", simIndex=simIndex, simType=i))
            tme.append(
                self.collectData("t_array", simIndex=simIndex, simType=i)
                / (95777791 / 1e6)
            )

        neAll = concatenate(ne)
        tmeAll = concatenate(tme)

        ix1 = self.ix1
        mid = int(0.5 * (self.j12 + self.j22))

        for i in range(len(subDirs)):
            tmp = fig.add_subplot(grid[0, i])
            tmp.plot(tme[i], ne[i][:, ix1, mid])
            tmp.set_title(subDirs[i])

        tmp = fig.add_subplot(grid[1, :])
        tmp.plot(tmeAll, neAll[:, ix1, mid], "ro", markersize=1)

        tme_cutoffs = []
        for i in range(len(tme)):
            tme_cutoffs.append(len(tme[i]))
        tme_cutoffs = np.cumsum(tme_cutoffs)

        for j in range(len(tme) - 1):
            tmp.axvline(tmeAll[tme_cutoffs[j]], color="k", linestyle="--")

        tmp.set_xlabel(r"Time ($m s$)")
        tmp.set_ylabel(r"N$_{e}$ ($x10^{19} m^{-3}$)")

        plt.show()

    def PItest(self, subDir=[], simIndex=[], xinds=[]):
        if len(subDir) == 0:
            subDirs = [""]
            os.chdir("{}/{}".format(self.outDir, 0))
            tmp = next(os.walk("./"))[1]
            tmp.sort()
            for i in tmp:
                subDirs.append(i)
        else:
            subDirs = subDir

        if len(simIndex) == 0:
            simIndex = list(range(self.scanNum))
        else:
            simIndex = simIndex

        fig = plt.figure()
        grid = plt.GridSpec(len(xinds), len(subDirs))

        ne = []
        tme = []
        for i in subDirs:
            tmp_ne = self.scanCollect("Ne", i, simIndex)
            y_avg_ne = []
            for j in tmp_ne:
                y_avg_ne.append(np.mean(j, axis=2))
            ne.append(y_avg_ne)
            tme.append(self.scanCollect("t_array", i, simIndex))

        ne_all = []
        tme_all = []
        for i in range(len(ne[0])):
            nec = []
            tmec = []
            for j in range(len(ne)):
                nec.append(ne[j][i])
                tmec.append(tme[j][i])
            ne_all.append(10 * concatenate(nec))
            tme_all.append(10 * concatenate(tmec))

        dx = self.collectData("dx")
        dy = self.collectData("dy")
        J = self.collectData("J")

        # sum_ne = []
        # for i in

        avg_tme_cutoffs = []
        for i in range(len(tme) - 1):
            a = 0
            for j in tme[i]:
                a += len(j)
            avg_tme_cutoffs.append(int(a / len(tme[0])))
        avg_tme_cutoffs = np.cumsum(avg_tme_cutoffs)

        for x in range(len(xinds)):
            tmp = fig.add_subplot(grid[x, :])
            for j in range(len(ne[0])):
                tmp.plot(tme_all[j], ne_all[j][:, xinds[x]], label=self.scanParams[j])
                tmp.set_ylabel("x-index: {}".format(xinds[x]))

        # if split is True:
        #     tmp = fig.add_subplot(grid[1, :])
        # elif split is False:
        #     tmp = fig.add_subplot(grid[0, :])

        # for j in range(len(ne[0])):
        #     tmp.plot(tme_all[j], ne_all[j][:, ix1, mid],
        #              label=self.scanParams[j])

        for i in range(len(tme) - 1):
            tmp.axvline(tme_all[0][avg_tme_cutoffs[i]], color="k", linestyle="--")
        tmp.set_xlabel(r"Time ($m s$) " + "x-index: {}".format(xinds[-1]))
        tmp.set_ylabel(r"N$_{e}$ ($x10^{19} m^{-3}$)")

        fig
        plt.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=[0.4, 1],
            bbox_transform=plt.gcf().transFigure,
            borderaxespad=0.0,
            # shadow=True,
            fancybox=True,
            title=self.title,
        )

        plt.show()

    def neScanConv(self, subDir=[], simIndex=[], split=False):
        if len(subDir) == 0:
            subDirs = [""]
            os.chdir("{}/{}".format(self.outDir, 0))
            tmp = next(os.walk("./"))[1]
            tmp.sort()
            for i in tmp:
                subDirs.append(i)
        else:
            subDirs = subDir

        if len(simIndex) == 0:
            simIndex = list(range(self.scanNum))
        else:
            simIndex = simIndex

        fig = plt.figure()
        if split is True:
            grid = plt.GridSpec(2, len(subDirs))
        elif split is False:
            grid = plt.GridSpec(1, len(subDirs))
        ix1 = self.ix1
        mid = self.outMid_idx
        # mid = -1
        # ix1 = 5

        ne = []
        tme = []
        for i in subDirs:
            ne.append(self.scanCollect("Ne", i, simIndex))
            tme.append(self.scanCollect("t_array", i, simIndex))

        ne_all = []
        tme_all = []
        for i in range(len(ne[0])):
            nec = []
            tmec = []
            for j in range(len(ne)):
                nec.append(ne[j][i])
                tmec.append(tme[j][i])
            ne_all.append(10 * concatenate(nec))
            tme_all.append(10 * concatenate(tmec))

        if split is True:
            for i in range(len(ne)):
                tmp = fig.add_subplot(grid[0, i])
                for j in range(len(ne[0])):
                    tmp.plot(tme[i][j], ne[i][j][:, ix1, mid])
                tmp.set_title(subDirs[i])

        avg_tme_cutoffs = []
        for i in range(len(tme) - 1):
            a = 0
            for j in tme[i]:
                a += len(j)
            avg_tme_cutoffs.append(int(a / len(tme[0])))
        avg_tme_cutoffs = np.cumsum(avg_tme_cutoffs)

        if split is True:
            tmp = fig.add_subplot(grid[1, :])
        elif split is False:
            tmp = fig.add_subplot(grid[0, :])

        for j in range(len(ne[0])):
            tmp.plot(tme_all[j], ne_all[j][:, ix1, mid], label=self.scanParams[j])

        for i in range(len(tme) - 1):
            tmp.axvline(tme_all[0][avg_tme_cutoffs[i]], color="k", linestyle="--")
        tmp.set_xlabel(r"Time ($m s$)")
        tmp.set_ylabel(r"N$_{e}$ ($x10^{19} m^{-3}$)")

        fig
        plt.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=[0.4, 1],
            bbox_transform=plt.gcf().transFigure,
            borderaxespad=0.0,
            # shadow=True,
            fancybox=True,
            title=self.title,
        )

        plt.show()

    def plotTargetFlux(self, fType="peak", subDirs=[], simType="3-addC"):
        tmp_nvi = self.scanCollect(quant="NVi", simType=simType, subDirs=subDirs)
        dx = self.scanCollect(quant="dx", simType=simType, subDirs=subDirs)
        dy = self.scanCollect(quant="dy", simType=simType, subDirs=subDirs)
        J = self.scanCollect(quant="J", simType=simType, subDirs=subDirs)
        if len(subDirs) == 0:
            subDirs = range(len(self.scanParams))
        else:
            subDirs = subDirs
        if self.title == "grid":
            nu = []
            for i in subDirs:
                nu.append(eval(self.scanParams[i].split("e")[-2][2:]))
        else:
            nu = []
            for i in subDirs:
                nu.append(self.scanParams[i])
        nvi = []
        pk_idx = []
        for i in range(len(subDirs)):
            # nvi.append(1e20 * 95777791 * 0.5*(tmp_nvi[i][-1, :, 2] +
            #                                   tmp_nvi[i][-1, :, 3]))
            nvi.append(
                1e20 * 95777791 * 0.5 * (tmp_nvi[i][-1, :, -3] + tmp_nvi[i][-1, :, -2])
            )
            pk_idx.append(np.where(nvi[-1] == np.amax(nvi[-1]))[0][0])

        if fType == "peak":
            for i in range(len(subDirs)):
                plt.scatter(nu[i], nvi[i][pk_idx[i]], s=50)
            plt.ylabel(r"Peak target flux [m$^{-2}$s$^{-1}$]")
        elif fType == "integrated":
            nvi_int = []
            for i in range(len(subDirs)):
                a = 0
                for j in range(nvi[i].shape[1]):
                    a += (
                        tmp_nvi[i][-1, j, -2]
                        * J[i][j, -2]
                        * dx[i][j, -2]
                        * dy[i][j, -2]
                    )
                nvi_int.append(a * 1e20 * 95777791)
                plt.scatter(nu[i], nvi_int[i], s=50)
            plt.ylabel(r"Integrated target flux [m$^{-2}$s$^{-1}$]")
            nvi = nvi_int
        elif fType == "t_average":
            prev_t = 300
            nvi_tAverage = []
            pk_idx = []
            for i in range(len(subDirs)):
                tmp = np.mean(tmp_nvi[i][-prev_t:, :, -3], axis=0) + np.mean(
                    tmp_nvi[i][-prev_t:, :, -2], axis=0
                )
                nvi_tAverage.append(tmp * 0.5 * 95777791e20)
                pk_idx.append(
                    np.where(nvi_tAverage[-1] == np.amax(nvi_tAverage[-1]))[0][0]
                )
                plt.scatter(nu[i], nvi_tAverage[i][pk_idx[i]], s=50)
            plt.ylabel(r"time averaged target flux [m$^{-2}$s$^{-1}$]")
            nvi = nvi_tAverage
        else:
            print('please select "integrated"/"peak"/"t_average"')
        plt.xlabel(r"Separatrix density [$\times 10^{19}$m$^{-3}$]")
        plt.show()

        return nu, nvi, pk_idx

    def plotPeakTargetFlux(self, subDirs=[], simType="3-addC"):
        tmp_nvi = self.scanCollect(quant="NVi", simType=simType, subDirs=subDirs)
        if len(subDirs) == 0:
            subDirs = range(len(self.scanParams))
        else:
            subDirs = subDirs
        if self.title == "grid":
            nu = []
            for i in subDirs:
                nu.append(eval(self.scanParams[i].split("e")[1][2:]))
        else:
            nu = []
            for i in subDirs:
                nu.append(self.scanParams[i])
        nvi = []
        pk_idx = []
        for i in range(len(subDirs)):
            # nvi.append(1e20 * 95777791 * 0.5*(tmp_nvi[i][-1, :, 2] +
            #                                   tmp_nvi[i][-1, :, 3]))
            nvi.append(
                1e20 * 95777791 * 0.5 * (tmp_nvi[i][-1, :, -3] + tmp_nvi[i][-1, :, -2])
            )
            pk_idx.append(np.where(nvi[-1] == np.amax(nvi[-1]))[0][0])

        for i in range(len(subDirs)):
            plt.scatter(nu[i], nvi[i][pk_idx[i]], s=50)
        plt.ylabel(r"Peak target flux [m$^{-2}$s$^{-1}$]")
        plt.xlabel(r"Separatrix density [$\times 10^{19}$m$^{-3}$]")
        plt.show()

        return nu, nvi, pk_idx

    def plotPeakTargetTe(self, subDirs=[], simType="3-addC"):
        tmp_te = self.scanCollect(quant="Telim", simType=simType, subDirs=subDirs)
        if len(subDirs) == 0:
            subDirs = range(len(self.scanParams))
        else:
            subDirs = subDirs
        if self.title == "grid":
            nu = []
            for i in subDirs:
                nu.append(eval(self.scanParams[i].split("e")[1][2:]))
        else:
            nu = []
            for i in subDirs:
                nu.append(self.scanParams[i])
        te = []
        pk_idx = []
        for i in range(len(subDirs)):
            te.append(100 * tmp_te[i][-1, :, -1])
            pk_idx.append(np.where(te[-1] == np.amax(te[-1]))[0][0])

        for i in range(len(subDirs)):
            plt.scatter(nu[i], te[i][pk_idx[i]], s=50)
        plt.ylabel(r"Peak target $T_{e}$ [eV]")
        plt.xlabel(r"Separatrix density [$\times 10^{19}$m$^{-3}$]")
        plt.show()

        return nu, te, pk_idx

    def xPlot(self, quant, simIndex=0, simType="3-addC", tvals=[], xvals=[], yvals=[]):
        quant = self.collectData(quant, simIndex, simType)
        if len(tvals) == 0:
            tvals = [None, None, 1]
        quant.isel(
            zeta=0, t=slice(tvals[0], tvals[1], tvals[2]),
        )

    def scanimate(
        self,
        quant,
        scanIds=[],
        simType="3-addC",
        frames=1,
        animate_over="t",
        save_as=None,
        show=False,
        fps=10,
        nrows=None,
        ncols=None,
        poloidal_plot=True,
        subplots_adjust=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        variables : list of str or BoutDataArray
            The variables to plot. For any string passed, the corresponding
            variable in this DataSet is used - then the calling DataSet must
            have only 3 dimensions. It is possible to pass BoutDataArrays to
            allow more flexible plots, e.g. with different variables being
            plotted against different axes.
        """

        if len(scanIds) == 0:
            nvars = self.scanNum
            scanIds = range(self.scanNum)
        else:
            nvars = len(scanIds)

        if nrows is None and ncols is None:
            ncols = int(np.ceil(np.sqrt(nvars)))
            nrows = int(np.ceil(nvars / ncols))
        elif nrows is None:
            nrows = int(np.ceil(nvars / ncols))
        elif ncols is None:
            ncols = int(np.ceil(nvars / nrows))
        else:
            if nrows * ncols < nvars:
                raise ValueError("Not enough rows*columns to fit all variables")

        fig, axes = plt.subplots(nrows, ncols, squeeze=False)

        scanData = self.scanCollect(quant, simType=simType, subDirs=scanIds)

        t_len = []
        for data in scanData:
            t_len.append(data.shape[0])
        min_tlen = min(t_len)
        for i, data in enumerate(scanData):
            scanData[i] = data.isel(t=slice(None, min_tlen, frames))

        if subplots_adjust is not None:
            fig.subplots_adjust(**subplots_adjust)

        blocks = []
        for i, (j, ax) in enumerate(zip(scanIds, axes.flatten())):
            # print(i, ax)
            v = scanData[i]
            print(self.scanParams[j])

            data = np.squeeze(v.bout.data)
            ndims = len(data.dims)
            # print(ax.title)

            if ndims == 2:
                blocks.append(
                    animate_line(
                        data=data,
                        ax=ax,
                        animate_over=animate_over,
                        animate=False,
                        **kwargs
                    )
                )
            elif ndims == 3:
                if poloidal_plot:
                    var_blocks = animate_poloidal(
                        data,
                        ax=ax,
                        animate_over=animate_over,
                        animate=False,
                        title=j,
                        **kwargs
                    )
                    for block in var_blocks:
                        blocks.append(block)
                else:
                    blocks.append(
                        animate_pcolormesh(
                            data=data,
                            ax=ax,
                            animate_over=animate_over,
                            animate=False,
                            **kwargs
                        )
                    )
            else:
                raise ValueError(
                    "Unsupported number of dimensions "
                    + str(ndims)
                    + ". Dims are "
                    + str(v.dims)
                )

        timeline = amp.Timeline(np.arange(v.sizes[animate_over]), fps=fps)
        anim = amp.Animation(blocks, timeline)
        anim.controls(timeline_slider_args={"text": animate_over})

        if save_as is not None:
            anim.save(save_as + ".gif", writer="imagemagick")

        if show:
            plt.show()

        return anim

    def calcPsol(self, simIndex=0, simType="3-addC"):
        spe = self.scanCollect("Spe", simType)[simIndex][-1, :, :]
        spi = self.scanCollect("Spi", simType)[simIndex][-1, :, :]
        J = self.scanCollect("J", simType)[simIndex]
        dx = self.scanCollect("dx", simType)[simIndex]
        dy = self.scanCollect("dy", simType)[simIndex]
        dz = 2 * np.pi
        Psol_e = 0
        Psol_i = 0
        for i in range(self.nx):
            for j in range(self.ny):
                Psol_e += spe[i][j] * J[i][j] * dx[i][j] * dy[i][j] * dz
                Psol_i += spi[i][j] * J[i][j] * dx[i][j] * dy[i][j] * dz
        Psol_e *= 3 / 2
        Psol_i *= 3 / 2
        return Psol_e + Psol_i

    def calc_qPar(self, simIndex=0, simType="3-addC"):
        if simType == "1-base":
            os.chdir("{}/{}".format(self.outDir, simIndex))
        else:
            os.chdir("{}/{}/{}".format(self.outDir, simIndex, simType))
        try:
            datFile = DataFile("squashed.nc")
        except (FileNotFoundError):
            print("data not squashed")
            datFile = DataFile("BOUT.dmp.0.nc")
        Tnorm = float(datFile["Tnorm"])
        Nnorm = float(datFile["Nnorm"])
        gamma_e = 4
        gamma_i = 2.5
        mi = 3.34524e-27  # 2*Mp - deuterium
        e = 1.6e-19
        Te = self.collectData("Telim", simIndex, simType)[-1, :, -1] * Tnorm
        Ti = self.collectData("Tilim", simIndex, simType)[-1, :, -1] * Tnorm
        n = self.collectData("Ne", simIndex, simType)[-1, :, -1] * Nnorm
        Cs = np.sqrt(Te + (5 / 3) * Ti) * np.sqrt(e / mi)
        q_e = gamma_e * n * e * Te * Cs
        q_i = gamma_i * n * e * Ti * Cs
        return q_e + q_i

    def centreNormalZ(self, simIndex=0, simType="3-addC"):
        self.gridData(simIndex)
        RsepOMP = self.R[self.ix1, self.outMid_idx]
        RsepTar = self.R[self.ix1, -1]
        Bp = self.collectData("Bpxy", simIndex, simType)
        BpSepOMP = Bp[self.ix1, self.outMid_idx]
        BpSepTar = Bp[self.ix1, -1]
        fx = (RsepOMP * BpSepOMP) / (RsepTar * BpSepTar)
        self.fx = fx
        zSep = self.Z[self.ix1, -1]
        s = self.Z[:, -1] - zSep
        return s

    def eich(self, x, qbg, q0, lambda_q, S):
        scale = q0 / 2
        exp = np.exp(((S / (2 * lambda_q)) ** 2) - (x / (lambda_q * self.fx)))
        erf = erfc((S / (2 * lambda_q)) - (x / (S * self.fx)))
        q = (scale * exp * erf) + qbg
        return q


def eich(x, qbg, q0, lambda_q, S):
    fx = 3.369536
    scale = q0 / 2
    exp = np.exp(((S / (2 * lambda_q)) ** 2) - (x / (lambda_q * fx)))
    erf = erfc((S / (2 * lambda_q)) - (x / (S * fx)))
    q = (scale * exp * erf) + qbg
    return q


def exp(x, q0, lambda_q):
    y = q0 * np.exp(-x / lambda_q) + 0.25
    return y


if __name__ == "__main__":
    font = {"family": "normal", "weight": "normal", "size": 14}
    matplotlib.rc("font", **font)

    dateDir = (
        "/home/hm1234/Documents/Project/remotefs/viking/"
        "TCV/longtime/cfrac-10-06-19_175728"
    )

    q_ids = [
        "t_array",
        "Telim",
        "Rzrad",
        "J",
        "dx",
        "dy",
        "dz",
        "Sn",
        "Spe",
        "Spi",
        "Nn",
        "Tilim",
        "Pi",
        "NVn",
        "Vort",
        "phi",
        "NVi",
        "VePsi",
        "Omega_ci",
        "Ve",
        "Pe",
        "Nnorm",
        "Tnorm",
        "Cs0",
        "Ne",
        "Qi",
        "S",
        "F",
        "Rp",
        "Pn",
        "PeSource",
        "PiSource",
        "NeSource",
    ]

    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/highNe-63127-28-09-20_215622")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/highNe-63161-28-09-20_215609")
    # x.saveData()

    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/slab-28-09-20_211134")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/slab-29-09-20_182246")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/DC_phi_bndry-30-09-20_200813")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/old_working-01-10-20_151503")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/old_working2-05-10-20_222448")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/old_working_s2-06-10-20_075536")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/old_working_s4-06-10-20_075559")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/nDC_old_working_s2-06-10-20_155853")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/test/nDC_old_working_s4-06-10-20_155832")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/DC_hyperpar_s2-08-10-20_171338")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/DC_hyperpar_s4-08-10-20_171318")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/mixmode_s2_mpv-11-10-20_001412")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/mixmode_s4-11-10-20_001259")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/mixmode_s4_mpv-11-10-20_001428")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/rolloverRestart/63127-14-10-20_004035")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/rolloverRestart/63161-14-10-20_004141")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/mixmode_s2-11-10-20_001325")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/mixmode2_s2_nhp-14-10-20_145023")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/fixedbuffers_s4-21-10-20_015815")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/fixedbuffers_s2-21-10-20_005335")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/fb2_s2-22-10-20_121004")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/fb3_s2-27-10-20_001324")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/working-28-10-20_113206")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/w3-03-11-20_182031")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/3D/nov/bigbox-niv-05-11-20_101508")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/nov/bb-07-11-20_080053")
    # x.saveData(subDir=["1.1-moretime"])
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/3D/nov/bigbox-iv-lessip-09-11-20_235824")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/nov/bb-08-11-20_214350")
    # x.saveData(subDir=[""])
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/nov/bb2-11-11-20_182752")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/nov/bb2_new-18-11-20_121118")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/nov/test-18-11-20_022517")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/3D/nov/bb4-25-11-20_153937")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/3D/nov/bb4-s4-25-11-20_183707")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/dec/s4-iv-lhp-phi_relax-04-12-20_120504")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/dec/bb2_s2_pd-09-12-20_233450/")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/dec/bb2_s4_pd-09-12-20_233612/")
    # x.saveData()
    x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/2D/dec/63127-ca-16-12-20_174905")
    x.saveData()

    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/oct/lr2_s2_63127-11-10-20_014249")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/oct/lr2_s2_63161-11-10-20_014220")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/oct/lr_s2_63127-09-10-20_200201")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/oct/lr_s2_63161-09-10-20_200215")
    # x.saveData()

    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.15_63127-14-10-20_200631")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.15_63161-14-10-20_200411")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.01_63127-14-10-20_195458")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.01_63161-14-10-20_195442")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.05_63127-14-10-20_195822")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.05_63161-14-10-20_200155")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.2_63127-14-10-20_201120")
    # x.saveData()
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/2D/targetGridSpacing/tpsl0.2_63161-14-10-20_201345")
    # x.saveData()
    
    # x = squashData("/marconi/home/userexternal/hmuhamme/work/archer/oct/mixmode2_s2-13-10-20_003803")
    # x.saveData()
    
    # s = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/highNe-63127-28-09-20_215622")
    # s.saveData()
    # s = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/highNe-63161-28-09-20_215609")
    # s.saveData()

    # q_ids = ['Ne']

    # cScan = analyse('/users/hm1234/scratch/TCV/'
    #                 'longtime/cfrac-10-06-19_175728')
    # rScan = analyse('/users/hm1234/scratch/TCV/'
    #                 'longtime/rfrac-19-06-19_102728')
    # # dScan = analyse('/users/hm1234/scratch/TCV2/'
    # #                 'gridscan/grid-20-06-19_135947')
    # # newDScan = analyse('/users/hm1234/scratch/newTCV/'
    # #                    'gridscan/grid-01-07-19_185351')
    # newCScan = analyse('/users/hm1234/scratch/newTCV/'
    #                    'scans/cfrac-23-07-19_163139')
    # newRScan = analyse('/users/hm1234/scratch/newTCV/'
    #                    'scans/rfrac-25-07-19_162302')
    # tScan = analyse('/users/hm1234/scratch/newTCV/gridscan/test')

    # x = pickleData('/fs2/e281/e281/hm1234/newTCV/hgridscan/grid-24-09-19_112435')
    # x.saveData(q_ids)
    # x = pickleData('/users/hm1234/scratch/newTCV/gridscan/grid-12-09-19_165234/')
    # x.saveData(q_ids)
    # x = squashData('/users/hm1234/scratch/newTCV/gridscan/grid-12-09-19_165234')
    # x.saveData

    qlabels = ["Telim", "Ne"]

    # x = squashData("/home/e281/e281/hm1234/hm1234/3D/test/slab-29-09-20_182246")
    # x.saveData()
    
    # x = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/3D/july/manmix-02-07-20_155914"
    # )

    # x = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july/63127-15-07-20_193445"
    # )
    # x.saveData()
    # x = squashData(
    #     "/marconi_work/FUA34_SOLBOUT4/hmuhamme/2D/july/63161-15-07-20_192619"
    # )
    # x.saveData()
    # x = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july/s2-63161-15-07-20_194136"
    # )
    # x.saveData()
    # x = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july/s2-63127-15-07-20_194136"
    # )
    # x.saveData()

    # j2 = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july2/s4-63127-16-07-20_232255"
    # )
    # j2.saveData()

    # j2 = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july2/s2-63127-16-07-20_232221"
    # )
    # j2.saveData()

    # j2 = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july2/s2-63161-16-07-20_232147"
    # )
    # j2.saveData()

    # j2 = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july2/adas63127-30-07-20_180810"
    # )
    # j2.saveData()

    # j2 = squashData(
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july2/adas63161-30-07-20_184953"
    # )
    # j2.saveData()

    # a = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/s4_63127-11-09-20_190916")
    # a.saveData()
    # a = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/s4_63161-12-09-20_113531")
    # a.saveData()
    # a = squashData("/marconi/home/userexternal/hmuhamme/work/3D/sep/sheath4-12-09-20_121337")
    # a.saveData()

    # a = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/newgrid-63127-21-09-20_164717")
    # a.saveData()
    # a = squashData("/marconi/home/userexternal/hmuhamme/work/2D/sep/newgrid-63161-21-09-20_164648")
    # a.saveData()
    # # d = newDScan
    # # d2 = analyse('/users/hm1234/scratch/newTCV/gridscan/grid-07-09-19_180613')
    # # d3 = analyse('/users/hm1234/scratch/newTCV/gridscan/grid-12-09-19_165234')
    # d4 = analyse('/users/hm1234/scratch/newTCV/gridscan/grid-24-09-19_112435')
    # c = newCScan
    # # r = newRScan
    # vd = analyse('/users/hm1234/scratch/newTCV/gridscan2/grid-13-09-19_153544')
    # vd2 = analyse('/users/hm1234/scratch/newTCV/gridscan2/grid-23-09-19_140426')
    # hrhd = analyse('/users/hm1234/scratch/newTCV/high_recycle/grid-25-09-19_165128')
    # hd = analyse('/mnt/lustre/users/hm1234/newTCV/high_density/grid-28-10-19_133357')

    # d5 = analyse('/users/hm1234/scratch/newTCV/gridscan/grid-07-11-19_155631')
    # # d5.saveData()
    # vd3 = analyse('/users/hm1234/scratch/newTCV/gridscan2/grid-07-11-19_154854')
    # vd3.saveData()

    # slab = analyse('/users/hm1234/scratch/slabTCV/test/slab-29-11-19_170638')
    # hdg = analyse('/users/hm1234/scratch/newTCV2/hdscan/hdg-02-12-19_172620')
    # hdg.saveData()

    # vd = squashData('/users/hm1234/scratch/newTCV/gridscan2/grid-13-09-19_153544')
    # vd2 = squashData('/users/hm1234/scratch/newTCV/gridscan2/grid-23-09-19_140426')
    # hrhd = squashData('/users/hm1234/scratch/newTCV/high_recycle/grid-25-09-19_165128')

    # vd.saveData()
    # vd2.saveData()
    # hrhd.saveData()

    # hd2 = analyse('/fs2/e281/e281/hm1234/newTCV/hgridscan/grid-24-09-19_112435')

    # q_par = d4.calc_qPar(1, '2-addN')/1e6
    # s = d4.centreNormalZ(1, '2-addN')*1000

    # # for some reason
    # s = s.astype('float64')
    # q_par = q_par.astype('float64')

    # # for k in np.arange(556):
    # # newDScan.quantYScan(simType='2-addN',
    # #                     quant=qlabels,
    # #                     yind=[-1, 37, -10],
    # #                     tind=-1)

    # # newDScan.neScanConv()

    # # newDScan.neConv(0)

    # # newCScan.neScanConv()

    # # cScan.quantYScan(simType='3-addC', quant=['Telim'], yind=[-1])

    # plt.plot(s, q_par, 'ro', markersize=4)
    # i = 0  # 34
    # j = 34  # 63
    # plt.plot(s[i], q_par[i], 'bo', markersize=4)
    # plt.plot(s[j], q_par[j], 'bo', markersize=4)
    # x = s[i:j]
    # y = q_par[i:j]
    # print('8888888888888888')
    # popt, pcov = curve_fit(d4.eich, x, y, maxfev=999999)
    # print('################')
    # plt.plot(x, eich(x, *popt), 'g-')
    # # popt, pcov = curve_fit(eich, x, y, p0=[2, 12, 1])
    # # plt.plot(x, eich(x, *popt), 'g-')
    # print(popt)
    # plt.show()

    # s_orig = s
    # q_orig = q_par

    # s = BoutArray([-62.41107, -60.263752, -58.12967, -56.00941,
    #                -53.90376, -51.813484, -49.73972, -47.68342,
    #                -45.645176, -43.625893, -41.62586, -39.645134,
    #                -37.683605, -35.74103, -33.816814, -31.91042,
    #                -30.021072, -28.148056, -26.290417, -24.44762,
    #                -22.618889, -20.80363, -19.001366, -17.211676,
    #                -15.434324, -13.669193, -11.916161, -10.175467,
    #                -8.447111, -6.731391, -5.028546, -3.3388734,
    #                -1.6625524, 0., 1.648426, 3.2827258,
    #                4.9026012, 6.508112, 8.099079, 9.675562,
    #                11.23768, 12.785494, 14.319181, 15.838921,
    #                17.34501, 18.837511, 20.31696, 21.783352,
    #                23.23711, 24.67841, 26.10755, 27.52465,
    #                28.92989, 30.323565, 31.705738, 33.076584,
    #                34.436165, 35.784603, 37.121952, 38.448273,
    #                39.76363, 41.068077, 42.361618, 43.64431])

    # q = BoutArray([1.65182085e-05, 1.35209137e-05, 1.35209137e-05, 1.65182085e-05,
    #                2.27037561e-05, 3.29299036e-05, 4.75111660e-05, 6.60582256e-05,
    #                9.37651236e-05, 1.33889566e-04, 1.92315972e-04, 2.77720388e-04,
    #                4.02801157e-04, 5.86084015e-04, 8.54342127e-04, 1.24662268e-03,
    #                1.82020695e-03, 2.65964844e-03, 3.88941465e-03, 5.69083765e-03,
    #                8.31951781e-03, 1.21071174e-02, 1.74353035e-02, 2.50544252e-02,
    #                3.61864442e-02, 5.27957207e-02, 7.80601478e-02, 1.17530582e-01,
    #                1.80962745e-01, 2.86309413e-01, 4.66792017e-01, 7.80775187e-01,
    #                1.30828433e+00, 1.96015507e+00, 1.93110608e+00, 1.68087375e+00,
    #                1.43046853e+00, 1.22325976e+00, 1.05978494e+00, 9.32189283e-01,
    #                8.33973790e-01, 7.57209533e-01, 6.95491924e-01, 6.43991803e-01,
    #                5.99133592e-01, 5.58512003e-01, 5.20618811e-01, 4.84938514e-01,
    #                4.51486103e-01, 4.20480660e-01, 3.92125286e-01, 3.66543353e-01,
    #                3.43831424e-01, 3.23985859e-01, 3.06967540e-01, 2.92689895e-01,
    #                2.80988723e-01, 2.71660289e-01, 2.64462261e-01, 2.59194711e-01,
    #                2.55764583e-01, 2.54243239e-01, 2.54243239e-01, 2.55765543e-01])

    # x = BoutArray([3.2827258,  4.9026012,  6.508112,  8.099079,  9.675562,
    #                11.23768, 12.785494, 14.319181, 15.838921, 17.34501,
    #                18.837511, 20.31696, 21.783352, 23.23711, 24.67841,
    #                26.10755, 27.52465, 28.92989, 30.323565, 31.705738,
    #                33.076584, 34.436165, 35.784603, 37.121952, 38.448273])

    # y = BoutArray([1.68087375, 1.43046853, 1.22325976, 1.05978494, 0.93218928,
    #                0.83397379, 0.75720953, 0.69549192, 0.6439918, 0.59913359,
    #                0.558512, 0.52061881, 0.48493851, 0.4514861, 0.42048066,
    #                0.39212529, 0.36654335, 0.34383142, 0.32398586, 0.30696754,
    #                0.2926899, 0.28098872, 0.27166029, 0.26446226, 0.25919471])

    # s = s_orig.astype('float64')
    # q = q_orig.astype('float64')

    # s = s.astype('float64')

    # plt.plot(s, q, 'ro', markersize=4)
    # # s = s_orig
    # # q = q_orig
    # i = 34
    # j = -1
    # x = s[i:]
    # y = q[i:]
    # plt.plot(x[0], y[0], 'bo', markersize=6)
    # # plt.plot(x[-1], y[-1], 'bo', markersize=4)
    # plt.axvline(0, linestyle='--', color='k')
    # popt2, pcov2 = curve_fit(eich, x, y)
    # plt.plot(x, eich(x, *popt2), 'b-', linewidth=2,
    #          label='S = {:.4f}'.format(popt2[-1]))
    # x = s[:i]
    # y = q[:i]
    # popt3, pcov3 = curve_fit(eich, x, y)
    # plt.plot(x, eich(x, *popt3), 'r-', linewidth=2,
    #          label='S = {:.4f}'.format(popt3[-1]))

    # # Svals = [0, 0.5, 1, 2, 4, 10]
    # # colors = getDistinctColors(len(Svals))
    # # for i, j in enumerate(Svals):
    # #     plt.plot(s, eich(s, 0.23741361, 1.97848096, 2.89837075, j),
    # #              color=colors[i], label=f'S = {j}')

    # plt.xlabel(r'$s-s_{0}$ [mm]')
    # plt.ylabel(r'$q_{\parallel}$ [MWm$^{-2}$]')
    # plt.axhspan(q[0], q[-1], color='green', alpha=0.3)
    # print('popt2 is:', popt2)
    # print('popt3 is:', popt3)
    # # plt.legend(bbox_to_anchor=[0.5, 1],
    # #            bbox_transform=plt.gcf().transFigure)
    # plt.legend()
    # plt.show()

# sn = vd2.scanCollect('Sn', simType='2-addN')
# spe = vd2.scanCollect('Spe', simType='2-addN')
# spi = vd2.scanCollect('Spi', simType='2-addN')
# J = vd2.scanCollect('J', simType='2-addN')
# dx = vd2.scanCollect('dx', simType='2-addN')
# dy = vd2.scanCollect('dy', simType='2-addN')

# P_sn = []
# for k in range(5):
#     x = 0
#     for i in range(64):
#         for j in range(64):
#             x += sn[k].values[-1, i, j]*J[k].values[i,j]*dx[k].values[i,j]*dy[k].values[i,j]
#     print(x)
#     P_sn.append(x)

# P_spi = []
# for k in range(5):
#     x = 0
#     for i in range(64):
#         for j in range(64):
#             x += spi[k].values[-1, i, j]*J[k].values[i,j]*dx[k].values[i,j]*dy[k].values[i,j]
#     print(x)
#     P_spi.append(x)

# P_spe = []
# for k in range(5):
#     x = 0
#     for i in range(64):
#         for j in range(64):
#             x += spe[k].values[-1, i, j]*J[k].values[i,j]*dx[k].values[i,j]*dy[k].values[i,j]
#     print(x)
#     P_spe.append(x)

# densities = [6, 6.5, 7, 7.5, 8.2]

# P_sn = [437.8769478077677,
#         452.62636585742615,
#         375.2648321441369,
#         481.5114781772363,
#         684.2161422505203]

# P_spi = [2543.8336685344702,
#          2882.750766461647,
#          3223.5583696973854,
#          3586.859441228167,
#          4130.976394727855]

# P_spe = [11577.611491243135,
#          12822.99044906976,
#          14144.60154753619,
#          15453.354321093278,
#          17251.87049585934]

# P_tot = [14559.322107585373,
#          16158.367581388833,
#          17743.42474937771,
#          19521.725240498683,
#          22067.063032837716]
