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

from boutdata.collect import collect
from boututils.datafile import DataFile
from boututils.showdata import showdata
from boutdata.griddata import gridcontourf
from boututils.plotdata import plotdata
import pickle
import colorsys
from inspect import getsource as GS


def getSource(obj):
    lines = GS(obj)
    print(lines)


def funcReqs(obj):
    lines = GS(obj).partition(':')[0]
    print(lines)


def getDistinctColors(n):
    huePartition = 1.0/(n+1)
    colors = [colorsys.hsv_to_rgb(
        huePartition*value, 1.0, 1.0) for value in range(0, n)]
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
    return (np.abs(data-v)).argmin()


def find_line(filename, lookup):
    # finds line in a file
    line_num = 'blah'
    with open(filename) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                line_num = num
    if line_num == 'blah':
        sys.exit('could not find "{}" in "{}"'.format(lookup, filename))
    return line_num


def read_line(filename, lookup):
    lines = open(filename, 'r').readlines()
    line_num = find_line(filename, lookup) - 1
    tmp = lines[line_num].split(': ')[1]
    try:
        tmp = eval(tmp)
    except(NameError, SyntaxError):
        tmp = tmp.strip()
    return tmp


class pickleData:
    def __init__(self, dataDir, logFile='log.txt', dataDirName='data'):
        self.dataDir = dataDir
        self.dataDirName = dataDirName
        os.chdir(dataDir)
        self.gridFile = read_line(logFile, 'gridFile')
        self.scanParams = read_line(logFile, 'scanParams')
        self.scanNum = len(self.scanParams)
        if os.path.isdir(dataDirName) is not True:
            os.mkdir(dataDirName)
        self.pickleDir = '{}/{}'.format(dataDir, dataDirName)
        self.subDirs = []
        for i in range(self.scanNum):
            # print('{}/{}'.format(dataDir, i))
            a = next(os.walk('{}/{}'.format(dataDir, i)))[1]
            a.sort()
            self.subDirs.append(a)
        for i in range(self.scanNum):
            os.system('mkdir -p {}/{}/{}'.format(self.pickleDir, i, '1-base'))
            for j in self.subDirs[i]:
                os.system('mkdir -p {}/{}/{}'.format(self.pickleDir, i, j))

    def saveData(self, quant, subDir=[]):
        '''
        quant, subDirs: will always assume populated base directory
        '''
        if len(subDir) == 0:
            subDirs = self.subDirs
        else:
            subDirs = []
            for i in range(self.scanNum):
                subDirs.append(subDir)
        for i in range(self.scanNum):
            print('################# collecting for scanParam {}'.format(
                self.scanParams[i]))
            for j in range(-1, len(subDirs[i])):
                if j == -1:
                    title = '1-base'
                else:
                    title = subDirs[i][j]
                print('############## collecting for {}'.format(title))
                for q_id in quant:
                    os.chdir('{}/{}/{}'.format(self.pickleDir, i, title))
                    if os.path.isfile(q_id) is True:
                        print('already pickled {}'.format(q_id))
                        continue
                    if j == -1:
                        os.chdir('{}/{}'.format(self.dataDir, i))
                    else:
                        os.chdir('{}/{}/{}'.format(self.dataDir, i, title))
                    try:
                        quant2 = collect(q_id)
                    except(ValueError, KeyError):
                        print('could not collect {}'.format(q_id))
                        continue
                    os.chdir('{}/{}/{}'.format(self.pickleDir, i, title))
                    pickle_on = open('{}'.format(q_id), 'wb')
                    pickle.dump(quant2, pickle_on)
                    pickle_on.close()
                    print('pickled {}'.format(q_id))


class analyse:
    def __init__(self, outDir, dataDir='data', analysisDir='analysis',
                 logFile='log.txt'):
        self.outDir = outDir
        self.dataDir = '{}/{}'.format(outDir, dataDir)
        self.analysisDir = '{}/{}'.format(outDir, analysisDir)
        self.scanParams = read_line('{}/{}'.format(outDir, logFile),
                                    'scanParams')
        self.title = read_line('{}/{}'.format(outDir, logFile),
                               'title')
        self.scanNum = len(self.scanParams)
        if os.path.isdir(self.analysisDir) is not True:
            os.mkdir(self.analysisDir)
        self.gridData()

    def listKeys(self, simIndex=0, simType='1-base'):
        if simType == '1-base':
            os.chdir('{}/{}'.format(self.outDir, simIndex))
        else:
            os.chdir('{}/{}/{}'.format(self.outDir, simIndex, simType))
        datFile = DataFile('BOUT.dmp.0.nc')
        self.datFile = datFile
        return datFile.keys()

    def gridData(self, simIndex=0):
        os.chdir('{}/{}'.format(self.outDir, simIndex))
        self.gridFile = fnmatch.filter(next(os.walk('./'))[2],
                                       '*profile*')[0]
        grid_dat = DataFile(self.gridFile)
        self.grid_dat = grid_dat
        self.j11 = int(grid_dat["jyseps1_1"])
        self.j12 = int(grid_dat["jyseps1_2"])
        self.j21 = int(grid_dat["jyseps2_1"])
        self.j22 = int(grid_dat["jyseps2_2"])
        self.ix1 = int(grid_dat["ixseps1"])
        self.ix2 = int(grid_dat["ixseps2"])
        try:
            self.nin = int(grid_dat["ny_inner"])
        except(KeyError):
            self.nin = self.j12
        self.nx = int(grid_dat["nx"])
        self.ny = int(grid_dat["ny"])
        self.R = grid_dat['Rxy']
        self.Z = grid_dat['Zxy']
        R2 = self.R[:, self.j12:self.j22]
        self.outMid_idx = self.j12 + np.where(R2 == np.amax(R2))[1][0]

    def unPickle(self, quant, simIndex=0, simType='1-base'):
        os.chdir('{}/{}/{}'.format(self.dataDir, simIndex, simType))
        return pickle.load(open('{}'.format(quant), 'rb'))

    def collectData(self, quant, simIndex=0, simType='1-base'):
        try:
            quant2 = np.squeeze(self.unPickle(quant, simIndex, simType))
        except(FileNotFoundError):
            print('{} has not been pickled'.format(quant))
            if simType == '1-base':
                os.chdir('{}/{}'.format(self.outDir, simIndex))
            else:
                os.chdir('{}/{}/{}'.format(self.outDir, simIndex, simType))
            try:
                quant2 = np.squeeze(collect(quant))
            except(ValueError):
                print('quant in gridFile')
                self.gridFile = fnmatch.filter(next(os.walk('./'))[2],
                                               '*profile*')[0]
                grid_dat = DataFile(self.gridFile)
                quant2 = grid_dat[quant]
        return quant2

    def scanCollect(self, quant, simType='3-addC'):
        x = []
        if quant == 't_array':
            for i in range(self.scanNum):
                tempTime = (self.collectData('t_array', i, simType)/int(
                    self.collectData('Omega_ci', i, simType)))*1e6
                # print(tempTime[int(len(tempTime)/2)])
                x.append(tempTime)
        else:
            for i in range(self.scanNum):
                x.append(self.collectData(quant, i, simType))
        return x

    def showScanData(self, quant, simType, interval=5, filename='blah',
                     movie=0, fps=5, dpi=300):
        '''
        can only be used on ypi laptop with ffmpeg installed
        gotta make sure path to dataDir is the laptop one
        '''
        if filename == 'blah':
            filename = '{}-{}.mp4'.format(quant, simType)
        else:
            filename = filename
        dataDir = self.dataDir.split('/')[1:-1]  # [4:]
        newDir = ''
        for i in dataDir:
            newDir += '/' + i

        quant = self.scanCollect(quant, simType)
        interval = slice(0, -1, interval)
        titles = self.scanParams

        tme = self.scanCollect('t_array', simType)
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
            vidDir = newDir + '/analysis/vid'
            if os.path.isdir(vidDir) is not True:
                os.mkdir(vidDir)
            os.chdir(vidDir)

        # return newQuant, newTitles, newTme

        showdata(newQuant, titles=newTitles, t_array=newTme,
                 movie=movie, fps=fps, dpi=dpi, cmap='ocean')

        if movie == 1:
            os.system('mv animation.mp4 {}'.format(filename))

    def plotGridContoursX(self, yind=[], labels=[]):
        if ((len(labels) == 0) and (len(yind) == 0)):
            yind = [self.outMid_idx, -1]
            labels = ['Out Mid', 'Target']
        for i in range(self.nx):
            plt.plot(self.R[:, i], self.Z[:, i], linewidth=1,
                     color='k', alpha=0.5)
        for i, j in enumerate(yind):
            plt.plot(self.R[:, j], self.Z[:, j],
                     linewidth=4, label=labels[i])
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.grid(False)
        plt.legend(bbox_to_anchor=[1, 0.5])
        plt.axis('scaled')
        plt.tight_layout()
        plt.show()

    def plotGridContoursY(self, xind=[], labels=[]):
        if ((len(labels) == 0) and (len(xind) == 0)):
            xind = [self.ix1]
            labels = ['Seperatrix']
        for i in range(self.ny):
            plt.plot(self.R[i, :], self.Z[i, :], linewidth=1,
                     color='k', alpha=0.2)
        for i, j in enumerate(xind):
            plt.plot(self.R[j, :], self.Z[j, :],
                     linewidth=2, label=labels[i])
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.grid(False)
        plt.legend(bbox_to_anchor=[1, 0.5])
        plt.axis('scaled')
        plt.tight_layout()
        plt.show()

    def noRxScan(self, simType, quant, yind, tind=-1, norms=None,
                 qlabels=None, ylabels=None):

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

        fig, axs = plt.subplots(qlen+1, ylen, figsize=(10, 10))
        colors = getDistinctColors(len(self.scanParams))

        quants = []
        for i, j in enumerate(quant):
            tmp = self.scanCollect(j, simType)
            for k in range(len(tmp)):
                tmp[k] = norms[i]*tmp[k]
            # tmp *= norms[i]*tmp
            quants.append(tmp)

        ix1 = self.ix1

        for qNum in range(qlen):
            for yNum, y in enumerate(yind):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval('axs[qNum, yNum]')
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval('axs[yNum]')
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval('axs[qNum]')
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval('axs')
                for i, q in enumerate(quants[qNum]):
                    # print(qNum, yNum)
                    a.plot(q[tind, :, y],
                           color=colors[i],
                           label=self.scanParams[i])
                    a.axvline(ix1, color='k',
                              linestyle='--')
                    # a.set_xlim([np.amin(np.array(Ry[yNum])),
                    #             np.amax(np.array(Ry[yNum]))])
                    a.yaxis.set_major_formatter(
                        FormatStrFormatter('%g'))

        for i in range(ylen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval('axs[-1, i]')
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval('axs[i]')
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval('axs[-1]')
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval('axs')
            a.set_xlabel(ylabels[i])

        for j in range(0, qlen-1):
            for i in range(ylen):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval('axs[j, i]')
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval('axs[i]')
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval('axs[j]')
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval('axs')
                a.xaxis.set_ticklabels([])
                # axs[j, i].xaxis.set_ticklabels([])

        for i in range(qlen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval('axs[i, 0]')
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval('axs[0]')
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval('axs[i]')
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval('axs')
            a.set_ylabel(qlabels[i])
            # axs[i, 0].set_ylabel(qlabels[i])

        fig
        plt.legend(loc='upper center', ncol=2,
                   bbox_to_anchor=[0.5, 1],
                   bbox_transform=plt.gcf().transFigure,
                   # shadow=True,
                   fancybox=True,
                   title=self.title)
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.04)  # wspace=0

        # plt.suptitle(self.title)
        # plt.subplot_tool()
        plt.show()

        # os.chdir('/users/hm1234/scratch/hermes-sim-tools/test3')
        # plt.savefig(str(tind).zfill(4), bbox_inches='tight')
        # plt.close()
        # plt.cla()

    def quantXScan(self, simType, quant, yind, tind=-1, norms=None,
                   qlabels=None, ylabels=None):

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
                tmp[k] = norms[i]*tmp[k]
            # tmp *= norms[i]*tmp
            quants.append(tmp)

        ix1 = self.ix1

        for qNum in range(qlen):
            for yNum, y in enumerate(yind):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval('axs[qNum, yNum]')
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval('axs[yNum]')
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval('axs[qNum]')
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval('axs')
                for i, q in enumerate(quants[qNum]):
                    # print(qNum, yNum)
                    a.plot(Ry[yNum], q[tind, :, y],
                           color=colors[i],
                           label=self.scanParams[i])
                    a.axvline(Ry[yNum][ix1], color='k',
                              linestyle='--')
                    a.set_xlim([np.amin(np.array(Ry[yNum])),
                                np.amax(np.array(Ry[yNum]))])
                    a.yaxis.set_major_formatter(
                        FormatStrFormatter('%g'))

        for i in range(ylen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval('axs[-1, i]')
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval('axs[i]')
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval('axs[-1]')
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval('axs')
            a.set_xlabel(ylabels[i])

        for j in range(0, qlen-1):
            for i in range(ylen):
                if np.logical_and(qlen > 1, ylen > 1):
                    a = eval('axs[j, i]')
                elif np.logical_and(qlen == 1, ylen > 1):
                    a = eval('axs[i]')
                elif np.logical_and(qlen > 1, ylen == 1):
                    a = eval('axs[j]')
                elif np.logical_and(qlen == 1, ylen == 1):
                    a = eval('axs')
                a.xaxis.set_ticklabels([])
                # axs[j, i].xaxis.set_ticklabels([])

        for i in range(qlen):
            if np.logical_and(qlen > 1, ylen > 1):
                a = eval('axs[i, 0]')
            elif np.logical_and(qlen == 1, ylen > 1):
                a = eval('axs[0]')
            elif np.logical_and(qlen > 1, ylen == 1):
                a = eval('axs[i]')
            elif np.logical_and(qlen == 1, ylen == 1):
                a = eval('axs')
            a.set_ylabel(qlabels[i])
            # axs[i, 0].set_ylabel(qlabels[i])

        fig
        plt.legend(loc='upper center', ncol=2,
                   bbox_to_anchor=[0.5, 1],
                   bbox_transform=plt.gcf().transFigure,
                   # shadow=True,
                   fancybox=True,
                   title=self.title)
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.04)  # wspace=0

        # plt.suptitle(self.title)
        # plt.subplot_tool()
        plt.show()

        # os.chdir('/users/hm1234/scratch/hermes-sim-tools/test3')
        # plt.savefig(str(tind).zfill(4), bbox_inches='tight')
        # plt.close()
        # plt.cla()

    def quantYScan(self, simType, quant, xind, tind=-1, norms=None,
                   qlabels=None, xlabels=None):
        '''
        simType: typically either 1-base, 2-AddN, 3-AddC, 4-addT
        quant: list of quantities to plot
        xind: list of x-indices
        '''
        style.use('seaborn-whitegrid')
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
                tmp[k] = norms[i]*tmp[k]
            # tmp *= norms[i]*tmp
            quants.append(tmp)

        for qNum in range(qlen):
            for xNum, x in enumerate(xind):
                if np.logical_and(qlen > 1, xlen > 1):
                    a = eval('axs[qNum, xNum]')
                elif np.logical_and(qlen == 1, xlen > 1):
                    a = eval('axs[xNum]')
                elif np.logical_and(qlen > 1, xlen == 1):
                    a = eval('axs[qNum]')
                elif np.logical_and(qlen == 1, xlen == 1):
                    a = eval('axs')
                for i, q in enumerate(quants[qNum]):
                    # print(qNum, yNum)
                    # a.plot(Rx[xNum], q[tind, x, :],
                    #        color=colors[i],
                    #        label=self.scanParams[i])
                    # a.set_xlim([np.amin(np.array(Rx[xNum])),
                    #             np.amax(np.array(Rx[xNum]))])
                    # a.yaxis.set_major_formatter(
                    #     FormatStrFormatter('%g'))
                    a.plot(q[tind, x, :],
                           color=colors[i],
                           label=self.scanParams[i])
                    a.axvline(self.j12, color='k', linestyle='--')
                    a.axvline(self.j11, color='k', linestyle='--')
                    a.axvline(self.j22, color='k', linestyle='--')
                    a.yaxis.set_major_formatter(
                        FormatStrFormatter('%g'))

        for i in range(xlen):
            if np.logical_and(qlen > 1, xlen > 1):
                a = eval('axs[-1, i]')
            elif np.logical_and(qlen == 1, xlen > 1):
                a = eval('axs[i]')
            elif np.logical_and(qlen > 1, xlen == 1):
                a = eval('axs[-1]')
            elif np.logical_and(qlen == 1, xlen == 1):
                a = eval('axs')
            a.set_xlabel(xlabels[i])

        for j in range(0, qlen-1):
            for i in range(xlen):
                if np.logical_and(qlen > 1, xlen > 1):
                    a = eval('axs[j, i]')
                elif np.logical_and(qlen == 1, xlen > 1):
                    a = eval('axs[i]')
                elif np.logical_and(qlen > 1, xlen == 1):
                    a = eval('axs[j]')
                elif np.logical_and(qlen == 1, xlen == 1):
                    a = eval('axs')
                a.xaxis.set_ticklabels([])
                # axs[j, i].xaxis.set_ticklabels([])

        for i in range(qlen):
            if np.logical_and(qlen > 1, xlen > 1):
                a = eval('axs[i, 0]')
            elif np.logical_and(qlen == 1, xlen > 1):
                a = eval('axs[0]')
            elif np.logical_and(qlen > 1, xlen == 1):
                a = eval('axs[i]')
            elif np.logical_and(qlen == 1, xlen == 1):
                a = eval('axs')
            a.set_ylabel(qlabels[i])
            # axs[i, 0].set_ylabel(qlabels[i])

        fig
        plt.legend(loc='upper center', ncol=2,
                   bbox_to_anchor=[0.5, 1],
                   bbox_transform=plt.gcf().transFigure,
                   # shadow=True,
                   fancybox=True,
                   title=self.title)
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.04)  # wspace=0

        # plt.suptitle(self.title)
        # plt.subplot_tool()
        plt.show()

        # os.chdir('/users/hm1234/scratch/hermes-sim-tools/test3')
        # plt.savefig(str(tind).zfill(4), bbox_inches='tight')
        # plt.close()
        # plt.cla()

    def neConv(self, simIndex):
        os.chdir('{}/{}'.format(self.outDir, simIndex))
        tmp = next(os.walk('./'))[1]
        tmp.sort()
        ne_list = ['./']
        for i in tmp:
            ne_list.append(i)

        fig = plt.figure()
        grid = plt.GridSpec(2, len(ne_list), wspace=0.4, hspace=0.3)

        ne = []
        tme = []
        for i in ne_list:
            ne.append(np.squeeze(collect('Ne', path=i)))
            tme.append(collect('t_array', path=i))

        neAll = concatenate(ne)
        tmeAll = concatenate(tme)

        ix1 = self.ix1
        mid = int(0.5*(self.j12+self.j22))

        for i in range(len(ne_list)):
            tmp = fig.add_subplot(grid[0, i])
            tmp.plot(tme[i], ne[i][:, ix1, mid])

        tmp = fig.add_subplot(grid[1, :])
        tmp.plot(tmeAll, neAll[:, ix1, mid])
        plt.show()

    def neScanConv(self, subDir=[]):
        if len(subDir) == 0:
            subDirs = ['1-base']
            os.chdir('{}/{}'.format(self.outDir, 0))
            tmp = next(os.walk('./'))[1]
            tmp.sort()
            for i in tmp:
                subDirs.append(i)
        else:
            subDirs = subDir

        fig = plt.figure()
        grid = plt.GridSpec(2, len(subDirs))
        ix1 = self.ix1
        mid = self.outMid_idx
        # mid = -1
        # ix1 = 5

        ne = []
        tme = []
        for i in subDirs:
            ne.append(self.scanCollect('Ne', i))
            tme.append(self.scanCollect('t_array', i))

        ne_all = []
        tme_all = []
        for i in range(len(ne[0])):
            nec = []
            tmec = []
            for j in range(len(ne)):
                nec.append(ne[j][i])
                tmec.append(tme[j][i])
            ne_all.append(concatenate(nec))
            tme_all.append(concatenate(tmec))

        for i in range(len(ne)):
            tmp = fig.add_subplot(grid[0, i])
            for j in range(len(ne[0])):
                tmp.plot(tme[i][j], ne[i][j][:, ix1, mid])
            tmp.set_title(subDirs[i])

        avg_tme_cutoffs = []
        for i in range(len(tme)-1):
            a = 0
            for j in tme[i]:
                a += len(j)
            avg_tme_cutoffs.append(int(a/len(tme[0])))
        avg_tme_cutoffs = np.cumsum(avg_tme_cutoffs)

        tmp = fig.add_subplot(grid[1, :])
        for j in range(len(ne[0])):
            tmp.plot(tme_all[j], ne_all[j][:, ix1, mid],
                     label=self.scanParams[j])

        for i in range(len(tme)-1):
            tmp.axvline(tme_all[0][avg_tme_cutoffs[i]], color='k',
                        linestyle='--')
        tmp.set_xlabel(r'Time ($\mu s$)')
        tmp.set_ylabel(r'N$_{e}$ ($x10^{19} m^{-3}$)')

        fig
        plt.legend(loc='upper center', ncol=2,
                   bbox_to_anchor=[0.4, 1],
                   bbox_transform=plt.gcf().transFigure,
                   borderaxespad=0.0,
                   # shadow=True,
                   fancybox=True,
                   title=self.title)

        plt.show()

    def calcPsol(self, simIndex=0, simType='3-addC'):
        spe = self.scanCollect('Spe', simType)[simIndex][-1, :, :]
        spi = self.scanCollect('Spi', simType)[simIndex][-1, :, :]
        J = self.scanCollect('J', simType)[simIndex]
        dx = self.scanCollect('dx', simType)[simIndex]
        dy = self.scanCollect('dy', simType)[simIndex]
        dz = 2*np.pi
        Psol_e = 0
        Psol_i = 0
        for i in range(self.nx):
            for j in range(self.ny):
                Psol_e += spe[i][j]*J[i][j]*dx[i][j]*dy[i][j]*dz
                Psol_i += spi[i][j]*J[i][j]*dx[i][j]*dy[i][j]*dz
        Psol_e *= 3/2
        Psol_i *= 3/2
        return Psol_e + Psol_i

    def calc_qPar(self, simIndex=0, simType='3-addC'):
        if simType == '1-base':
            os.chdir('{}/{}'.format(self.outDir, simIndex))
        else:
            os.chdir('{}/{}/{}'.format(self.outDir, simIndex, simType))
        datFile = DataFile('BOUT.dmp.0.nc')
        Tnorm = float(datFile['Tnorm'])
        Nnorm = float(datFile['Nnorm'])
        gamma_e = 4
        gamma_i = 2.5
        mi = 3.34524e-27  # 2*Mp - deuterium
        e = 1.6e-19
        Te = self.collectData('Telim', simIndex, simType)[-1, :, -1]*Tnorm
        Ti = self.collectData('Tilim', simIndex, simType)[-1, :, -1]*Tnorm
        n = self.collectData('Ne', simIndex, simType)[-1, :, -1]*Nnorm
        Cs = np.sqrt(Te + (5/3)*Ti)*np.sqrt(e/mi)
        q_e = gamma_e * n * e * Te * Cs
        q_i = gamma_i * n * e * Ti * Cs
        return q_e + q_i

    def centreNormalZ(self, simIndex=0, simType='3-addC'):
        self.gridData(simIndex)
        RsepOMP = self.R[self.ix1, self.outMid_idx]
        RsepTar = self.R[self.ix1, -1]
        Bp = self.collectData('Bpxy', simIndex, simType)
        BpSepOMP = Bp[self.ix1, self.outMid_idx]
        BpSepTar = Bp[self.ix1, -1]
        fx = (RsepOMP * BpSepOMP) / (RsepTar * BpSepTar)
        self.fx = fx
        zSep = self.Z[self.ix1, -1]
        s = (self.Z[:, -1] - zSep)
        return s

    def eich(self, x, qbg, q0, lambda_q, S):
        scale = (q0/2)
        exp = np.exp(((S/(2*lambda_q))**2) - (x/(lambda_q*self.fx)))
        erf = erfc((S/(2*lambda_q)) - (x/(S*self.fx)))
        q = (scale * exp * erf) + qbg
        return q


def eich(x, qbg, q0, lambda_q, S):
    fx = 3.369536
    scale = (q0/2)
    exp = np.exp(((S/(2*lambda_q))**2) - (x/(lambda_q*fx)))
    erf = erfc((S/(2*lambda_q)) - (x/(S*fx)))
    q = (scale * exp * erf) + qbg
    return q


if __name__ == "__main__":
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 14}
    matplotlib.rc('font', **font)

    dateDir = '/home/hm1234/Documents/Project/remotefs/viking/'\
        'TCV/longtime/cfrac-10-06-19_175728'

    q_ids = ['t_array', 'Telim', 'Rzrad', 'J', 'dx', 'dy', 'dz',
             'Sn', 'Spe', 'Spi', 'Nn', 'Tilim', 'Pi', 'NVn', 'Vort',
             'phi', 'NVi', 'VePsi', 'Omega_ci', 'Ve', 'Pe', 'Nnorm',
             'Tnorm', 'Cs0', 'Ne', 'Qi', 'S', 'F', 'Rp', 'Pn']
    # q_ids = ['Ne']

    cScan = analyse('/users/hm1234/scratch/TCV/'
                    'longtime/cfrac-10-06-19_175728')
    rScan = analyse('/users/hm1234/scratch/TCV/'
                    'longtime/rfrac-19-06-19_102728')
    dScan = analyse('/users/hm1234/scratch/TCV2/'
                    'gridscan/grid-20-06-19_135947')
    newDScan = analyse('/users/hm1234/scratch/newTCV/'
                       'gridscan/grid-01-07-19_185351')
    newCScan = analyse('/users/hm1234/scratch/newTCV/'
                       'scans/cfrac-23-07-19_163139')
    newRScan = analyse('/users/hm1234/scratch/newTCV/'
                       'scans/rfrac-25-07-19_162302')
    # tScan = analyse('/users/hm1234/scratch/newTCV/gridscan/test')

    # x = pickleData()
    # x.saveData(q_ids)

    qlabels = ['Telim', 'Ne']

    d = newDScan
    c = newCScan
    r = newRScan

    q_par = d.calc_qPar(1, '3-addC')/1e6
    s = d.centreNormalZ(1, '3-addC')*1000
    plt.plot(s, q_par, 'ro', markersize=4)
    plt.show()

    # for k in np.arange(556):
    # newDScan.quantYScan(simType='2-addN',
    #                     quant=qlabels,
    #                     yind=[-1, 37, -10],
    #                     tind=-1)

    # newDScan.neScanConv()

    # newDScan.neConv(0)

    # newCScan.neScanConv()

    # cScan.quantYScan(simType='3-addC', quant=['Telim'], yind=[-1])
