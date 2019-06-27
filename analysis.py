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

# def HSVToRGB(h, s, v):
#     (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
#     # return (int(255*r), int(255*g), int(255*b))
#     return (r, g, b)


# def getDistinctColors(n):
#     huePartition = 1.0 / (n + 1)
#     return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in
#             range(0, n)]

def getDistinctColors(n):
    huePartition = 1.0/(n+1)
    colors = [colorsys.hsv_to_rgb(huePartition*value, 1.0, 1.0) for value in
              range(0, n)]
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
            os.system('mkdir -p {}/{}/{}'.format(dataDirName, i, '1-base'))
            for j in self.subDirs[i]:
                os.system('mkdir -p {}/{}/{}'.format(dataDirName, i, j))

    def saveData(self, quant):
        for i in range(self.scanNum):
            print('################# collecting for scanParam {}'.format(
                self.scanParams[i]))
            for j in range(-1, len(self.subDirs[i])):
                if j == -1:
                    title = '1-base'
                else:
                    title = self.subDirs[i][j]
                print('############## collecting for {}'.format(title))
                for q_id in quant:
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

    def unPickle(self, quant, simIndex=0, simType='1-base'):
        os.chdir('{}/{}/{}'.format(self.dataDir, simIndex, simType))
        return pickle.load(open('{}'.format(quant), 'rb'))

    def collectData(self, quant, simIndex=0, simType='1-base'):
        try:
            quant2 = self.unPickle(quant, simIndex, simType)
        except(FileNotFoundError):
            print('{} has not been pickled'.format(quant))
            os.chdir('{}/{}/{}'.format(self.outDir, simIndex, simType))
            quant2 = collect(quant)
        return quant2

    def scanCollect(self, simType, quant):
        x = []
        if quant == 't_array':
            for i in range(self.scanNum):
                tempTime = (self.unPickle('t_array', i, simType)/int(
                    self.unPickle('Omega_ci', i, simType)))*1e6
                # print(tempTime[int(len(tempTime)/2)])
                x.append(tempTime)
        else:
            for i in range(self.scanNum):
                x.append(np.squeeze(self.unPickle(quant, i, simType)))
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

        quant = self.scanCollect(simType, quant)
        interval = slice(0, -1, interval)
        titles = self.scanParams

        tme = self.scanCollect(simType, 't_array')
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
                 movie=movie, fps=fps, dpi=dpi)

        if movie == 1:
            os.system('mv animation.mp4 {}'.format(filename))

    def densityScan(self, simType, quant, yind, tind=-1, norms=None,
                    qlabels=None, ylabels=None,):

        # style.use('seaborn-whitegrid')
        qlen = len(quant)
        ylen = len(yind)

        if norms is None:
            norms = np.ones(qlen)

        if ylabels is None:
            ylabels = []
            for i in range(ylen):
                ylabels.append('blah')

        if qlabels is None:
            qlabels = []
            for i in range(qlen):
                qlabels.append('blah')

        Ry = []
        for i in yind:
            Ry.append(self.R[:, i])

        fig, axs = plt.subplots(qlen, ylen)
        colors = getDistinctColors(len(self.scanParams))

        quants = []
        for i, j in enumerate(quant):
            tmp = self.scanCollect(simType, j)
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


if __name__ == "__main__":
    dateDir = '/home/hm1234/Documents/Project/remotefs/viking/'\
        'TCV/longtime/cfrac-10-06-19_175728'

    q_ids = ['t_array', 'Telim', 'Rzrad', 'J', 'dx', 'dy', 'dz',
             'Sn', 'Spe', 'Spi', 'Nn', 'Tilim', 'Pi', 'NVn', 'Vort',
             'phi', 'NVi', 'VePsi', 'Omega_ci', 'Ve', 'Pe', 'Nnorm',
             'Tnorm', 'Cs0']  # 'Ne'
    # q_ids = ['Nnorm', 'Tnorm', 'Cs0']

    cScan = analyse('/users/hm1234/scratch/TCV/longtime/cfrac-10-06-19_175728')
    rScan = analyse('/users/hm1234/scratch/TCV/longtime/rfrac-19-06-19_102728')
    dScan = analyse('/users/hm1234/scratch/TCV2/gridscan/grid-20-06-19_135947')

    # x = pickleData(dateDir, dataDirName='data')
    # x.saveData(q_ids)

    qlabels = ['Telim', 'Tilim', 'NVn', 'Nn']
    qlabels = ['Tilim']
    dScan.densityScan(simType='3-addC',
                      quant=qlabels,
                      yind=[-1],
                      qlabels=qlabels,)
                      # norms=[100, 1e20, 100*1e20*1.6e-19],
                      # ylabels=[37, -1, -10])
