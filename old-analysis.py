#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:16:33 2019

@author: hm1234

exec(open('/work/e281/e281/hm1234/test7/analysis/analysis7.py').read())
"""
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from boutdata.collect import collect
from boututils.datafile import DataFile
from boututils.showdata import showdata
from boutdata.griddata import gridcontourf
from boututils.plotdata import plotdata
import pickle


def FindClosest(data, v):
    return (np.abs(data-v)).argmin()


def GridDat(grid_dir, grid_file):
    os.chdir(grid_dir)
    global dat, grid_dat, j11, j12, j21, j22, ix1, ix2, nin, nx, ny, R, Z
    dat = DataFile('BOUT.dmp.0.nc')
    grid_dat = DataFile(grid_file)
    j11 = int(grid_dat["jyseps1_1"])
    j12 = int(grid_dat["jyseps1_2"])
    j21 = int(grid_dat["jyseps2_1"])
    j22 = int(grid_dat["jyseps2_2"])
    ix1 = int(grid_dat["ixseps1"])
    ix2 = int(grid_dat["ixseps2"])
    try:
        nin = int(grid_dat["ny_inner"])
    except:
        nin = j12
    nx = int(grid_dat["nx"])
    ny = int(grid_dat["ny"])
    R = grid_dat['Rxy']
    Z = grid_dat['Zxy']


def CollectTime():
    for i in vals:
        os.chdir('{}/{}/currents_on'.format(data_dir, i))
        # dat = DataFile("BOUT.dmp.0.nc")
        quant = collect('t_array')
        tme.append((quant/int(collect('Omega_ci')))*1e6)


def CollectFlux():  # [-1,:,-1,0]
    for i in vals:
        os.chdir('{}/{}/add-neutrals'.format(data_dir, i))
        # dat = DataFile("BOUT.dmp.0.nc")
        for j, q_id in enumerate(q_ids):
            quant2 = collect(q_id, tind=-1, yind=-1, zind=0)
            if j == 0:
                te.append(np.squeeze(quant2))
            elif j == 1:
                ne.append(np.squeeze(quant2))
            elif j == 2:
                rzrad.append(np.squeeze(quant2))
            print('#'*72)
            print('{} is done'.format(i))


def PickleTime2():
    os.chdir(pickle_dir)
    pickle_on = open('tme', 'wb')
    pickle.dump(tme, pickle_on)
    pickle_on.close()
    print('Pickled Time')


def PickleTime():
    for i in vals:
        os.chdir('{}/{}'.format(data_dir, i))
        # dat = DataFile("BOUT.dmp.0.nc")
        quant = collect('t_array')
        temp = (quant/int(collect('Omega_ci')))*1e6
        os.chdir(pickle_dir)
        pickle_on = open('tme-neutral_currents-{}'.format(i), 'wb')
        pickle.dump(temp, pickle_on)
        pickle_on.close()
        print('Pickled time for {}'.format(i))
    print('Pickled all Time')


def PickleData():  # [-1,:,-1,0]
    for i in vals:
        os.chdir('{}/{}'.format(data_dir, i))
        # dat = DataFile("BOUT.dmp.0.nc")
        for j, q_id in enumerate(q_ids):
            quant2 = collect(q_id)
            os.chdir(pickle_dir)
            pickle_on = open('{}-neutral_currents-{}'.format(q_id, i), 'wb')
            pickle.dump(quant2, pickle_on)
            pickle_on.close()
            print('Pickled data for {}-{}'.format(i, q_id))
            os.chdir('{}/{}'.format(data_dir, i))
    print('Pickled Data')

def PickleData2():  # [-1,:,-1,0]
    i = '1e19'
    for j, q_id in enumerate(q_ids):
        quant2 = collect(q_id)
        os.chdir(pickle_dir)
        pickle_on = open('{}-neutral_currents-{}'.format(q_id, i), 'wb')
        pickle.dump(quant2, pickle_on)
        pickle_on.close()
        os.chdir(data_dir)
        print('Pickled data for {}-{}'.format(i, q_id))
    print('Pickled Data')

# pickle_on = open('te-140', 'wb')
# pickle.dump(te, pickle_on)
# pickle_on.close()


def unpickle(quant, frac, nModel):
    if nModel == 'none':
        nModel = 'noNeutral'
    os.chdir(pickle_dir)
    tmp = pickle.load(open('{}-{}-{}'.format(quant, nModel, frac), 'rb'))
    os.chdir(current_dir)
    return tmp


def PlotFlux():
    a = (100*np.array(vals))  # - np.ones(len(vals))).astype(int)
    for j, i in enumerate(a):
        flux = ne[j]*np.sqrt(te[j])
        plt.plot(flux/max(flux), label='{}%'.format(i))

    plt.xlabel(r'Te ($eV$)')
    plt.ylabel('Normalised flux')
    plt.title('[-1,:,-1,0]')
    plt.legend()
    plt.show()


def ContourPlot(quant):
    Rmin = 0.4
    Rmax = np.max(R)
    Zmin = np.min(Z)
    Zmax = -1.2
    fig, axis = plt.subplots()
    axis.set_xlim(Rmin, Rmax)
    axis.set_ylim(Zmin, Zmax)
    c = gridcontourf(grid=grid_dat, data2d=quant, show=False, ax=axis)
    fig.colorbar(c, ax=axis)
    plt.show()


def NeConvPlot(ix, fracs):
    for i, j in enumerate(fracs):
        os.chdir('{}/{}/add-neutrals'.format(data_dir, j))
        iy = int((j12+j22)/2)
        out_mid = np.squeeze(collect('Nn', xind=ix, yind=iy, zind=0))
        iy = int((j11+j21)/2)
        in_mid = np.squeeze(collect('Nn', xind=ix, yind=iy, zind=0))
        iy = -1
        target_plate = np.squeeze(collect('Nn', xind=ix, yind=iy, zind=0))

        tme2 = tme[i]

        plt.plot(tme2, target_plate, label='target plate')
        plt.plot(tme2, out_mid, label='outboard midplane')
        plt.plot(tme2, in_mid, label='inboard midplane')
        plt.legend(loc='center right')

        plt.title('{}%'.format(j*100))
        plt.xlabel(r'Time ($\mu s$)')
        plt.ylabel(r'Ne')

        plt.show()


def NeConvPlot2(ix, fracs):
    for i, j in enumerate(fracs):
        os.chdir('{}/{}/currents_on'.format(data_dir, j))
        iy = int((j12+j22)/2)
        out_mid = np.squeeze(collect('Ne', xind=ix, yind=iy, zind=0))
        iy = int((j11+j21)/2)
        in_mid = np.squeeze(collect('Ne', xind=ix, yind=iy, zind=0))
        iy = -1
        target_plate = np.squeeze(collect('Ne', xind=ix, yind=iy, zind=0))

        CollectTime()

        tme2 = tme[i]

        plt.plot(tme2, target_plate, label='target plate')
        plt.plot(tme2, out_mid, label='outboard midplane')
        plt.plot(tme2, in_mid, label='inboard midplane')
        plt.legend(loc='upper right')

        plt.title('{}%'.format(j*100))
        plt.xlabel(r'Time ($\mu s$)')
        plt.ylabel(r'Ne')

        plt.show()


def NeConvPlot3(ix):
    print('entered NECONVPLOT3')
    os.chdir(data_dir)
    iy = int((j12+j22)/2)
    out_mid = np.squeeze(collect('Ne', xind=ix, yind=iy, zind=0))
    iy = int((j11+j21)/2)
    in_mid = np.squeeze(collect('Ne', xind=ix, yind=iy, zind=0))
    iy = -1
    target_plate = np.squeeze(collect('Ne', xind=ix, yind=iy, zind=0))
    
    tme2 = collect('t_array')/int(collect('Omega_ci'))*1e6
    
    plt.plot(tme2, target_plate, label='target plate')
    plt.plot(tme2, out_mid, label='outboard midplane')
    plt.plot(tme2, in_mid, label='inboard midplane')
    plt.legend(loc='upper right')
    
    plt.title('{}%'.format(vals[0]*100))
    plt.xlabel(r'Time ($\mu s$)')
    plt.ylabel(r'Ne')
    
    plt.show()

#        mid_sep = quant[:,ix2, int((j12+j22)/2), 0]
#        mid_back = quant[:,ix2, int((j11+j21)/2), 0]
#        lower_div = quant[:,ix2, 130, 0]
#
#        tme = collect('t_array')
#        tme = (tme/int(collect('Omega_ci')))*1e6
#
#        os.chdir('/work/e281/e281/hm1234/test7/analysis/impurity-scan/')
#
#        plt.plot(tme, lower_div, label='lower divertor')
#        plt.plot(tme, mid_sep, label='outboard midplane')
#        plt.plot(tme, mid_back, label='inboard midplane')
#        plt.legend(loc='center right')
#
#        plt.title('{}%'.format(i*100))
#        plt.xlabel(r'Time ($\mu s$)')
#        plt.ylabel(r'Temperature ($eV$)')
#
#        plt.savefig('{}-{}.png'.format(q_id, i))
#        plt.clf()


def test_error():
    global test_list, test_list_keys, other_list, other_keys
    dat_list = dat.list()
    test_list = []
    test_list_keys = []
    other_list = []
    other_keys = []
    ref_shape = collect('Ne').shape
    for i in range(len(dat_list)):
        temp_dat = collect(dat_list[i])
        if (temp_dat.shape == ref_shape):
            test_list_keys.append(dat_list[i])
            test_list.append(temp_dat)
            print(dat_list[i], temp_dat.shape)
        else:
            print(dat_list[i], temp_dat.shape)
            other_list.append(temp_dat)
            other_keys.append(dat_list[i])


def ErrorPlotLastT():
    for i in range(len(test_list_keys)):
        plt.plot(test_list[i][-1, ix2, :, 0])
        plt.title(test_list_keys[i])
        plt.show()


def test_rads(new_dir, x, y): # x = ix2, y = j22 for divertor
    os.chdir(new_dir)
    a = []
    Rzrad = collect('Rzrad')
    J = collect('J')
    dx = collect('dx')
    dy = collect('dy')
    dz = collect('dz')
    Rtot = np.squeeze(Rzrad)

    for i in range(Rtot.shape[0]):
        b = 0
        for j in range(x, Rtot.shape[1]):
            for k in range(y, Rtot.shape[2]):
                b += abs(Rtot[i][j][k]*J[j][k]*dx[j][k]*dy[j][k]*dz)
        a.append(b)

    return a

    
def rads(cfrac, ntype):
    os.chdir(pickle_dir)
    a = []

    Rzrad = unpickle('Rzrad', cfrac, ntype)
    J = unpickle('J', cfrac, ntype)
    dx = unpickle('dx', cfrac, ntype)
    dy = unpickle('dy', cfrac, ntype)
    dz = unpickle('dz', cfrac, ntype)
    Rtot = np.squeeze(Rzrad)

    for i in range(Rtot.shape[0]):
        b = 0
        for j in range(Rtot.shape[1]):
            for k in range(Rtot.shape[2]):
                b += abs(Rtot[i][j][k]*J[j][k]*dx[j][k]*dy[j][k]*dz)
        a.append(b)

    # plt.plot(a)
    # plt.show()
    os.chdir(current_dir)
    return a

def rads2(cfrac, ntype):
    os.chdir(pickle_dir)
    a = []

    Rzrad = unpickle('Rzrad', cfrac, ntype)
    J = unpickle('J', cfrac, ntype)
    dx = unpickle('dx', cfrac, ntype)
    dy = unpickle('dy', cfrac, ntype)
    dz = unpickle('dz', cfrac, ntype)
    Rtot = np.squeeze(Rzrad)

    for i in range(Rtot.shape[0]):
        b = 0
        for j in range(ix2, Rtot.shape[1]):
            for k in range(j22, Rtot.shape[2]):
                b += abs(Rtot[i][j][k]*J[j][k]*dx[j][k]*dy[j][k]*dz)
        a.append(b)

    # plt.plot(a)
    # plt.show()
    os.chdir(current_dir)
    return a

def PlotRad(ntype):
    test = []
    tmeNtype = []
    for i in [0.02, 0.04, 0.06, 0.08]:
        test.append(rads2(i, ntype))
        tmeNtype.append(unpickle('tme', i, ntype))
    for i, j in enumerate([0.02, 0.04, 0.06, 0.08]):
        plt.plot(tmeNtype[i], test[i], label=j)
    plt.xlabel('time ($\mu s$)')
    plt.ylabel('Rtot')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    pickle_dir = '/mnt/lustre/users/hm1234/analysis/21May19-results/2e19/2-add-neutrals'
    # data_dir = '/work/e281/e281/hm1234/test9/impurity-scan/noNeutral-28-03-19_123755'
    # data_dir = '/fs2/e281/e281/hm1234/test8/impurity-scan/noNeutral-27-03-19_103032'
    # data_dir = '/mnt/lustre/users/hm1234/TCV/test/cc-13-05-19_101417'
    data_dir = '/users/hm1234/scratch/TCV/grid-test/try1/2e19'

    # new_dir = '/mnt/lustre/users/hm1234/TCV/test/cc-13-05-19_101417/0.02'
    
    # vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    vals = [0.02]

    # load in some grid data
    #grid_dir = data_dir + '/' + str(vals[0])
    grid_dir = data_dir
    grid_file = 'tcv_52068_64x64_profiles_2e19.nc'

    os.chdir(grid_dir)
    current_dir = os.getcwd()

    print(1)
    GridDat(grid_dir, grid_file)
    print(2)

    q_ids = ['t_array', 'Telim', 'Ne', 'Rzrad', 'J', 'dx', 'dy', 'dz',
             'Sn', 'Spe', 'Spi', 'Nn', 'Tilim', 'Pi', 'NVn', 'Vort',
             'phi', 'NVi', 'VePsi']
    # q_ids = ['Tilim', 'Vi', 'Pi']
    # units = ['Temperature (eV)', r'Electron Density ($\times 10^{20} m^{-3}$)',
    #      r'Radiated power ($W m^{-3}$)' ]
    # q_ids = ['J', 'dx', 'dy', 'dz', ]
    te = []
    ne = []
    rzrad = []
    tme = []

    # CollectTime()
    # CollectFlux()

    # PickleTime()
    # PickleData2()

    # PlotFlux()

    NeConvPlot3(ix1)

    # rad = test_rads(new_dir, 0, 0)
    # rad2 = test_rads(new_dir, ix2, j22)

    # plt.plot(rad/max(rad), label='total')
    # plt.plot(rad2/max(rad2), label='divertor')
    # plt.legend()
    # plt.show()

    # os.chdir(new_dir)
    
    # quant = np.squeeze(collect('NVn'))*(int(collect('Nnorm'))*int(collect('Cs0')))
    # os.chdir('/work/e281/e281/hm1234/analysis/10Apr19-results/highres/nvn')
    # for i in [1,50,100,150,200,-1]:
    #     figg = gridcontourf(grid=grid_dat, data2d=quant[i], separatrix=True, show=False)
    #     plt.savefig('nvn_{}.png'.format(i), dpi=600)

    # os.chdir(current_dir)
# plt.clf()
#    showdata(quant[:,:,:,0])


# te = collect('Telim', xind=62, yind=-1,zind=0)
# ne = collect('Ne', xind=62,zind=0)
# ne = np.squeeze(ne)
# plt.plot(ne[900,:], label='900')
# plt.plot(ne[700,:], label='700')
# plt.legend()
# plt.show()


# os.chdir('/work/e281/e281/hm1234/test7/analysis/impurity-scan/pickles')
# pickle_on = open('te-140', 'wb')
# pickle.dump(te, pickle_on)
# pickle_on.close()

# pickle_on = open('ne-140', 'wb')
# pickle.dump(ne, pickle_on)
# pickle_on.close()
