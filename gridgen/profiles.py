# Modified Tanh profile
# Formula from R.J.Groebner Nucl. Fusion 41, 1789 (2001)

from boututils.datafile import DataFile
# from numpy import exp, arange, zeros, clip
import numpy as np
import os
import matplotlib.pyplot as plt
import colorsys


def getDistinctColors(n):
    huePartition = 1.0/(n+1)
    colors = [colorsys.hsv_to_rgb(huePartition*value, 1.0, 1.0) for value in
              range(0, n)]
    return colors


def mtanh(alpha, z):
    return ((1 + alpha*z)*np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def mtanh_profile(x, xsym, hwid, offset, pedestal, alpha):
    """
    Profile with linear slope in the core

    xsym - Symmetry location
    hwid - Width of the tanh drop
    offset - zero offset
    pedestal - Height of the tanh
    """
    b = 0.5*(offset + pedestal)
    a = pedestal - b
    z = (xsym - x) / hwid
    return a * mtanh(alpha, z) + b


def profile(filename, name, offset, pedestal, hwid=0.1, alpha=0.1):
    """
    Calculate a radial profile, and add to file
    """
    with DataFile(filename, write=True) as d:
        nx = d["nx"]
        ny = d["ny"]
        x = np.arange(nx)
        ix = d["ixseps1"]

        prof = mtanh_profile(x, ix, hwid*nx, offset, pedestal, alpha)
        prof2d = np.zeros([nx, ny])
        for y in range(ny):
            prof2d[:, y] = prof

        # Handle X-points

        # Lower inner PF region
        j11 = d["jyseps1_1"]
        if j11 >= 0:
            # Reflect around separatrix
            ix = d["ixseps1"]

            for x in range(0, ix):
                prof2d[x, 0:(j11+1)] = prof2d[np.clip(2*ix - x, 0, nx-1),
                                              0:(j11+1)]

        # Lower outer PF region
        j22 = d["jyseps2_2"]
        if j22 < ny-1:
            ix = d["ixseps1"]
            for x in range(0, ix):
                prof2d[x, (j22+1):] = prof2d[np.clip(2*ix - x, 0, nx-1),
                                             (j22+1):]

        # Upper PF region
        j21 = d["jyseps2_1"]
        j12 = d["jyseps1_2"]

        if j21 != j12:
            ix = d["ixseps2"]
            for x in range(0, ix):
                prof2d[x, (j21+1):(j12+1)] = prof2d[np.clip(2*ix - x, 0, nx-1),
                                                    (j21+1):(j12+1)]

        d.write(name, prof2d)


def createNewProfile(baseGrid, newProfile, offset, pedestal):
    os.system('cp {} {}'.format(baseGrid, newProfile))
    profile(newProfile, 'Te0', 5, 100, hwid=0.1, alpha=0.1)
    profile(newProfile, 'Ti0', 5, 100, hwid=0.1, alpha=0.1)
    profile(newProfile, 'Ne0', offset, pedestal, hwid=0.1, alpha=0.1)

    with DataFile(newProfile, write=True) as d:
        d['Ni0'] = d['Ne0']

    print('generated {}'.format(newProfile))


# filename = "tcv_63127_64x64_profiles_1e19.nc"
# os.system('cp tcv_63127_64x64.nc {}'.format(filename))

# profile(filename, "Te0", 5, 35, hwid=0.1, alpha=0.1)
# profile(filename, "Ti0", 5, 35, hwid=0.1, alpha=0.1)
# profile(filename, "Ne0", 0.01, 0.19, hwid=0.1, alpha=0.1)

# with DataFile(filename, write=True) as d:
#     d["Ni0"] = d["Ne0"]

def checkProfiles(gridFiles=[], densities=[]):
    if len(densities) < 1:
        densities = np.zeros(len(gridFiles))
    grids = []
    ne = []
    te = []
    for i in gridFiles:
        grd = DataFile(i)
        grids.append(grd)
        ne.append(grd['Ne0']*1e20)
        te.append(grd['Te0'])

    ix1 = grids[0]['ixseps1']
    j11 = grids[0]['jyseps1_1']
    j12 = grids[0]['jyseps1_2']
    j22 = grids[0]['jyseps2_2']
    j21 = grids[0]['jyseps2_1']
    mid = int((j12+j22)/2)

    colors = getDistinctColors(len(gridFiles))

    plt.figure(1)
    plt.axvline(x=ix1, color='black', linestyle='--')
    for i in range(len(ne)):
        plt.plot(ne[i][:, mid], color=colors[i], label=gridFiles[i])
        print(densities[i])
        plt.axhline(y=eval('{}e19'.format(densities[i])), color=colors[i],
                    linestyle='--')

    plt.figure(2)
    plt.axvline(x=ix1, color='black', linestyle='--')
    for i in range(len(te)):
        plt.plot(te[i][:, mid], color=colors[i], label=gridFiles[i])
        plt.axhline(y=te[i][ix1, mid], color=colors[i],
                    linestyle='--')

    plt.legend()
    plt.show()


# grid1 = DataFile('tcv_63127_64x64_profiles_1e19.nc'); ne1 = grid1['Ne0']*1e20
# grid2 = DataFile('tcv_63127_64x64_profiles_2e19.nc'); ne2 = grid2['Ne0']*1e20
# grid3 = DataFile('tcv_63127_64x64_profiles_3e19.nc'); ne3 = grid3['Ne0']*1e20
# grid4 = DataFile('tcv_63127_64x64_profiles_4e19.nc'); ne4 = grid4['Ne0']*1e20
# #grid4 = DataFile('tcv_test_grid.nc'); ne4 = grid4['Ne0']

# ix1 = grid1['ixseps1']
# j11 = grid1['jyseps1_1']
# j12 = grid1['jyseps1_2']
# j22 = grid1['jyseps2_2']
# j21 = grid1['jyseps2_1']
# mid = int((j12+j22)/2)
# # mid = -4

# plt.axvline(x=ix1, color='black', linestyle='--')
# plt.axhline(y=0.1e20, color='r', linestyle='--')
# plt.axhline(y=0.2e20, color='g', linestyle='--')
# plt.axhline(y=0.3e20, color='b', linestyle='--')
# plt.axhline(y=0.4e20, color='cyan', linestyle='--')
# plt.plot(ne1[:,mid], color='r', label='1e19')
# plt.plot(ne2[:,mid], color='g', label='2e19')
# plt.plot(ne3[:,mid], color='b', label='3e19')
# plt.plot(ne4[:,mid], color='cyan', label='4e19')
# plt.legend()
# plt.show()

if __name__ == "__main__":
    shot = 63127
    dimension = '128x64'
    baseGrid = 'tcv_{}_{}.nc'.format(shot, dimension)
    # baseGrid = 'test.nc'

    densities = [0.8, 1.2, 1.6, 2.0]
    densities = [2.5, 3.0, 3.5, 4.0]
    densities = [3.25, 3.75, 4.6, 5.2]
    densities = [1, 2, 3, 3.25, 3.5, 3.75, 4.5, 5.5]
    densities = [5.8, 6.5, 7.3, 8, 8.7, 9.3]
    densities = [6, 6.5, 7, 7.5, 8.2]
    densities = [5.8, 6.5, 7.3, 8, 8.7, 9.3]
    densities = [8.9, 9.6, 10.2, 11, 12]
    densities = [9.9, 10.5, 11, 12, 13]
    densities = [1,2,3,4,5,6,7,8,9,10]

    pedBase = 0.2
    offsets = []
    pedestals = []
    gridFiles = []
    for d in densities:
        offset = 0.02*d
        offsets.append(offset)
        pedestals.append((0.2*d)-offset)
        gridFiles.append('tcv_{}_{}_profiles_{}e19.nc'.format(
            shot, dimension, d))

    # offsets = [0.02*1.2]
    # pedestals = [(0.2*1.2)-offsets[0]]
    # gridFiles = ['test_profiles.nc']
    # densities = [1.2]

    for i in range(len(densities)):
        createNewProfile(baseGrid, gridFiles[i], offsets[i], pedestals[i])

    checkProfiles(gridFiles, densities)
