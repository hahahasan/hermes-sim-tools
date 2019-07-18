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
from inspect import getsource as GS


def getSource(obj):
    lines = GS(obj)
    print(lines)


def funcReqs(obj):
    lines = GS(obj).partition(':')[0]
    print(lines)


class dirAnalysis:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.pickle_dir = '{}/pickles'.format(out_dir)
        self.fig_dir = '{}/figures'.format(out_dir)
        os.chdir(out_dir)
        os.system('mkdir -p {}'.format(self.pickle_dir))
        os.system('mkdir -p {}'.format(self.fig_dir))
        self.dat = DataFile('BOUT.dmp.0.nc')

    def pickleAll(self):
        for q_id in self.dat.list():
            os.chdir(self.pickle_dir)
            if os.path.isfile(q_id) is True:
                continue
            os.chdir(self.out_dir)
            quant = collect(q_id)
            os.chdir(self.pickle_dir)
            pickle_on = open('{}'.format(q_id), 'wb')
            pickle.dump(quant, pickle_on, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_on.close()
            print('########## pickled {}'.format(q_id))

    def unpickle(self, quant):
        os.chdir(self.pickle_dir)
        return pickle.load(open('{}'.format(quant), 'rb'))

    def plotFigs(self, datList=[], tind=-1, zind=0, output=None):
        if len(datList) == 0:
            datList = self.dat.list()
        for i in datList:
            if output is not None:
                output = '{}-{}'.format(i, output)
            try:
                quant = self.unpickle(i)
                plotdata(quant[tind, 2:-2, :, zind],
                         title='{} at t={}, z={}'.format(i, tind, zind),
                         output=output)
                plt.clf()
                plt.cla()
            except(IndexError):
                print('dimension of {} not correct'.format(i))
                continue

    def plotDiffs(self):
        for i in x.dat.list():
            if '(' in i:
                i2 = '{}_{}'.format(i.partition('(')[0],
                                    i.partition('(')[-1][: -1])
            else:
                i2 = i
            quant = x.unpickle(i)
            for j in range(64):
                print('########################### doing z={}'.format(j))
                os.system('mkdir -p z-idx/{}'.format(j))
                try:
                    quant_j = quant[:, 2:-2, :, j]
                    quant_mean = np.mean(quant[:, 2:-2, :, :], axis=3)
                    diff = abs(quant_j - quant_mean)
                    os.chdir('z-idx/{}'.format(j))
                    plotdata(diff[-1, :, :], title='{}-{}'.format(i, j),
                             output='{}-{}'.format(i2, str(j).zfill(2)))
                    plt.cla()
                    plt.clf()
                    os.chdir('../')
                    print('done {}'.format(i))
                except(IndexError):
                    print('{} does not have correct dimensions'.format(i))
                    continue

    def redistFiles(self):
        for i in datList:
            if '(' in i:
                i2 = '{}_{}'.format(i.partition('(')[0],
                                    i.partition('(')[-1][: -1])
            else:
                i2 = i
            os.system(f'mkdir -p quants/{i2}')
            for j in range(64):
                os.system(
                    'cp -v z-idx/{}/"{}-{}.png" quants/{}/'.format(
                        j, i2, str(j).zfill(2), i2))


if __name__ == "__main__":
    out_dir = '/mnt/lustre/users/hm1234/newTCV/gridscan/'\
        'test/3/5-addT/output_ddt'
    # out_dir = '/home/hm1234/Documents/Project/remotefs/viking/'\
    #     'newTCV/gridscan/test/3/5-addT/output_ddt'

    datList = ['ddt(Ne)', 'ddt(Pe)', 'ddt(Pi)', 'ddt(Vort)', 'ddt(VePsi)',
               'ddt(NVi)', 'Ti', 'Wi', 'Vi', 'S', 'F', 'Qi', 'Rp', 'Rzrad',
               'phi', 'Ve', 'psi', 'Telim', 'Tilim', 'Jpar', 'tau_e', 'tau_i',
               'kappa_epar', 'kappa_ipar', 'nu', 'Pi_ci', 'Pi_ciperp',
               'Pi_cipar', 'NeSource', 'PeSource', 'PiSource', 'Ne', 'Pe',
               'Pi', 'Vort', 'VePsi', 'NVi', 'Nn', 'Pn', 'NVn']

    x = dirAnalysis(out_dir)
