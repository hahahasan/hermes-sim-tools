from __future__ import print_function
from __future__ import division
from builtins import str, range

import os
import glob

from boutdata.collect import collect, create_cache
from boututils.datafile import DataFile
from boututils.boutarray import BoutArray
from boutdata.processor_rearrange import get_processor_layout, create_processor_layout

import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import math
from numpy import mean, zeros, arange
from numpy.random import normal

from scipy.interpolate import interp1d
try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    pass


def file_generator(path, num_chunks):
    lst = glob.glob(os.path.join(path, "BOUT.dmp.*.nc"))
    n = math.ceil(len(lst)/max(1, num_chunks))
    return list((lst[i:i+n] for i in range(0, len(lst), n)))


def multi_create(path, cpus=10, averagelast=1, final=-1, output="./", informat="nc", outformat=None):
    pool = mp.Pool(cpus)

    # Parallel(n_jobs=cpus)(delayed)

    [pool.apply(create,
                args=(subList,
                      averagelast,
                      final,
                      output,
                      informat,
                      outformat)) for subList in file_generator(path, cpus)]
    pool.close()


def create(file_list, path, averagelast=1, final=-1, output="./", informat="nc", outformat=None):
    """Create restart files from data (dmp) files.

    Parameters
    ----------
    averagelast : int, optional
        Number of time points (counting from `final`, inclusive) to
        average over (default is 1 i.e. just take last time-point)
    final : int, optional
        The last time point to use (default is last, -1)
    path : str, optional
        Path to original restart files (default: "data")
    output : str, optional
        Path to write new restart files (default: current directory)
    informat : str, optional
        File extension of original files (default: "nc")
    outformat : str, optional
        File extension of new files (default: use the same as `informat`)

    """

    if outformat is None:
        outformat = informat

    nfiles = len(file_list)

    print(("Number of data files: ", nfiles))

    for i in range(nfiles):
        # Open each data file
        infname = os.path.join(path, "BOUT.dmp."+str(i)+"."+informat)
        outfname = os.path.join(output, "BOUT.restart."+str(i)+"."+outformat)

        print((infname, " -> ", outfname))

        infile = DataFile(infname)
        outfile = DataFile(outfname, create=True)

        # Get the data always needed in restart files
        hist_hi = infile.read("iteration")
        print(("hist_hi = ", hist_hi))
        outfile.write("hist_hi", hist_hi)

        t_array = infile.read("t_array")
        tt = t_array[final]
        print(("tt = ", tt))
        outfile.write("tt", tt)

        tind = final
        if tind < 0.0:
            tind = len(t_array) + final

        NXPE = infile.read("NXPE")
        NYPE = infile.read("NYPE")
        print(("NXPE = ", NXPE, " NYPE = ", NYPE))
        outfile.write("NXPE", NXPE)
        outfile.write("NYPE", NYPE)

        # Get a list of variables
        varnames = infile.list()

        for var in varnames:
            if infile.ndims(var) == 4:
                # Could be an evolving variable

                print((" -> ", var))

                data = infile.read(var)

                if averagelast == 1:
                    slice = data[final, :, :, :]
                else:
                    slice = mean(data[(final - averagelast)
                                 :final, :, :, :], axis=0)

                print(slice.shape)

                outfile.write(var, slice)

        infile.close()
        outfile.close()


if __name__ == "__main__":
    path = "/mnt/lustre/users/hm1234/newTCV2/test/test/grid-03-12-19_111553/0"
    multi_create(path, 5, final=-10,
                 output="/mnt/lustre/users/hm1234/newTCV2/test/test/grid-03-12-19_111553/0/multi_create_test")
