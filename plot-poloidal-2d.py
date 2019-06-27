import matplotlib.pyplot as plt
from boutdata import collect
from boutdata.griddata import gridcontourf
from boututils.datafile import DataFile

#path = "2e19/part-03"
path = "4e19"

gridfile = "TCV_52061_t1.0_64x64_2e19.nc"
grid = DataFile(gridfile)

t = collect("t_array", path=path).ravel()

nt = len(t)
time = t[-1]

n = collect("Ne", path=path, tind=nt-1)
nnorm = collect("Nnorm", path=path)
n *= nnorm

gridcontourf(grid, n[0,:,:,0], show=False)
plt.title(r"Density [m$^{-3}$]")
plt.savefig(path+"/density.pdf")
plt.savefig(path+"/density.png")
plt.show()

te = collect("Te", path=path, tind=nt-1)
tnorm = collect("Tnorm", path=path)
te *= tnorm
gridcontourf(grid, te[0,:,:,0], show=False)
plt.title(r"Electron temperature [eV]")
plt.savefig(path+"/electron-temperature.pdf")
plt.savefig(path+"/electron-temperature.png")
plt.show()

ti = collect("Ti", path=path, tind=nt-1)
ti *= tnorm
gridcontourf(grid, ti[0,:,:,0], show=False)
plt.title(r"Ion temperature [eV]")
plt.savefig(path+"/ion-temperature.pdf")
plt.savefig(path+"/ion-temperature.png")
plt.show()

cmap = plt.cm.get_cmap("bwr")

phi = collect("phi", path=path, tind=nt-1)
phi *= tnorm
gridcontourf(grid, phi[0,:,:,0], show=False, symmetric=True, cmap=cmap)
plt.title(r"Electrostatic potential [eV]")
plt.savefig(path+"/potential.pdf")
plt.savefig(path+"/potential.png")
plt.show()

cs = collect("cs", path=path)

nvi = collect("nvi", path=path, tind=nt-1)
nvi *= nnorm * cs

gridcontourf(grid, nvi[0,:,:,0], show=False, symmetric=True, cmap=cmap)
plt.title(r"Parallel flow [m$^{-2}$s$^{-1}$]")
plt.savefig(path+"/parallel-flow.pdf")
plt.savefig(path+"/parallel-flow.png")
plt.show()

ve = collect("ve", path=path, tind=nt-1)
ve *= cs

jpar = 1.602e-19 * (nvi - n * ve)

# jpar = collect("jpar", path=path, tind=nt-1)
# jpar *= 1.602e-19 * nnorm * cs

gridcontourf(grid, jpar[0,:,:,0], show=False, symmetric=True, cmap=cmap)
plt.title(r"Parallel current [Am$^{-2}$]")
plt.savefig(path+"/parallel-current.pdf")
plt.savefig(path+"/parallel-current.png")
plt.show()
