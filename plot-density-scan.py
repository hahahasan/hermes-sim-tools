
import matplotlib.pyplot as plt
from boututils.datafile import DataFile
import numpy as np
from boutdata import collect


# Relative paths to directories containing simulation outputs
# The last time point is taken from each of these
paths = ["1e19", "2e19/part-03", "3e19/part-03", "4e19"]
colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

gridfile = "TCV_52061_t1.0_64x64_2e19.nc"
grid = DataFile(gridfile)
Rxy = grid["Rxy"]

ymid = np.argmax(Rxy[-1,:])
xsep = grid["ixseps1"]

rmid = Rxy[:,ymid]
rtarg = Rxy[:,-1]

fig, axs = plt.subplots(2, 3)

for path, color in zip(paths, colors):
    n = collect("ne", tind=-1, path=path)
    nnorm = collect("nnorm", path=path)
    n *= nnorm

    te = collect("te", tind=-1, path=path)
    tnorm = collect("tnorm", path=path)
    te *= tnorm

    axs[0,0].plot(rmid, n[0,:,ymid,0], color=color)
    axs[0,0].axvline(rmid[xsep], color="k", linestyle="--")
    
    axs[1,0].plot(rmid, te[0,:,ymid,0], color=color)
    axs[1,0].axvline(rmid[xsep], color="k", linestyle="--")

    axs[0,1].plot(rtarg, n[0,:,-1,0], color=color)
    axs[0,1].axvline(rtarg[xsep], color="k", linestyle="--") 

    axs[1,1].plot(rtarg, te[0,:,-1,0], color=color)
    axs[1,1].axvline(rtarg[xsep], color="k", linestyle="--") 

    # Density at separatrix
    nsep = 0.5 * (n[0,xsep-1,ymid,0] + n[0,xsep,ymid,0])

    # Peak target temperature
    targte = np.amax(te[0,:,-1,0])

    axs[0,2].plot([nsep], [targte], color=color, marker="o")

    nvi = collect("nvi", tind=-1, path=path, yguards=True)
    cs0 = collect("cs0", path=path)
    nvi = nnorm * cs0 * 0.5*(nvi[0,:,-3,0] + nvi[0,:,-2,0])
    targflux = np.amax(nvi)

    axs[1,2].plot([nsep], [targflux], color=color, marker="o")

axs[0,0].xaxis.set_ticklabels([])
axs[0,1].xaxis.set_ticklabels([])
axs[0,2].xaxis.set_ticklabels([])


axs[0,0].set_ylabel(r"Density [m$^{-3}$]")
axs[1,0].set_ylabel("Electron temperature [eV]")
axs[1,0].set_xlabel("Radius at midplane [m]")
axs[1,1].set_xlabel("Radius at target [m]")

axs[0,2].yaxis.tick_right()
axs[0,2].yaxis.set_label_position("right")

axs[1,2].yaxis.tick_right()
axs[1,2].yaxis.set_label_position("right")

axs[0,2].set_ylabel("Peak target Te [eV]")
axs[1,2].set_ylabel(r"Peak target flux [m$^{-2}$s$^{-1}$")
axs[1,2].set_xlabel(r"Separatrix density [m$^{-3}$]")

plt.savefig("profiles.png")
plt.savefig("profiles.pdf")
plt.show()
