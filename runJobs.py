import numpy as np
import os
import sys
import datetime
import time
from boutdata.restart import addvar
from boutdata.restart import addnoise
from boutdata.restart import resizeZ
from boutdata.restart import redistribute
from inspect import getsource as GS


def func_reqs(obj):
    lines = GS(obj).partition(':')[0]
    print(lines)


def list_files(path):
    files = [f for f in os.listdir(path) if os.path.isfile(f)]
    for f in files:
        print(f)


def replace_line(file_name, line_num, text):
    # replaces lines in a file
    lines = open(file_name, 'r').readlines()
    lines[line_num - 1] = text + '\n'
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


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


def list_grids(densities, shotnum, machine='tcv', resolution='64x64'):
    d_names = []
    for d in densities:
        d_names.append(f'{machine}_{shotnum}_{resolution}_profiles_{d}e19.nc')
    return d_names


class logSim:
    '''
    For logging simulation parameters
    '''
    def __init__(self, location, filename):
        self.fileName = '{}/{}'.format(location, filename)
        self.logFile = open(self.fileName, 'w+')
        self.logFile.close()

    def __call__(self, message):
        self.logFile = open(self.fileName, 'a+')
        self.logFile.write('{}\r\n'.format(message))
        self.logFile.close()


class startSim:
    '''
    Create a simulation object, defined with path to pertinent files and
    where they should be copied to. Also defines methods for modifying
    BOUT.inp file and job submission scripts
    '''
    def __init__(self, pathOut, pathIn, dateDir, inpFile='BOUT.inp',
                 gridFile=None, scanParams=None, title='sim'):
        os.chdir(pathOut)
        self.pathOut = pathOut
        self.pathIn = pathIn
        self.inpFile = inpFile
        self.gridFile = gridFile
        self.runDir = '{}/{}/{}-{}'.format(pathOut, pathIn, title, dateDir)
        self.scanParams = scanParams
        self.title = title
        if self.scanParams is not None:
            self.scanNum = len(scanParams)
        else:
            self.scanNum = 1
        if os.path.isdir('{}/{}'.format(pathOut, pathIn)) is not True:
            os.chdir(pathOut)
            os.mkdir(pathIn)

    def modInp1(self, param, lineNum=None):
        if lineNum is None:
            lineNum = find_line('{}/{}'.format(self.pathOut, self.inpFile),
                                param)
        else:
            lineNum = lineNum
        for i, j in enumerate(self.scanParams):
            os.chdir('{}/{}'.format(self.runDir, i))
            replace_line('{}'.format(self.inpFile),
                         lineNum,
                         '{} = {}'.format(param, j))

    def modInp2(self, param, value, lineNum=None):
        self.log('Modified {} to: {}'.format(param, value))
        if lineNum is None:
            lineNum = find_line('{}/{}'.format(self.pathOut, self.inpFile),
                                param)
        else:
            lineNum = lineNum
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            replace_line('{}'.format(self.inpFile),
                         lineNum,
                         '{} = {}'.format(param, value))

    def modJob(self, nProcs, hermesVer, tme, optNodes=True):
        self.log('nProcs: {}'.format(nProcs))
        self.log('hermesVer: {}'.format(hermesVer))
        self.log('simTime: {}'.format(tme))
        if optNodes is True:
            nodes = int(np.ceil(nProcs/40))
            for i in range(self.scanNum):
                os.chdir('{}/{}'.format(self.runDir, i))
                replace_line('{}.job'.format(self.title),
                             find_line('{}.job'.format(self.title),
                                       '--nodes'),
                             '#SBATCH --nodes={}'.format(nodes))
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            # os.system('cp {}/test.job {}.job'.format(
            #     self.pathOut, self.title))
            replace_line('{}.job'.format(self.title),
                         find_line('{}.job'.format(self.title),
                                   '--ntasks'),
                         '#SBATCH --ntasks={}'.format(nProcs))
            replace_line('{}.job'.format(self.title),
                         find_line('{}.job'.format(self.title),
                                   'mpiexec'),
                         'mpiexec -n {} {} -d {}/{}'.format(nProcs,
                                                            hermesVer,
                                                            self.runDir,
                                                            i))
            replace_line('{}.job'.format(self.title),
                         find_line('{}.job'.format(self.title),
                                   '--job-name'),
                         '#SBATCH --job-name={}-{}'.format(self.title, i))
            replace_line('{}.job'.format(self.title),
                         find_line('{}.job'.format(self.title),
                                   '--time'),
                         '#SBATCH --time={}'.format(tme))

    def setup(self):
        os.mkdir('{}'.format(self.runDir))
        os.chdir('{}'.format(self.runDir))
        self.log = logSim(self.runDir, 'log.txt')
        self.log('title: {}'.format(self.title))
        self.log('inpFile: {}'.format(self.inpFile))
        self.log('gridFile: {}'.format(str(self.gridFile)))
        self.log('scanParams: {}'.format(str(self.scanParams)))
        for i in range(self.scanNum):
            os.mkdir(str(i))
            os.system('cp {}/{} {}/BOUT.inp'.format(self.pathOut,
                                                    self.inpFile, i))
            os.system('cp {}/{} {}/{}.job'.format(
                self.pathOut, 'test.job', i, self.title))
            if type(self.gridFile) == str:
                os.system('cp /users/hm1234/scratch/gridfiles/{} {}'.format(
                    self.gridFile, i))
        self.inpFile = 'BOUT.inp'
        if self.gridFile is not None:
            self.modInp2('grid', self.gridFile)

    def subJob(self, shortQ=False):
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            if shortQ is False:
                os.system('sbatch {}.job'.format(self.title))
            elif shortQ is True:
                os.system('sbatch -q short {}.job'.format(self.title))


class slabSim(startSim):
    pass
    # def __init__(self, pathOut, pathIn, dateDir, inpFile,
    #              gridFile=None, scanParams=None, title='sim'):
    #     super().__init__(pathOut, pathIn, dateDir, inpFile, None,
    #                      scanParams=None, title='sim')

    # def setup(self):
    #     os.mkdir('{}'.format(self.runDir))
    #     os.chdir('{}'.format(self.runDir))
    #     self.log = logSim(self.runDir, 'log.txt')
    #     self.log('title: {}'.format(self.title))
    #     self.log('inpFile: {}'.format(self.inpFile))
    #     self.log('gridFile: {}'.format(str(self.gridFile)))
    #     self.log('scanParams: {}'.format(str(self.scanParams)))
    #     for i in range(self.scanNum):
    #         os.mkdir(str(i))
    #         os.system('cp {}/{} {}/BOUT.inp'.format(self.pathOut,
    #                                                 self.inpFile, i))
    #     self.inpFile = 'BOUT.inp'


class multiGridSim(startSim):
    def __init__(self, pathOut, pathIn, dateDir, inpFile,
                 scanParams, title='sim'):
        super().__init__(pathOut, pathIn, dateDir, inpFile, 'blah',
                         scanParams, title)
        self.gridFile = self.scanParams

    def setup(self):
        super().setup()
        for i in range(self.scanNum):
            os.system('cp /users/hm1234/scratch/gridfiles/{} {}/{}'.format(
                self.pathOut, self.scanParams[i], self.runDir, i))
        self.modInp1('grid')


class addSim:
    def __init__(self, runDir, scanIDs=[], logFile='log.txt'):
        self.runDir = runDir
        os.chdir(runDir)
        self.scanParams = read_line(logFile, 'scanParams')
        if len(scanIDs) == 0:
            if self.scanParams is not None:
                self.scanIDs = list(np.arange(len(self.scanParams)))
            else:
                self.scanIDs = [0]
        else:
            self.scanIDs = scanIDs
        if self.scanParams is not None:
            self.scanNum = len(self.scanParams)
        else:
            self.scanNum = 1
        self.title = read_line(logFile, 'title')
        # self.inpFile = read_line(logFile, 'inpFile')
        self.inpFile = 'BOUT.inp'
        self.gridFile = read_line(logFile, 'gridFile')
        self.hermesVer = read_line(logFile, 'hermesVer')
        self.nProcs = read_line(logFile, 'nProcs')

    def copyNewInp(self, oldDir, inpName):
        for i in self.scanIDs:
            os.system('cp {}/{} {}/{}/{}/BOUT.inp'.format(
                oldDir, inpName, self.runDir, i, self.addType))

    def modInp(self, param, lineNum=None):
        scanParams = []
        for i in self.scanIDs:
            scanParams.append(self.scanParams[i])
        for i in self.scanIDs:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            if lineNum is None:
                lineNum = find_line('{}/{}/{}/{}'.format(
                    self.runDir, i, self.addType, self.inpFile),
                                    param)
            else:
                lineNum = lineNum
            for j in scanParams:
                replace_line('{}'.format(self.inpFile),
                             lineNum,
                             '{} = {}'.format(param, j))

    def copyInpFiles(self, oldDir=None, addType='restart'):
        self.addType = addType
        for i in self.scanIDs:
            os.chdir('{}/{}'.format(self.runDir, i))
            os.system('mkdir -p {}'.format(addType))
            # print(os.system('pwd'))
            # os.system('cp {}/{}/{}/BOUT.inp {}'.format(self.runDir, i,
            #                                            oldDir, addType))
            if type(self.gridFile) == list:
                os.system('cp {} {}'.format(self.gridFile[i], addType))
            elif self.gridFile is None:
                pass
            else:
                os.system('cp {} {}'.format(self.gridFile, addType))
            os.system('cp *.job {}/{}.job'.format(addType, addType))
            if oldDir is None:
                cmd = 'cp {} {}'.format(self.inpFile, addType)
            else:
                cmd = 'cp {}/{} {}'.format(oldDir, self.inpFile, addType)
            os.system(cmd)

    def copyRestartFiles(self, oldDir=None, addType='restart'):
        if oldDir is None:
            cmd = 'cp BOUT.restart.* {}'.format(addType)
        else:
            cmd = 'cp {}/BOUT.restart.* {}'.format(oldDir, addType)
        for i in self.scanIDs:
            os.chdir('{}/{}'.format(self.runDir, i))
            os.system(cmd)

    def redistributeProcs(self, oldDir, addType, npes):
        self.copyInpFiles(oldDir, addType)
        for i in self.scanIDs:
            os.chdir('{}/{}'.format(self.runDir, i))
            redistribute(npes=npes, path=oldDir, output=addType)
        self.nProcs = npes

    def modFile(self, param, value, lineNum=None):
        for i in self.scanIDs:
            if lineNum is None:
                lineNum = find_line('{}/{}/{}/{}'.format(
                    self.runDir, i, self.addType, self.inpFile),
                                    param)
            else:
                lineNum = lineNum
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            replace_line('{}'.format(self.inpFile),
                         lineNum,
                         '{} = {}'.format(param, value))

    def modJob(self, tme, nProcs=None, optNodes=True):
        if nProcs is None:
            nProcs = self.nProcs
        if optNodes is True:
            nodes = int(np.ceil(nProcs/40))
            for i in self.scanIDs:
                os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
                replace_line('{}.job'.format(self.addType),
                             find_line('{}.job'.format(self.addType),
                                       '--nodes'),
                             '#SBATCH --nodes={}'.format(nodes))
        for i in self.scanIDs:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--ntasks'),
                         '#SBATCH --ntasks={}'.format(nProcs))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   'mpiexec'),
                         'mpiexec -n {} {} -d {}/{}/{} restart'.format(
                             nProcs, self.hermesVer, self.runDir,
                             i, self.addType))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--job-name'),
                         '#SBATCH --job-name={}-{}'.format(self.addType, i))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--time'),
                         '#SBATCH --time={}'.format(tme))

    def subJob(self, shortQ=False):
        for i in self.scanIDs:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            if shortQ is False:
                os.system('sbatch {}.job'.format(self.addType))
            elif shortQ is True:
                os.system('sbatch -q short {}.job'.format(self.addType))


class addNeutrals(addSim):
    def addVar(self, Nn=0.1, Pn=0.05):
        for i in self.scanIDs:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            addvar('Nn', Nn)
            addvar('Pn', Pn)


class addCurrents(addSim):
    pass


class restartSim(addSim):
    pass


class addTurbulence(addSim):
    '''
    make sure to use
    export
    PYTHONPATH=/mnt/lustre/groups/phys-bout-2019/BOUT-dev/tools/pylib/:$PYTHONPATH
    '''
    def addTurb(self, oldDir, addType,  npes=None, MZ=64,
                param='Vort', pScale=1e-5, multiply=True):
        if npes is not None:
            tempDir = 'temp-turb'
            self.redistributeProcs(npes=npes, oldDir=oldDir, addType=tempDir)
            oldDir = tempDir
        self.copyInpFiles(oldDir, addType)
        for i in self.scanIDs:
            os.chdir('{}/{}'.format(self.runDir, i))
            resizeZ(newNz=MZ, path=oldDir, output=addType)
            addnoise(path=addType, var=param, scale=pScale)
        self.modFile('nz', MZ)
        if npes is not None:
            for i in self.scanIDs:
                os.chdir('{}/{}'.format(self.runDir, i))
                os.system('rm -rf {}'.format(tempDir))


if __name__ == "__main__":
    inpFile = 'BOUT.inp'
    gridFile = 'tcv_52068_64x64_profiles_1e19.nc'
    gridFile = 'tcv_63127_64x64_profiles_1.2e19.nc'

    pathOut = '/users/hm1234/scratch/newTCV2'
    pathIn = 'gridscan2'
    pathIn = 'gridscan'
    pathIn = 'init'
    pathIn = 'gridscan3'
    pathIn = 'hdscan'
    # pathIn = 'high_recycle'
    # pathIn = 'high_density'
    dateDir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    # dateDir = '_turbTest'

    # title = 'cfrac'
    # scanParams = [0.01, 0.02, 0.03, 0.05, 0.07]

    title = 'rfrac'
    scanParams = [0.9, 0.93, 0.96, 0.99]
    # scanParams = [0.95]

    nProcs = 234
    tme = '00:19:19'  # day-hr:min:sec
    # tme = '10:10:00'
    # hermesVer = '/users/hm1234/scratch/BOUT-test4/hermes-2/hermes-2'
    # hermesVer = '/mnt/lustre/groups/phys-bout-2019/hermes-2-next/hermes-2'
    # hermesVer = '/users/hm1234/scratch/BOUT25Jun19/hermes-2/hermes-2'
    hermesVer = '/users/hm1234/scratch/BOUT5Jul19/hermes-2/hermes-2'
    hermesVer = '/users/hm1234/scratch/BOUT18Sep19/hermes-2/hermes-2'
    hermesVer = '/users/hm1234/scratch/BOUT28Oct19/hermes-2/hermes-2'
    hermesVer = '/users/hm1234/scratch/BOUT21Nov19/hermes-2v2/hermes-2'
    hermesVer = '/users/hm1234/scratch/BOUT21Nov19/test/hermes-2/hermes-2'

    title = 'grid'

    grids = ['tcv_63127_256x64_profiles_2e19.nc',
             'tcv_63127_256x64_profiles_4e19.nc',
             'tcv_63127_256x64_profiles_6e19.nc',
             'tcv_63127_256x64_profiles_8e19.nc']
             # 'tcv_63127_256x64_profiles_10e19.nc']
    # grids = ['tcv_63161_64x64_profiles_1e19.nc',
    #          'tcv_63161_64x64_profiles_2e19.nc',
    #          'tcv_63161_64x64_profiles_3e19.nc',
    #          'tcv_63161_64x64_profiles_3.25e19.nc',
    #          'tcv_63161_64x64_profiles_3.5e19.nc',
    #          'tcv_63161_64x64_profiles_3.75e19.nc',
    #          'tcv_63161_64x64_profiles_4.5e19.nc',
    #          'tcv_63161_64x64_profiles_5.5e19.nc']
    # grids = ['tcv_63161_64x64_profiles_6e19.nc',
    #          'tcv_63161_64x64_profiles_6.5e19.nc',
    #          'tcv_63161_64x64_profiles_7e19.nc',
    #          'tcv_63161_64x64_profiles_7.5e19.nc',
    #          'tcv_63161_64x64_profiles_8.2e19.nc']
    # gridFile = 'tcv_63127_64x64_profiles_1.6e19.nc'

    # grids = ['newtcv_63127_64x64_profiles_22e19.nc',
    #          'newtcv_63161_64x64_profiles_22e19.nc']

    # grids = ['tcv_63161_64x64_profiles_8.9e19.nc',
    #          'tcv_63161_64x64_profiles_9.6e19.nc',
    #          'tcv_63161_64x64_profiles_10.2e19.nc',
    #          'tcv_63161_64x64_profiles_11e19.nc',
    #          'tcv_63161_64x64_profiles_12e19.nc']

    # grids = ['tcv_63127_64x64_profiles_9.9e19.nc',
    #          'tcv_63127_64x64_profiles_10.5e19.nc',
    #          'tcv_63127_64x64_profiles_11e19.nc',
    #          'tcv_63127_64x64_profiles_12e19.nc',
    #          'tcv_63127_64x64_profiles_13e19.nc']

    title = 'gauss'
    tme = '08:88:88'
    # tme = '00:19:59'
    nProcs = 512
    hermesVer = '/users/hm1234/scratch/hermes2/9Jan20/hermes-2'

    slabSim = slabSim('/users/hm1234/scratch/slabTCV', '2020runs',
                      dateDir, 'BOUT5.inp', title=title)
    slabSim.setup()
    slabSim.modInp2('NOUT', 222)
    slabSim.modInp2('TIMESTEP', 22)
    # power of 3 still fast using fft but maybe more
    # robust to triangular instabilities)
    # slabSim.modInp2('nz', 243)
    # slabSim.modInp2('ny', 32, lineNum=18)
    # slabSim.modInp2('ramp_j_diamag', 1.0)
    slabSim.modJob(nProcs, hermesVer, tme)
    slabSim.subJob(shortQ=False)

    tme = '08:88:88'
    # runDir = '/users/hm1234/scratch/slabTCV/2020runs/slab-17-01-20_104716'
    runDir = '/users/hm1234/scratch/slabTCV/2020runs/gauss-24-01-20_155235'
    # res = restartSim(runDir,)
    # old = None
    # new = '2-moreTime'
    # res.copyInpFiles(old, new)
    # res.copyRestartFiles(old, new)
    # # res.modFile('output_ddt', 'true')
    # res.modFile('NOUT', 128)
    # res.modFile('TIMESTEP', 32)
    # # res.modFile('ion_viscosity', 'false')
    # res.modFile('ramp_j_diamag', 1.0)
    # res.modJob(tme)
    # res.subJob()

    # runDir = '/users/hm1234/scratch/slabTCV/2020runs/sim-13-01-20_112202'
    # tme = '12:12:12'
    # addN = addNeutrals(runDir)
    # addType = '2-addN'
    # addN.copyInpFiles(addType=addType)
    # addN.copyRestartFiles(addType=addType)
    # # addN.copyNewInp(oldDir='/users/hm1234/scratch/newTCV',
    # #                 inpName='BOUT-2Dworks.inp')
    # addN.modFile('NOUT', 100)
    # addN.modFile('TIMESTEP', 2)
    # # addN.modFile('neutral_friction', 'true')
    # addN.modFile('type', 'mixed', lineNum=241)
    # addN.modJob(tme, optNodes=True)
    # addN.addVar(Nn=0.04, Pn=0.02)
    # addN.subJob()

    # grids = ['tcv_63127_64x64_profiles_1.6e19.nc',
    #          'tcv_63127_64x64_profiles_4.0e19.nc']

    grids = ['tcv_63161_128x64_profiles_1e19.nc',
             'tcv_63161_128x64_profiles_2e19.nc',
             'tcv_63161_128x64_profiles_3e19.nc',
             'tcv_63161_128x64_profiles_4e19.nc',
             'tcv_63161_128x64_profiles_5e19.nc',
             'tcv_63161_128x64_profiles_6e19.nc',
             'tcv_63161_128x64_profiles_7e19.nc',
             'tcv_63161_128x64_profiles_8e19.nc',
             'tcv_63161_128x64_profiles_9e19.nc',
             'tcv_63161_128x64_profiles_10e19.nc']

    grids = ['tcv_63161_128x64_profiles_1e19.nc',
             'tcv_63161_128x64_profiles_5e19.nc',
             'tcv_63161_128x64_profiles_9e19.nc']

    grids = ['tcv_63161_extendedpsi_64x64_profiles_1e19.nc',
             'tcv_63161_extendedpsi_64x64_profiles_3e19.nc',
             'tcv_63161_extendedpsi_64x64_profiles_5e19.nc']

    # qgrids = ['tcv_63161_128x64_profiles_1e19.nc']

    # inpFile = 'BOUT2.inp'
    # title = 'expsi'
    # nProcs = 256
    # tme = '06:66:66'
    # gridSim = multiGridSim(pathOut, pathIn, dateDir, inpFile, grids, title)
    # gridSim.setup()
    # gridSim.modInp2('carbon_fraction', 0.04)
    # gridSim.modInp2('frecycle', 0.99)
    # gridSim.modInp2('NOUT', 444)
    # gridSim.modInp2('TIMESTEP', 222)
    # gridSim.modJob(nProcs, hermesVer, tme)
    # gridSim.subJob()

    # pathOut = '/users/hm1234/scratch/slabTCV/'
    # pathIn = 'test'
    # inpFile = 'BOUT2.inp'
    # nProcs = 192
    # tme = '00:11:11'
    # title = 'slab'
    # scanParams = [0.0]
    # sim1 = slabSim(pathOut, pathIn, dateDir, inpFile, scanParams, title)
    # sim1.setup()
    # sim1.modInp1('carbon_fraction')
    # # sim1.modInp2('carbon_fraction', 0.95)
    # # sim1.modInp2('ion_viscosity', 'true')
    # sim1.modInp2('NOUT', 444)
    # sim1.modInp2('TIMESTEP', 222)
    # # sim1.modInp2('carbon_fraction', 0.04)
    # sim1.modJob(nProcs, hermesVer, tme)
    # sim1.subJob(shortQ=True)

    # runDir = '/users/hm1234/scratch/TCV/NeScan2/NeScan-03-06-19_171145'
    # runDir = '/users/hm1234/scratch/TCV/NeScan2/frecycle-05-06-19_145457'
    # runDir = '/users/hm1234/scratch/TCV/longtime/cfrac-10-06-19_175728'
    # runDir = '/users/hm1234/scratch/TCV/longtime/rfrac-19-06-19_102728'
    # runDir = '/users/hm1234/scratch/TCV2/gridscan/grid-20-06-19_135947'
    # runDir = '/users/hm1234/scratch/newTCV/gridscan/grid-01-07-19_185351'
    # runDir = '/users/hm1234/scratch/newTCV/gridscan/test'
    runDir = '/users/hm1234/scratch/newTCV/turb-test/g-18-07-19_133047'
    runDir = '/users/hm1234/scratch/newTCV/scans/cfrac-23-07-19_163139'
    runDir = '/users/hm1234/scratch/newTCV/scans/rfrac-25-07-19_162302'
    runDir = '/users/hm1234/scratch/newTCV/gridscan/grid-07-09-19_180613'
    runDir = '/users/hm1234/scratch/newTCV/gridscan/grid-12-09-19_165234'
    # runDir = '/users/hm1234/scratch/newTCV/gridscan2/grid-13-09-19_153544'
    # runDir = '/users/hm1234/scratch/newTCV/gridscan2/grid-18-09-19_111405'
    runDir = '/users/hm1234/scratch/newTCV/gridscan2/grid-23-09-19_140426'
    # runDir = '/users/hm1234/scratch/newTCV/high_recycle/grid-25-09-19_165128'
    # runDir = '/users/hm1234/scratch/newTCV/high_density/grid-28-10-19_133357'
    # runDir = '/users/hm1234/scratch/newTCV/gridscan2/grid-07-11-19_154854'
    runDir = '/users/hm1234/scratch/newTCV/gridscan/grid-07-11-19_155631'
    runDir = '/users/hm1234/scratch/slabTCV/init/slab-25-11-19_143529'
    runDir = '/users/hm1234/scratch/newTCV2/hdscan/hdg-02-12-19_172620'
    runDir = '/users/hm1234/scratch/newTCV2/hdscan/expsi-04-12-19_203752'

    # tme = '06:66:33'
    # addN = addNeutrals(runDir)
    # addType = '2-addN'
    # addN.copyInpFiles(addType=addType)
    # addN.copyRestartFiles(addType=addType)
    # # addN.copyNewInp(oldDir='/users/hm1234/scratch/newTCV',
    # #                 inpName='BOUT-2Dworks.inp')
    # addN.modFile('NOUT', 555)
    # addN.modFile('TIMESTEP', 150)
    # # addN.modFile('neutral_friction', 'true')
    # addN.modFile('type', 'mixed', lineNum=229)
    # addN.modJob(tme)
    # addN.addVar(Nn=0.04, Pn=0.02)
    # addN.subJob()

    # tme = '06:66:66'
    # res = restartSim(runDir, scanIDs=[0])
    # old = '2-addN'
    # new = '2.2-addN'
    # res.copyInpFiles(old, new)
    # res.copyRestartFiles(old, new)
    # # res.modFile('adapt_source', 'false')
    # res.modJob(tme)
    # res.subJob()

    # res = restartSim(runDir, scanIDs=[0])
    # old = '2.1-resN'
    # new = '2.2-modSolver'
    # res.copyInpFiles(old, new)
    # res.copyRestartFiles(old, new)
    # # res.modFile('use_precon', 'false')
    # res.modFile('maxl', '5')
    # res.modJob(tme)
    # res.subJob()

    # tme = '23:59:59'
    # addN = addNeutrals(runDir)
    # addType = '2-addN2'
    # addN.copyInpFiles(addType=addType)
    # addN.copyRestartFiles(addType=addType)
    # # addN.copyNewInp(oldDir='/users/hm1234/scratch/newTCV',
    # #                 inpName='BOUT-2Dworks.inp')
    # addN.modFile('NOUT', 555)
    # addN.modFile('TIMESTEP', 150)
    # # addN.modFile('neutral_friction', 'true')
    # addN.modFile('type', 'mixed', lineNum=214)
    # addN.modFile('split_n0 ', 'true')
    # addN.modFile('split_n0_psi', 'true')
    # addN.modJob(tme)
    # addN.addVar(Nn=0.04, Pn=0.02)
    # addN.subJob()

    # tme = '1-23:59:59'
    # old = '2-addN'
    # new = '3.1-addC'
    # addC = addCurrents(runDir)
    # addC.copyInpFiles(old, new)
    # addC.copyRestartFiles(old, new)
    # addC.modFile('j_par', 'true')
    # addC.modFile('j_diamag', 'true')
    # # addC.modFile('split_n0 ', 'false')
    # # addC.modFile('split_n0_psi', 'false')
    # # addC.modFile('adapt_source', 'false')
    # addC.modFile('NOUT', 444)  # 600
    # addC.modFile('TIMESTEP', 2)  # 333
    # addC.modJob(tme, optNodes=True)
    # addC.subJob()

    # tme = '1-23:59:59'
    # old = '3-addC'
    # new = '4-addT'
    # addT = addTurbulence(runDir, scanIDs=[0])
    # newProcs = 512
    # addT.addTurb(old, new, npes=newProcs, MZ=256)
    # # addT.modFile('nz', 256)
    # addT.modFile('NOUT', 222)
    # addT.modFile('TIMESTEP', 5)
    # addT.modJob(tme=tme, nProcs=newProcs, optNodes=True)
    # addT.subJob()

    # tme = '1-23:59:59'
    # old = '3-addC'
    # new = '3.1-modPI'
    # addC = addCurrents(runDir, scanIDs=[0,1,2])
    # addC.copyInpFiles(old, new)
    # addC.copyRestartFiles(old, new)
    # addC.modFile('source_p', 1e-1)
    # addC.modFile('source_i', 1e-4)
    # # addC.modFile('split_n0 ', 'false')
    # # addC.modFile('split_n0_psi', 'false')
    # # addC.modFile('adapt_source', 'false')
    # # addC.modFile('NOUT', 500)  # 600
    # # addC.modFile('TIMESTEP', 500)  # 333
    # addC.modJob(tme)
    # addC.subJob()

    # addC = addCurrents(runDir)
    # addC.copyInpFiles(addType='2-addC')
    # addC.copyRestartFiles(addType='2-addC')
    # addC.modFile('j_par', 'true')
    # addC.modFile('j_diamag', 'true')
    # addC.modFile('TIMESTEP', 333)
    # addC.modJob(tme)
    # addC.subJob()

    # tme = '1-23:59:59'
    # old = '3-addC'
    # new = '3-resC'
    # resC = addCurrents(runDir)
    # resC.copyInpFiles(old, new)
    # resC.copyRestartFiles(old, new)
    # # addC.modFile('psi_')
    # resC.modFile('NOUT', 666)
    # resC.modFile('TIMESTEP', 444)
    # resC.modJob(tme)
    # resC.subJob()

    # tme = '1-23:59:59'
    # old = '3-resC3'
    # new = '4-addD'
    # addD = addCurrents(runDir)
    # addD.copyInpFiles(old, new)
    # addD.copyRestartFiles(old, new)
    # # addC.modFile('psi_')
    # addD.modFile('NOUT', 666)
    # addD.modFile('TIMESTEP', 444)
    # addD.modFile('pctype', 'hypre')
    # addD.modFile('neutral_gamma', 0.3, lineNum=218)
    # addD.modJob(tme)
    # addD.subJob()

    # tme = '1-23:59:59'
    # old = '3-addC2'
    # new = '4-split'
    # addC = addCurrents(runDir)
    # addC.copyInpFiles(old, new)
    # addC.copyRestartFiles(old, new)
    # addC.modFile('j_par', 'true')
    # addC.modFile('j_diamag', 'true')
    # addC.modFile('split_n0 ', 'true')
    # addC.modFile('split_n0_psi', 'true')
    # # addC.modFile('psi_')
    # addC.modFile('NOUT', 555)
    # addC.modFile('TIMESTEP', 55)
    # addC.modJob(tme)
    # addC.subJob()

    # tme = '23:55:55'
    # # addT = testTurbulence(runDir)
    # # addT.copyFiles2('3-addC', '4-addT')
    # # addT.modJob(tme)
    # addT = addTurbulence(runDir, scanIDs=[3])
    # addT.hermesVer = hermesVer
    # # addT.redistributeProcs('3-addC', '4-redistribute', 480)
    # # addT.addTurb('4-redistribute', '5-addT')
    # addT.addTurb('3-addC', '5-addT')
    # # addT.copyInpFiles('3-addC', '5-addT')
    # addT.copyNewInp(runDir, 'BOUT2.inp')
    # addT.modInp('grid')
    # addT.modFile('TIMESTEP', 0.002)
    # addT.modJob(tme)
    # addT.subJob()

    # addT = addTurbulence(runDir)
    # addT.copyInpFiles('3-addC', '4-addT')
    # addT.addTurb('3-addC', '4-addT')
    # addT.modJob(tme)
    # addT.modFile('MZ', 64)
    # addT.modFile('NOUT', 333)
    # addT.modFile('TIMESTEP', 0.04)
    # addT.modFile('output_ddt', 'true')
    # addT.modFile('verbose', 'true')
    # addT.subJob()
