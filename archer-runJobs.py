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
    def __init__(self, pathOut, pathIn, dateDir, inpFile, gridFile,
                 scanParams, title='sim'):
        os.chdir(pathOut)
        self.pathOut = pathOut
        self.pathIn = pathIn
        self.inpFile = inpFile
        self.gridFile = gridFile
        self.runDir = '{}/{}/{}-{}'.format(pathOut, pathIn, title, dateDir)
        self.scanParams = scanParams
        self.title = title
        self.scanNum = len(scanParams)
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

    def modJob(self, nProcs, hermesVer, tme):
        self.log('nProcs: {}'.format(nProcs))
        self.log('hermesVer: {}'.format(hermesVer))
        self.log('simTime: {}'.format(tme))
        nNodes = int(np.ceil(nProcs/24))
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            os.system('cp {}/job.pbs {}.pbs'.format(self.pathOut, self.title))
            replace_line('{}.pbs'.format(self.title),
                         find_line('{}.pbs'.format(self.title),
                                   'select'),
                         '#PBS -l select={}'.format(nNodes))
            replace_line('{}.pbs'.format(self.title),
                         find_line('{}.pbs'.format(self.title),
                                   'aprun'),
                         'aprun -n {} {} -d {}/{} 2>&1 {}/{}/zzz'.format(
                             nProcs, hermesVer,
                             self.runDir, i,
                             self.runDir, i))
            replace_line('{}.pbs'.format(self.title),
                         find_line('{}.pbs'.format(self.title),
                                   'jobname') + 1,
                         '#PBS -N {}-{}'.format(self.title, i))
            replace_line('{}.pbs'.format(self.title),
                         find_line('{}.pbs'.format(self.title),
                                   'walltime'),
                         '#PBS -l walltime={}'.format(tme))
            replace_line('{}.pbs'.format(self.title),
                         find_line('{}.pbs'.format(self.title),
                                   'PBS_O_WORKDIR') + 1,
                         'cd {}/{}'.format(self.runDir, i))

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
            if type(self.gridFile) == str:
                os.system('cp {}/{} {}'.format(self.pathOut, self.gridFile, i))
        self.inpFile = 'BOUT.inp'
        self.modInp2('grid', self.gridFile)

    def subJob(self, shortQ=False):
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            if shortQ is False:
                os.system('qsub {}.pbs'.format(self.title))
            elif shortQ is True:
                os.system('qsub -q short {}.pbs'.format(self.title))


class multiGridSim(startSim):
    def __init__(self, pathOut, pathIn, dateDir, inpFile,
                 scanParams, title='sim'):
        super().__init__(pathOut, pathIn, dateDir, inpFile, 'blah',
                         scanParams, title)
        self.gridFile = self.scanParams

    def setup(self):
        super().setup()
        for i in range(self.scanNum):
            os.system('cp {}/{} {}/{}'.format(self.pathOut, self.scanParams[i],
                                              self.runDir, i))
        self.modInp1('grid')


class addSim:
    def __init__(self, runDir, scanIDs=[], logFile='log.txt'):
        self.runDir = runDir
        os.chdir(runDir)
        self.scanParams = read_line(logFile, 'scanParams')
        if len(scanIDs) == 0:
            self.scanIDs = list(np.arange(len(self.scanParams)))
        else:
            self.scanIDs = scanIDs
        self.scanNum = len(self.scanParams)
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
            if type(self.gridFile) != str:
                os.system('cp {} {}'.format(self.gridFile[i], addType))
            else:
                os.system('cp {} {}'.format(self.gridFile, addType))
            os.system('cp *.pbs {}/{}.pbs'.format(addType, addType))
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

    def modJob(self, tme):
        nNodes = int(np.ceil(self.nProcs/24))
        for i in self.scanIDs:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            replace_line('{}.pbs'.format(self.addType),
                         find_line('{}.pbs'.format(self.addType),
                                   'select'),
                         '#PBS -l select={}'.format(nNodes))
            replace_line('{}.pbs'.format(self.addType),
                         find_line('{}.pbs'.format(self.addType),
                                   'aprun'),
                         'aprun -n {} {} -d {}/{}/{} restart 2>&1 > {}/{}/{}/zzz'.format(
                             nProcs, hermesVer,
                             self.runDir, i, self.addType,
                             self.runDir, i, self.addType))
            replace_line('{}.pbs'.format(self.addType),
                         find_line('{}.pbs'.format(self.addType),
                                   'jobname') + 1,
                         '#PBS -N {}-{}'.format(self.addType, i))
            replace_line('{}.pbs'.format(self.addType),
                         find_line('{}.pbs'.format(self.addType),
                                   'walltime'),
                         '#PBS -l walltime={}'.format(tme))
            replace_line('{}.pbs'.format(self.addType),
                         find_line('{}.pbs'.format(self.addType),
                                   'PBS_O_WORKDIR') + 1,
                         'cd {}/{}/{}'.format(self.runDir, i, self.addType))

    def subJob(self, shortQ=False):
        for i in self.scanIDs:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            if shortQ is False:
                os.system('qsub {}.pbs'.format(self.addType))
            elif shortQ is True:
                os.system('qsub -q short {}.pbs'.format(self.addType))


class addNeutrals(addSim):
    def addVar(self, Nn=0.1, Pn=0.05):
        scanIds = self.scanIDs
        # for i in range(self.scanNum):
        for i in scanIds:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            addvar('Nn', Nn)
            addvar('Pn', Pn)


class addCurrents(addSim):
    pass


class addTurbulence(addSim):
    '''
    make sure to use
    export
    PYTHONPATH=/mnt/lustre/groups/phys-bout-2019/BOUT-dev/tools/pylib/:$PYTHONPATH
    '''
    def addTurb(self, oldDir, addType, MZ=64, param='Vort',
                pScale=1e-5, multiply=True):
        self.copyInpFiles(oldDir, addType)
        for i in self.scanIDs:
            os.chdir('{}/{}'.format(self.runDir, i))
            resizeZ(newNz=MZ, path=oldDir, output=addType)
            addnoise(path=addType, var=param, scale=pScale)


class testTurbulence(addSim):
    def redistributeProcs(self, ):
        print('hi')

    def addTurb(self, inpPath, turbPath, MZ=64, param='Vort', pScale=1e-5,
                multiply=True):
        os.chdir(inpPath)
        os.system('mkdir -p {}'.format(turbPath))
        resizeZ(newNz=MZ, path='./', output=turbPath)
        addnoise(path='.', var=param, scale=pScale, multiply=True)


if __name__ == "__main__":
    inpFile = 'BOUT.inp'
    gridFile = 'tcv_52068_64x64_profiles_1e19.nc'
    gridFile = 'tcv_63127_64x64_profiles_1.2e19.nc'

    pathOut = '/home/e281/e281/hm1234/hm1234/newTCV'
    pathIn = 'hgridscan'
    dateDir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    # dateDir = '_turbTest'

    # title = 'cfrac'
    # scanParams = [0.01, 0.02, 0.03, 0.05, 0.07]

    title = 'rfrac'
    scanParams = [0.9, 0.93, 0.96, 0.99]
    # scanParams = [0.95]

    nProcs = 160
    tme = '00:22:22'  # day-hr:min:sec
    # tme = '10:10:00'
    # hermesVer = '/users/hm1234/scratch/BOUT-test4/hermes-2/hermes-2'
    # hermesVer = '/mnt/lustre/groups/phys-bout-2019/hermes-2-next/hermes-2'
    # hermesVer = '/users/hm1234/scratch/BOUT25Jun19/hermes-2/hermes-2'
    hermesVer = '/users/hm1234/scratch/BOUT5Jul19/hermes-2/hermes-2'
    hermesVer = '/fs2/e281/e281/hm1234/BOUT17Sep19/hermes-2/hermes-2'

    title = 'grid'
    grids = ['tcv_63127_64x64_profiles_5.8e19.nc',
             'tcv_63127_64x64_profiles_6.5e19.nc',
             'tcv_63127_64x64_profiles_7.3e19.nc',
             'tcv_63127_64x64_profiles_8e19.nc',
             'tcv_63127_64x64_profiles_8.7e19.nc',
             'tcv_63127_64x64_profiles_9.3e19.nc']
             # 'tcv_63161_64x64_profiles_3e19.nc',
             # 'tcv_63161_64x64_profiles_3.25e19.nc',
             # 'tcv_63161_64x64_profiles_3.5e19.nc',
             # 'tcv_63161_64x64_profiles_3.75e19.nc',
             # 'tcv_63161_64x64_profiles_4.5e19.nc',
             # 'tcv_63161_64x64_profiles_5.5e19.nc']
    # gridFile = 'tcv_63127_64x64_profiles_1.6e19.nc'

    # gridSim = multiGridSim(pathOut, pathIn, dateDir, inpFile, grids, title)
    # gridSim.setup()
    # gridSim.modInp2('carbon_fraction', 0.04)
    # gridSim.modInp2('frecycle', 0.95)
    # gridSim.modInp2('NOUT', 444)
    # gridSim.modInp2('TIMESTEP', 222)
    # gridSim.modJob(nProcs, hermesVer, tme)
    # gridSim.subJob()

    # inpFile = 'BOUT.inp'
    # sim1 = startSim(pathOut, pathIn, dateDir, inpFile, gridFile,
    #                 scanParams, title)
    # sim1.setup()
    # sim1.modInp1('frecycle')
    # sim1.modInp2('carbon_fraction', 0.95)
    # sim1.modInp2('ion_viscosity', 'true')
    # sim1.modInp2('NOUT', 444)
    # sim1.modInp2('TIMESTEP', 222)
    # # sim1.modInp2('carbon_fraction', 0.04)
    # sim1.modJob(nProcs, hermesVer, tme)
    # sim1.subJob(shortQ=True)

    tme = '23:59:59'
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
    # runDir = '/fs2/e281/e281/hm1234/test/scan/grid-17-09-19_164731'
    # runDir = '/fs2/e281/e281/hm1234/newTCV/hgridscan/grid-24-09-19_112435'

    # addN = addNeutrals(runDir)
    # addN.copyInpFiles(addType='2-addN')
    # addN.copyRestartFiles(addType='2-addN')
    # # addN.copyNewInp(oldDir='/users/hm1234/scratch/newTCV',
    # #                 inpName='BOUT-2Dworks.inp')
    # addN.modFile('NOUT', 555)
    # addN.modFile('TIMESTEP', 150)
    # # addN.modFile('neutral_friction', 'true')
    # addN.modFile('type', 'mixed', lineNum=214)
    # addN.modJob(tme)
    # addN.addVar(Nn=0.04, Pn=0.02)
    # addN.subJob()

    # tme = '23:59:59'
    # addN = addNeutrals(runDir, scanIDs=[0, 5])
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

    # tme = '23:59:59'
    # old = '2-addN2'
    # new = '3-addC2'
    # addC = addCurrents(runDir, scanIDs=[0, 1, 2, 3, 4])
    # addC.copyInpFiles(old, new)
    # addC.copyRestartFiles(old, new)
    tme = '1-23:59:59'
    old = '3-addC2'
    new = '3-resC3'
    addC = addCurrents(runDir)
    addC.copyInpFiles(old, new)
    addC.copyRestartFiles(old, new)
    # addC.modFile('j_par', 'true')
    # addC.modFile('j_diamag', 'true')
    # addC.modFile('split_n0 ', 'false')
    # addC.modFile('split_n0_psi', 'false')
    # addC.modFile('NOUT', 600)
    # addC.modFile('TIMESTEP', 333)
    # addC.modJob(tme)
    # addC.subJob()

    tme = '23:59:59'
    old = '3-resC4'
    new = '3-resC5'
    resC = addCurrents(runDir)
    resC.copyInpFiles(old, new)
    resC.copyRestartFiles(old, new)
    resC.modFile('NOUT', 666)
    resC.modFile('TIMESTEP', 444)
    # addP.modFile('split_n0 ', 'true')
    # addP.modFile('split_n0_psi', 'true')
    # addP.modFile('poloidal_flows', 'true')
    # addP.modFile('NOUT', 555)
    # addP.modFile('TIMESTEP', 1)
    resC.modJob(tme)
    resC.subJob()

    # tme = '23:59:59'
    # old = '3-addC2'
    # new = '4-addP-ksp'
    # addP = addCurrents(runDir, scanIDs=[0])
    # addP.copyInpFiles(old, new)
    # addP.copyRestartFiles(old, new)
    # addP.modFile('split_n0 ', 'true')
    # addP.modFile('split_n0_psi', 'true')
    # addP.modFile('poloidal_flows', 'true')
    # addP.modFile('NOUT', 555)
    # addP.modFile('TIMESTEP', 1)
    # addP.modJob(tme)
    # addP.subJob()
    addC.modFile('NOUT', 600)
    addC.modFile('TIMESTEP', 333)
    addC.modJob(tme)
    addC.subJob()

    # tme = '23:59:59'
    # old = '2-addN2'
    # new = '3-addC2'
    # addC = addCurrents(runDir, scanIDs=[5])
    # addC.copyInpFiles(old, new)
    # addC.copyRestartFiles(old, new)
    # addC.modFile('j_par', 'true')
    # addC.modFile('j_diamag', 'true')
    # addC.modFile('split_n0 ', 'false')
    # addC.modFile('split_n0_psi', 'false')
    # addC.modFile('NOUT', 600)
    # addC.modFile('TIMESTEP', 333)
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
    # addT.copyInpFiles('2-addC', '4-addT')
    # addT.addTurb('2-addC', '4-addT')
    # addT.modJob(tme)
    # addT.modFile('MZ', 64)
    # addT.modFile('NOUT', 333)
    # addT.modFile('TIMESTEP', 0.04)
    # addT.modFile('output_ddt', 'true')
    # addT.modFile('verbose', 'true')
    # addT.subJob()
