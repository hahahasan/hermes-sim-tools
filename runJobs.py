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

    def modInp1(self, param, ambiguous=False, lineNum=None):
        if ambiguous is False:
            lineNum = find_line('{}/{}'.format(self.pathOut, self.inpFile),
                                param)
        else:
            lineNum = lineNum
        for i, j in enumerate(self.scanParams):
            os.chdir('{}/{}'.format(self.runDir, i))
            replace_line('{}'.format(self.inpFile),
                         lineNum,
                         '{} = {}'.format(param, j))

    def modInp2(self, param, value, ambiguous=False, lineNum=None):
        self.log('Modified {} to: {}'.format(param, value))
        if ambiguous is False:
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
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            os.system('cp {}/test.job {}.job'.format(self.pathOut, self.title))
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
            if type(self.gridFile) == str:
                os.system('cp {}/{} {}'.format(self.pathOut, self.gridFile, i))
        self.inpFile = 'BOUT.inp'
        self.modInp2('grid', self.gridFile)

    def subJob(self, shortQ=False):
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            if shortQ is False:
                os.system('sbatch {}.job'.format(self.title))
            elif shortQ is True:
                os.system('sbatch -q short {}.job'.format(self.title))


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
    def __init__(self, runDir, logFile='log.txt'):
        self.runDir = runDir
        os.chdir(runDir)
        self.scanParams = read_line(logFile, 'scanParams')
        self.scanNum = len(read_line(logFile, 'scanParams'))
        self.title = read_line(logFile, 'title')
        # self.inpFile = read_line(logFile, 'inpFile')
        self.inpFile = 'BOUT.inp'
        self.gridFile = read_line(logFile, 'gridFile')
        self.hermesVer = read_line(logFile, 'hermesVer')
        self.nProcs = read_line(logFile, 'nProcs')

    def copyFiles(self, addType='restart'):
        self.addType = addType
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            os.mkdir(addType)
            os.system('cp {} {}'.format(self.inpFile, addType))
            os.system('cp BOUT.restart.* {}'.format(addType))
            # if len(self.gridFile) > 1:
            if type(self.gridFile) != str:
                os.system('cp {} {}'.format(self.gridFile[i], addType))
            else:
                os.system('cp {} {}'.format(self.gridFile, addType))

            os.system('cp *.job {}/{}.job'.format(addType, addType))

    def copyFiles2(self, oldDir, addType='restart'):
        self.addType = addType
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            os.mkdir(addType)
            os.system('cp {}/{} {}'.format(oldDir, self.inpFile, addType))
            os.system('cp {}/BOUT.restart.* {}'.format(oldDir, addType))
            if type(self.gridFile) != str:
                os.system('cp {} {}'.format(self.gridFile[i], addType))
            else:
                os.system('cp {} {}'.format(self.gridFile, addType))

            os.system('cp *.job {}/{}.job'.format(addType, addType))

    def modFile(self, param, value, ambiguous=False, lineNum=None):
        if ambiguous is False:
            lineNum = find_line('{}/0/{}'.format(self.runDir, self.inpFile),
                                param)
        else:
            lineNum = lineNum
        for i in range(self.scanNum):
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            replace_line('{}'.format(self.inpFile),
                         lineNum,
                         '{} = {}'.format(param, value))

    def modJob(self, tme):
        for i in range(self.scanNum):
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--ntasks'),
                         '#SBATCH --ntasks={}'.format(self.nProcs))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   'mpiexec'),
                         'mpiexec -n {} {} -d {}/{}/{} restart'.format(
                             self.nProcs, self.hermesVer, self.runDir,
                             i, self.addType))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--job-name'),
                         '#SBATCH --job-name={}-{}'.format(self.addType, i))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--time'),
                         '#SBATCH --time={}'.format(tme))

    def subJob(self):
        for i in range(self.scanNum):
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            os.system('sbatch {}.job'.format(self.addType))


class addNeutrals(addSim):
    def addVar(self, Nn=0.1, Pn=0.05):
        for i in range(self.scanNum):
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
    def __init__(self, runDir, logFile='log.txt', scanParams=[]):
        super().__init__(runDir, logFile)
        if len(scanParams) == 0:
            self.paramNums = list(np.arange(len(self.scanParams)))
        else:
            self.paramNums = scanParams

    def copyNonRestart(self, oldDir, addType):
        self.addType = addType
        for i in self.paramNums:
            os.chdir('{}/{}'.format(self.runDir, i))
            os.system('mkdir -p {}'.format(addType))
            os.system('cp {}/{} {}'.format(oldDir, self.inpFile, addType))
            if type(self.gridFile) != str:
                os.system('cp {} {}'.format(self.gridFile[i], addType))
            else:
                os.system('cp {} {}'.format(self.gridFile, addType))
            os.system('cp *.job {}/{}.job'.format(addType, addType))

    def copyNewInp(self, oldDir, inpName):
        for i in self.paramNums:
            os.system('cp {}/{} {}/{}/{}/BOUT.inp'.format(
                oldDir, inpName, self.runDir, i, self.addType))

    def redistributeProcs(self, oldDir, addType, npes):
        self.copyNonRestart(oldDir, addType)
        for i in self.paramNums:
            os.chdir('{}/{}'.format(self.runDir, i))
            redistribute(npes=npes, path=oldDir, output=addType)
        self.nProcs = npes

    def addTurb(self, oldDir, addType, MZ=64, param='Vort',
                pScale=1e-5, multiply=True):
        self.copyNonRestart(oldDir, addType)
        for i in self.paramNums:
            os.chdir('{}/{}'.format(self.runDir, i))
            resizeZ(newNz=MZ, path=oldDir, output=addType)
            addnoise(path=addType, var=param, scale=pScale)

    def modJob(self, tme):
        for i in self.paramNums:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--ntasks'),
                         '#SBATCH --ntasks={}'.format(self.nProcs))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   'mpiexec'),
                         'mpiexec -n {} {} -d {}/{}/{} restart'.format(
                             self.nProcs, self.hermesVer, self.runDir,
                             i, self.addType))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--job-name'),
                         '#SBATCH --job-name={}-{}'.format(self.addType, i))
            replace_line('{}.job'.format(self.addType),
                         find_line('{}.job'.format(self.addType),
                                   '--time'),
                         '#SBATCH --time={}'.format(tme))

    def modFile(self, param, value, lineNum=None):
        if lineNum is not None:
            lineNum = find_line('{}/0/{}'.format(self.runDir, self.inpFile),
                                param)
        else:
            lineNum = lineNum
        for i in self.paramNums:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            replace_line('{}'.format(self.inpFile),
                         lineNum,
                         '{} = {}'.format(param, value))

    def subJob(self):
        for i in self.paramNums:
            os.chdir('{}/{}/{}'.format(self.runDir, i, self.addType))
            os.system('sbatch {}.job'.format(self.addType))


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

    pathOut = '/users/hm1234/scratch/newTCV'
    pathIn = 'gridscan'
    dateDir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")

    # title = 'cfrac'
    # scanParams = [0.01, 0.02, 0.03, 0.05, 0.07]

    title = 'grid'
    # scanParams = [0.9, 0.93, 0.96, 0.99]

    nProcs = 160
    tme = '23:55:55'  # hr:min:sec
    # tme = '10:10:00'
    # hermesVer = '/users/hm1234/scratch/BOUT-test4/hermes-2/hermes-2'
    # hermesVer = '/mnt/lustre/groups/phys-bout-2019/hermes-2-next/hermes-2'
    # hermesVer = '/users/hm1234/scratch/BOUT25Jun19/hermes-2/hermes-2'
    hermesVer = '/users/hm1234/scratch/BOUT5Jul19/hermes-2/hermes-2'

    grids = ['tcv_63127_64x64_profiles_0.8e19.nc',
             'tcv_63127_64x64_profiles_1.2e19.nc',
             'tcv_63127_64x64_profiles_1.6e19.nc',
             'tcv_63127_64x64_profiles_2.0e19.nc']

    # gridSim = multiGridSim(pathOut, pathIn, dateDir, inpFile, grids, title)
    # gridSim.setup()
    # gridSim.modInp2('carbon_fraction', 0.04)
    # gridSim.modInp2('frecycle', 0.95)
    # gridSim.modInp2('NOUT', 444)
    # gridSim.modInp2('TIMESTEP', 222)
    # gridSim.modJob(nProcs, hermesVer, tme)
    # gridSim.subJob()

    # sim1 = startSim(pathOut, pathIn, dateDir, inpFile, gridFile,
    #                 scanParams, title)
    # sim1.setup()
    # sim1.modInp1('frecycle')
    # sim1.modInp2('NOUT', 444)
    # sim1.modInp2('TIMESTEP', 222)
    # sim1.modInp2('carbon_fraction', 0.04)
    # sim1.modJob(nProcs, hermesVer, tme)
    # sim1.subJob()

    # runDir = '/users/hm1234/scratch/TCV/NeScan2/NeScan-03-06-19_171145'
    # runDir = '/users/hm1234/scratch/TCV/NeScan2/frecycle-05-06-19_145457'
    # runDir = '/users/hm1234/scratch/TCV/longtime/cfrac-10-06-19_175728'
    # runDir = '/users/hm1234/scratch/TCV/longtime/rfrac-19-06-19_102728'
    # runDir = '/users/hm1234/scratch/TCV2/gridscan/grid-20-06-19_135947'
    runDir = '/users/hm1234/scratch/newTCV/gridscan/grid-01-07-19_185351'
    runDir = '/users/hm1234/scratch/newTCV/gridscan/test'

    # addN = addNeutrals(runDir)
    # addN.copyFiles('2-addN')
    # addN.modFile('NOUT', 555)
    # addN.modFile('TIMESTEP', 150)
    # addN.modFile('type', 'mixed', ambiguous=True, lineNum=214)
    # addN.modJob(tme)
    # addN.addVar(Nn=0.1, Pn=0.05)
    # addN.subJob()

    # tme = '23:55:55'
    # addC = addCurrents(runDir)
    # addC.copyFiles2('2-addN', '3-addC')
    # addC.modFile('j_par', 'true')
    # addC.modFile('j_diamag', 'true')
    # addC.modFile('TIMESTEP', 333)
    # addC.modJob(tme)
    # addC.subJob()

    tme = '23:55:55'
    # addT = testTurbulence(runDir)
    # addT.copyFiles2('3-addC', '4-addT')
    # addT.modJob(tme)
    addT = addTurbulence(runDir, scanParams=[0])
    addT.hermesVer = hermesVer
    # addT.redistributeProcs('3-addC', '4-redistribute', 480)
    # addT.addTurb('4-redistribute', '5-addT')
    addT.copyFiles2('3-addC', 'test-turb')
    addT.copyNewInp(runDir, 'BOUT2.inp')
    addT.modJob(tme)
    # addT.subJob()
