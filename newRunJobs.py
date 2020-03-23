import numpy as np
import os
import subprocess
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

def get_last_line(file_name):
    lines = open(file_name, 'r').readlines()
    return lines[-1].strip()

def find_line(filename, lookup):
    # finds line in a file
    line_num = None
    with open(filename) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                line_num = num
    if line_num is None:
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


class addToLog(logSim):
    '''
    For adding to the logfile when restarting sims
    '''
    def __init__(self, logFile):
        self.fileName = logFile

    def __call__(self, message):
        super().__call__(message)


class baseSim:
    def __init__(self, cluster, pathOut, pathIn, dateDir, gridFile, scanParams,
                 hermesVer,runScript, inpFile='BOUT.inp', title='sim'):
        os.chdir(pathOut)
        self.pathOut = pathOut
        self.pathIn = pathIn
        self.inpFile = inpFile
        self.gridFile = gridFile
        self.runDir = '{}/{}/{}-{}'.format(pathOut, pathIn, title, dateDir)
        self.scanParams = scanParams
        self.title = title
        self.hermesVer = hermesVer
        self.cluster = cluster
        self.runScript = runScript
        if self.scanParams is not None:
            self.scanNum = len(scanParams)
        else:
            self.scanNum = 1
        os.system('mkdir {} -p'.format(self.runDir))
        os.chdir('{}'.format(self.runDir))

    def getHermesGit(self):
        currDir = subprocess.run(['pwd', '-P'],
                                 capture_output=True,
                                 text=True).stdout.strip()
        os.chdir(self.hermesVer[:-8])
        hermesGitID = subprocess.run('git describe --always'.split(),
                                     capture_output=True,
                                     text=True).stdout.strip()
        hermesURL = subprocess.run('git remote get-url origin'.split(),
                                     capture_output=True,
                                     text=True).stdout.strip()
        BOUTgitID = get_last_line('BOUT_commit')
        os.chdir(currDir)
        return hermesURL, hermesGitID, BOUTgitID

    def setup(self):
        self.log = logSim(self.runDir, 'log.txt')
        self.log('cluster: {}'.format(self.cluster))
        self.log('title: {}'.format(self.title))
        self.log('inpFile: {}'.format(self.inpFile))
        self.log('gridFile: {}'.format(str(self.gridFile)))
        self.log('scanParams: {}'.format(str(self.scanParams)))
        self.log('BOUT_commit: {}'.format(
            self.getHermesGit()[2]))
        self.log('hermesInfo: {} - {}'.format(
            self.getHermesGit()[0], self.getHermesGit()[1]))
        
        
        for i in range(self.scanNum):
            os.mkdir(str(i))
            os.system('cp {}/{} {}/BOUT.inp'.format(
                self.pathOut, self.inpFile, i))
            os.system('cp {}/{} {}/'.format(
                self.pathOut, self.runScript, i))
            if type(self.gridFile) == str:
                if self.cluster == 'viking':
                    cpGridCmd = 'cp /users/hm1234/scratch/gridfiles/{} {}'.format(
                        self.gridFile, i)
                elif self.cluster == 'archer':
                    cpGridCmd = 'cp /work/e281/e281/hm1234/gridfiles/{} {}'.format(
                        self.gridFile, i)
                elif self.cluster == 'marconi':
                    cpGridCmd = 'cp /marconi_work/FUA34_SOLBOUT4/hmuhamme/gridfiles/{} {}'.format(
                        self.gridFile, i)
                os.system(cpGridCmd)
        self.inpFile = 'BOUT.inp'
        if self.gridFile is not None:
            self.modInp(param='grid', value=self.gridFile)

    def modInp(self, param, value=None, lineNum=None):
        if lineNum is None:
            lineNum = find_line('{}/0/{}'.format(
                self.runDir, self.inpFile),
                                param)
        if value is None:
            for i, j in enumerate(self.scanParams):
                os.chdir('{}/{}'.format(self.runDir, i))
                replace_line('{}'.format(self.inpFile),
                             lineNum,
                             '{} = {}'.format(param, j))
        else:
            for i in range(self.scanNum):
                os.chdir('{}/{}'.format(self.runDir, i))
                replace_line('{}'.format(self.inpFile),
                             lineNum,
                             '{} = {}'.format(param, value))

    def modJob(self, nProcs, tme, optNodes=True):
        if self.cluster == 'viking':
            self.vikingModJob(nProcs, tme, optNodes)
        elif self.cluster == 'archer':
            self.archerModJob(nProcs, tme, optNodes)
        elif self.cluster == 'marconi':
            msg = 'need to figure it oot'


    def archerModJob(self, nProcs, tme, optNodes=True):
        if optNodes is True:
            nodes = int(np.ceil(nProcs/24))
            for i in range(self.scanNum):
                os.chdir('{}/{}'.format(self.runDir, i))
                replace_line(self.runScript,
                             find_line(self.runScript,
                                       'select'),
                             '#PBS -l select={}'.format(nodes))
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   'aprun'),
                         'aprun -n {} {} -d {}/{} 2>&1 {}/{}/zzz'.format(
                             nProcs, self.hermesVer,
                             self.runDir, i,
                             self.runDir, i))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   'jobname') + 1,
                         '#PBS -N {}-{}'.format(self.title, i))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   'walltime'),
                         '#PBS -l walltime={}'.format(tme))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   'PBS_O_WORKDIR') + 1,
                         'cd {}/{}'.format(self.runDir, i))
            
    def vikingModJob(self, nProcs, tme, optNodes=True):
        if optNodes is True:
            nodes = int(np.ceil(nProcs/40))
            for i in range(self.scanNum):
                os.chdir('{}/{}'.format(self.runDir, i))
                replace_line(self.runScript,
                             find_line(self.runScript,
                                       '--nodes'),
                             '#SBATCH --nodes={}'.format(nodes))
        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   '--ntasks'),
                         '#SBATCH --ntasks={}'.format(nProcs))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   'mpiexec'),
                         'mpiexec -n {} {} -d {}/{}'.format(nProcs,
                                                            self.hermesVer,
                                                            self.runDir,
                                                            i))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   '--job-name'),
                         '#SBATCH --job-name={}-{}'.format(self.title, i))
            replace_line(self.runScript,
                         find_line(self.runScript,
                                   '--time'),
                         '#SBATCH --time={}'.format(tme))

    def subJob(self, shortQ=False):
        if shortQ is False:
            queue = ''
        elif shortQ is True:
            queue = '-q short'

        if self.cluster == 'viking':
            cmd = 'sbatch {} {}'.format(queue, self.runScript)
        elif self.cluster == 'archer':
            cmd = 'qsub {} {}'.format(queue, self.runScript)
        elif self.cluster == 'marconi':
            cmd = 'figure it oot'

        for i in range(self.scanNum):
            os.chdir('{}/{}'.format(self.runDir, i))
            cmdInfo = subprocess.run(cmd.split(),
                                     capture_output=True,
                                     text=True)
            if cmdInfo.stderr == '':
                self.log('jobID: {}'.format(cmdInfo.stdout.strip()))
            else:
                sys.exit(cmdInfo.stderr)



class multiGridSim(baseSim):
    def __init__(self, cluster, pathOut, pathIn, dateDir, scanParams,
                 hermesVer, inpFile='BOUT.inp', title='sim'):
        super().__init__(cluster, pathOut, pathIn, dateDir, 'blah',
                         scanParams, hermesVer, inpFile, title)
        self.gridFile = self.scanParams

    def setup(self):
        super().setup()
        for i in range(self.scanNum):
            os.system('cp /users/hm1234/scratch/gridfiles/{} {}/{}'.format(
                self.scanParams[i], self.runDir, i))
        self.modInp('grid')


class slabSim(baseSim):
    pass


class addSim(baseSim):
    def __init__(self, oldDir, addType='restart', scanIDs=[], logFile='log.txt'):
        self.addType = addType
        self.runDir = self.extract_rundir(oldDir)
        os.chdir(self.runDir)
        self.scanParams = read_line(logFile, 'scanParams')
        if len(scanIDs) == 0:
            self.scanIDs = list(np.arange(len(self.scanParams)))
        # self.scanNum = len(self.scanParams)
        self.title = read_line(logFile, 'title')
        self.inpFile = 'BOUT.inp'
        self.gridFile = read_line(logFile, 'gridFile')
        self.hermesVer = read_line(logFile, 'hermesVer')
        self.nProcs = read_line(logFile, 'nProcs')
        log = addToLog('{}/{}'.format(self.runDir, logFile))
        log('sim modified at: {}'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")))

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

    def extract_rundir(self, runDir):
        stringSplit = list(np.roll(runDir.split('/'), -1))
        for i, j in enumerate(stringSplit):
            temp = None
            try:
                temp = type(eval(j)) 
            except(NameError, SyntaxError):
                pass
            if temp is int:
                stringID = i
                break
        newString = '/'
        for i in stringSplit[0:stringID]:
            newString += i + '/'
        return newString
        
        
if __name__=="__main__":
    ###################################################
    ##################  Archer Jobs ###################
    ###################################################
    # inpFile = 'BOUT.inp'
    # pathOut = '/home/e281/e281/hm1234/hm1234/TCV2020'
    # pathIn = 'test2'
    # dateDir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    # title = 'cfrac'
    # scanParams = [0.02, 0.04, 0.06, 0.08]
    # nProcs = 16
    # tme = '00:19:00'
    # hermesVer = '/home/e281/e281/hm1234/hm1234/BOUTtest/hermes-2/hermes-2'
    # gridFile = 'newtcv2_63161_64x64_profiles_5e19.nc'

    # archerSim = baseSim(cluster = 'archer',
    #                     pathOut = pathOut,
    #                     pathIn = pathIn,
    #                     dateDir = dateDir,
    #                     gridFile = gridFile,
    #                     scanParams = scanParams,
    #                     hermesVer = hermesVer,
    #                     runScript = 'job.pbs',
    #                     inpFile = 'BOUT.inp',
    #                     title = 'newRunJobTest')

    # archerSim.setup()
    # archerSim.modInp('carbon_fraction')
    # archerSim.modInp('NOUT', 444)
    # archerSim.modInp('TIMESTEP', '222')
    # archerSim.modJob(nProcs, tme, optNodes=True)
    # archerSim.subJob(shortQ=True)
