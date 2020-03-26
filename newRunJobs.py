import numpy as np
import os
import subprocess
import sys
import datetime
import time
from boutdata.restart import addvar
from boutdata.restart import addnoise
from boutdata.restart import create
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


class LogSim:
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


class AddToLog(LogSim):
    '''
    For adding to the logfile when restarting sims
    '''
    def __init__(self, logFile):
        self.fileName = logFile

    def __call__(self, message):
        super().__call__(message)


class BaseSim:
    def __init__(self, cluster, path_out, path_in, date_dir, grid_file, scan_params,
                 hermes_ver, run_script, inp_file='BOUT.inp', title='sim'):
        os.chdir(path_out)
        self.path_out = path_out
        self.path_in = path_in
        self.inp_file = inp_file
        self.grid_file = grid_file
        self.run_dir = '{}/{}/{}-{}'.format(path_out, path_in, title, date_dir)
        self.scan_params = scan_params
        self.title = title
        self.add_type = ''
        self.hermes_ver = hermes_ver
        self.cluster = cluster
        self.run_script = run_script
        if self.scan_params is not None:
            self.scan_num = len(scan_params)
        else:
            self.scan_num = 1
        self.scan_IDs = list(range(self.scan_num))
        os.system('mkdir {} -p'.format(self.run_dir))
        os.chdir('{}'.format(self.run_dir))

    def get_hermes_git(self):
        curr_dir = subprocess.run(['pwd', '-P'],
                                 capture_output=True,
                                 text=True).stdout.strip()
        os.chdir(self.hermes_ver[:-8])
        hermes_git_ID = subprocess.run('git rev-parse --short HEAD'.split(),
                                     capture_output=True,
                                     text=True).stdout.strip()
        hermes_URL = subprocess.run('git remote get-url origin'.split(),
                                     capture_output=True,
                                     text=True).stdout.strip()
        BOUT_git_ID = get_last_line('BOUT_commit')
        os.chdir(curr_dir)
        return hermes_URL, hermes_git_ID, BOUT_git_ID

    def setup(self):
        self.log = LogSim(self.run_dir, 'log.txt')
        self.log('cluster: {}'.format(self.cluster))
        self.log('title: {}'.format(self.title))
        self.log('path_out: {}'.format(self.path_out))
        self.log('inp_file: {}'.format(self.inp_file))
        self.log('run_script: {}'.format(self.run_script))
        self.log('grid_file: {}'.format(str(self.grid_file)))
        self.log('scan_params: {}'.format(str(self.scan_params)))
        self.log('BOUT_commit: {}'.format(
            self.get_hermes_git()[2]))
        self.log('hermes_info: {} - {}'.format(
            self.get_hermes_git()[0], self.get_hermes_git()[1]))
        
        
        for i in self.scan_IDs:
            os.mkdir(str(i))
            os.system('cp {}/{} {}/BOUT.inp'.format(
                self.path_out, self.inp_file, i))
            os.system('cp {}/{} {}/'.format(
                self.path_out, self.run_script, i))
            if type(self.grid_file) == str:
                if self.cluster == 'viking':
                    cp_grid_cmd = 'cp /users/hm1234/scratch/gridfiles/{} {}'.format(
                        self.grid_file, i)
                elif self.cluster == 'archer':
                    cp_grid_cmd = 'cp /work/e281/e281/hm1234/gridfiles/{} {}'.format(
                        self.grid_file, i)
                elif self.cluster == 'marconi':
                    cp_grid_cmd = 'cp /marconi_work/FUA34_SOLBOUT4/hmuhamme/gridfiles/{} {}'.format(
                        self.grid_file, i)
                os.system(cp_grid_cmd)
        self.inp_file = 'BOUT.inp'
        if self.grid_file is not None:
            self.mod_inp(param='grid', value=self.grid_file)

    def mod_inp(self, param, value=None, line_num=None):
        if line_num is None:
            line_num = find_line('{}/0/{}'.format(
                self.run_dir, self.inp_file),
                                 param)
        scan_params = []
        for i in self.scan_IDs:
            scan_params.append(self.scan_params[i])
        if value is None:
            for i, j in enumerate(scan_params):
                os.chdir('{}/{}/{}'.format(
                    self.run_dir, i, self.add_type))
                replace_line('{}'.format(self.inp_file),
                             line_num,
                             '{} = {}'.format(param, j))
            self.log('modified: {}, to {}'.format(param, scan_params))
        else:
            for i in self.scan_IDs:
                os.chdir('{}/{}/{}'.format(
                    self.run_dir, i, self.add_type))
                replace_line('{}'.format(self.inp_file),
                             line_num,
                             '{} = {}'.format(param, value))
            self.log('modified: {}, to {}'.format(param, value))

    def mod_job(self, n_procs, tme, opt_nodes=True):
        if self.cluster == 'viking':
            self.viking_mod_job(n_procs, tme, opt_nodes)
        elif self.cluster == 'archer':
            self.archer_mod_job(n_procs, tme, opt_nodes)
        elif self.cluster == 'marconi':
            msg = 'need to figure it oot'


    def archer_mod_job(self, n_procs, tme, opt_nodes=True):
        if opt_nodes is True:
            nodes = int(np.ceil(n_procs/24))
            for i in self.scan_IDs:
                os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
                replace_line(self.run_script,
                             find_line(self.run_script,
                                       'select'),
                             '#PBS -l select={}'.format(nodes))
        for i in self.scan_IDs:
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
            replace_line(self.run_script,
                         find_line(self.run_script,
                                   'jobname') + 1,
                         '#PBS -N {}-{}'.format(self.title, i))
            replace_line(self.run_script,
                         find_line(self.run_script,
                                   'walltime'),
                         '#PBS -l walltime={}'.format(tme))
            replace_line(self.run_script,
                         find_line(self.run_script,
                                   'PBS_O_WORKDIR') + 1,
                         'cd {}/{}'.format(self.run_dir, i))
        if self.add_type == '':
            run_command = 'aprun -n {} {} -d {}/{} 2>&1 {}/{}/zzz'.format(
                             n_procs, self.hermes_ver,
                             self.run_dir, i,
                             self.run_dir, i)
        else:
            run_command = 'aprun -n {} {} -d {}/{}/{} restart 2>&1 > {}/{}/{}/zzz'.format(
                             n_procs, self.hermes_ver,
                             self.run_dir, i, self.add_type,
                             self.run_dir, i, self.add_type)
        for i in self.scan_IDs:
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
            replace_line(self.run_script,
                         find_line(self.run_script,
                                   'aprun'),
                         run_command)
            
    def viking_mod_job(self, n_procs, tme, opt_nodes=True):
        if opt_nodes is True:
            nodes = int(np.ceil(n_procs/40))
            for i in self.scan_IDs:
                os.chdir('{}/{}'.format(self.run_dir, i))
                replace_line(self.run_script,
                             find_line(self.run_script,
                                       '--nodes'),
                             '#SBATCH --nodes={}'.format(nodes))
        for i in self.scan_IDs:
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
            replace_line(self.run_script,
                         find_line(self.run_script,
                                   '--ntasks'),
                         '#SBATCH --ntasks={}'.format(n_procs))
            replace_line(self.run_script,
                         find_line(self.run_script,
                                   '--job-name'),
                         '#SBATCH --job-name={}-{}'.format(self.title, i))
            replace_line(self.run_script,
                         find_line(self.run_script,
                                   '--time'),
                         '#SBATCH --time={}'.format(tme))
        if self.add_type == '':
            run_command = 'mpiexec -n {} {} -d {}/{}'.format(
                             n_procs, self.hermes_ver, self.run_dir, i)
        else:
            run_command = 'mpiexec -n {} {} -d {}/{}/{} restart'.format(
                             n_procs, self.hermes_ver, self.run_dir, i, self.add_type)
        for i in self.scan_IDs:
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
            replace_line(self.run_script,
                         find_line(self.run_script, 'mpiexec'),
                         run_command)

    def sub_job(self, shortQ=False):
        if shortQ is False:
            queue = ''
        elif shortQ is True:
            queue = '-q short'

        if self.cluster == 'viking':
            cmd = 'sbatch {} {}'.format(queue, self.run_script)
        elif self.cluster == 'archer':
            cmd = 'qsub {} {}'.format(queue, self.run_script)
        elif self.cluster == 'marconi':
            cmd = 'figure it oot'

        for i in range(self.scan_num):
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
            cmdInfo = subprocess.run(cmd.split(),
                                     capture_output=True,
                                     text=True)
            if cmdInfo.stderr == '':
                self.log('jobID: {}'.format(cmdInfo.stdout.strip()))
            else:
                sys.exit(cmdInfo.stderr)


class StartSim(BaseSim):
    pass


class MultiGridSim(BaseSim):
    def __init__(self, cluster, path_out, path_in, date_dir, scan_params,
                 hermes_ver, run_script, inp_file='BOUT.inp', title='sim'):
        super().__init__(cluster, path_out, path_in, date_dir, 'blah',
                         scan_params, hermes_ver, run_script, inp_file, title)
        self.grid_file = self.scan_params

    def setup(self):
        super().setup()
        for i in range(self.scan_num):
            os.system('cp /users/hm1234/scratch/gridfiles/{} {}/{}'.format(
                self.scan_params[i], self.run_dir, i))
        self.mod_inp('grid')


class SlabSim(BaseSim):
    pass


class AddSim(BaseSim):
    def __init__(self, old_dir, add_type='restart', scan_IDs=[], logFile='log.txt', **kwargs):
        try:
            self.t = kwargs['t']
        except(KeyError):
            self.t = None
        self.add_type = add_type
        self.old_dir = old_dir
        self.run_dir = self.extract_rundir(old_dir)[0]
        os.chdir(self.run_dir)
        self.scan_params = read_line(logFile, 'scan_params')
        if len(scan_IDs) == 0:
            self.scan_IDs = list(range(len(self.scan_params)))
        # self.scan_num = len(self.scan_params)
        self.title = read_line(logFile, 'title')
        self.inp_file = 'BOUT.inp'
        self.run_script = read_line(logFile, 'run_script')
        self.grid_file = read_line(logFile, 'grid_file')
        self.hermes_ver = read_line(logFile, 'hermes_ver')
        self.n_procs = read_line(logFile, 'n_procs')
        self.path_out = read_line(logFile, 'path_out')
        log = AddToLog('{}/{}'.format(self.run_dir, logFile))
        log('sim modified at: {}'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")))
        log('new_sim_type: {}'.format(add_type))
        for i in self.scan_IDs:
            os.chdir('{}/{}'.format(self.run_dir, i))
            os.system('mkdir -p {}'.format(add_type))

    def setup(self, )

    def extract_rundir(self, run_dir):
        string_split = list(np.roll(run_dir.split('/'), -1))
        for i, j in enumerate(string_split):
            temp = None
            try:
                temp = type(eval(j))
            except(NameError, SyntaxError):
                pass
            if temp is int:
                stringID = i
                break
        new_string = '/'
        run_dir = new_string + new_string.join(string_split[:stringID])
        old_type = new_string + new_string.join(string_split[stringID+1:])
        if old_type in [new_string, 2*new_string]:
            old_type = ''
        return run_dir, old_type

    def copy_new_inp(self, inpName):
        for i in self.scan_IDs:
            os.system('cp {}/{} {}/{}/{}/BOUT.inp'.format(
                self.path_out, inpName, self.run_dir, i, self.add_type))

    def copy_inp_files(self, add_type=self.add_type):
        for i in self.scan_IDs:
            os.chdir('{}/{}'.format(self.run_dir, i))
            if type(self.grid_file) == list:
                os.system('cp {} {}'.format(self.grid_file[i], add_type))
            elif self.grid_file is None:
                pass
            else:
                os.system('cp {} {}'.format(self.grid_file, add_type))
            os.system('cp {} {}/{}'.format(self.run_script, add_type,
                                           self.run_script))
            cmd = 'cp {}/{} {}'.format(self.old_dir, self.inp_file, add_type)
            os.system(cmd)

    def copy_restart_files(self, old_dir=None, add_type='restart', t=None):
        if t is None:
            if old_dir is None:
                cmd = 'cp BOUT.restart.* {}'.format(add_type)
            else:
                cmd = 'cp {}/BOUT.restart.* {}'.format(old_dir, add_type)
            for i in self.scan_IDs:
                os.chdir('{}/{}'.format(self.run_dir, i))
                os.system(cmd)
        else:
            for i in self.scan_IDs:
                if old_dir is None:
                    os.chdir('{}/{}'.format(self.run_dir, i))
                else:
                    os.chdir('{}/{}/{}'.format(
                        self.run_dir, i, old_dir))
                create(final=t, path="./", output='{}/{}/{}'.format(
                    self.run_dir, i, add_type))
    

        
if __name__=="__main__":
    ###################################################
    ##################  Archer Jobs ###################
    ###################################################
    # inp_file = 'BOUT.inp'
    # path_out = '/home/e281/e281/hm1234/hm1234/TCV2020'
    # path_in = 'test2'
    # date_dir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    # title = 'cfrac'
    # scan_params = [0.02, 0.04, 0.06, 0.08]
    # n_procs = 16
    # tme = '00:19:00'
    # hermes_ver = '/home/e281/e281/hm1234/hm1234/BOUTtest/hermes-2/hermes-2'
    # grid_file = 'newtcv2_63161_64x64_profiles_5e19.nc'

    # archerSim = StartSim(cluster = 'archer',
    #                     path_out = path_out,
    #                     path_in = path_in,
    #                     date_dir = date_dir,
    #                     grid_file = grid_file,
    #                     scan_params = scan_params,
    #                     hermes_ver = hermes_ver,
    #                     run_script = 'job.pbs',
    #                     inp_file = 'BOUT.inp',
    #                     title = 'newRunJobTest')

    # archerSim.setup()
    # archerSim.mod_inp('carbon_fraction')
    # archerSim.mod_inp('NOUT', 444)
    # archerSim.mod_inp('TIMESTEP', '222')
    # archerSim.mod_job(n_procs, tme, opt_nodes=True)
    # archerSim.sub_job(shortQ=True)
    a = 1
