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
        self.log_file = open(self.fileName, 'w+')
        self.log_file.close()

    def __call__(self, message):
        self.log_file = open(self.fileName, 'a+')
        self.log_file.write('{}\r\n'.format(message))
        self.log_file.close()


class AddToLog(LogSim):
    '''
    For adding to the logfile when restarting sims
    '''
    def __init__(self, log_file):
        self.fileName = log_file

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
        self.log('hermes_ver: {}'.format(self.hermes_ver))
        self.log('n_procs: {}'.format(self.n_procs))
        
        for i in self.scan_IDs:
            os.system('mkdir -p {}'.format(i))
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
            if type(self.grid_file) is list:
                value = None
            else:
                value = self.grid_file
            self.mod_inp(param='grid', value=value)

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
        for i in self.scan_IDs:
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
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
            replace_line(self.run_script,
                         find_line(self.run_script, 'aprun'),
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
        for i in self.scan_IDs:
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
            if self.add_type == '':
                run_command = 'mpiexec -n {} {} -d {}/{}'.format(
                    n_procs, self.hermes_ver, self.run_dir, i)
            else:
                run_command = 'mpiexec -n {} {} -d {}/{}/{} restart'.format(
                    n_procs, self.hermes_ver, self.run_dir, i, self.add_type)
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

        for i in self.scan_IDs:
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
        for i in self.scan_IDs:
            if self.cluster == 'viking':
                cp_grid_cmd = 'cp /users/hm1234/scratch/gridfiles/{} {}/{}'.format(
                    self.grid_file[i], self.run_dir, i)
            elif self.cluster == 'archer':
                cp_grid_cmd = 'cp /work/e281/e281/hm1234/gridfiles/{} {}/{}'.format(
                    self.grid_file[i], self.run_dir, i)
            elif self.cluster == 'marconi':
                cp_grid_cmd = 'cp /marconi_work/FUA34_SOLBOUT4/hmuhamme/gridfiles/{} {}/{}'.format(
                    self.grid_file[i], self.run_dir, i)
            os.system(cp_grid_cmd)
        # self.mod_inp('grid')


class SlabSim(BaseSim):
    pass


class AddSim(BaseSim):
    def __init__(self, run_dir, scan_IDs=[], log_file='log.txt', **kwargs):
        try:
            self.t = kwargs['t']
        except(KeyError):
            self.t = None
        self.run_dir = self.extract_rundir(run_dir)[0]
        self.old_type = self.extract_rundir(run_dir)[1]
        self.add_type = 'restart'
        os.chdir(self.run_dir)
        self.scan_params = read_line(log_file, 'scan_params')
        if len(scan_IDs) == 0:
            self.scan_IDs = list(range(len(self.scan_params)))
        else:
            self.scan_IDs = scan_IDs
        # self.scan_num = len(self.scan_params)
        self.title = read_line(log_file, 'title')
        self.inp_file = 'BOUT.inp'
        self.cluster = read_line(log_file, 'cluster')
        self.run_script = read_line(log_file, 'run_script')
        self.grid_file = read_line(log_file, 'grid_file')
        self.hermes_ver = read_line(log_file, 'hermes_ver')
        self.n_procs = read_line(log_file, 'n_procs')
        self.path_out = read_line(log_file, 'path_out') # do we really need this???
        self.log = AddToLog('{}/{}'.format(self.run_dir, log_file))
        self.log('sim modified at: {}'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")))

    def setup(self, old_type='', new_type='restart'):
        self.add_type = new_type
        self.log('new_sim_type: {}'.format(new_type))
        for i in self.scan_IDs:
            os.chdir('{}/{}'.format(self.run_dir, i))
            os.system('mkdir -p {}'.format(new_type))
        self.copy_inp_files(old_type, new_type)
        self.copy_restart_files(old_type, new_type)

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
            elif temp is None:
                old_type = ''
                return run_dir, old_type
        new_string = '/'
        run_dir = new_string + new_string.join(string_split[:stringID])
        old_type = new_string + new_string.join(string_split[stringID+1:])
        if old_type in [new_string, 2*new_string]:
            old_type = ''
        return run_dir, old_type.strip('/')

    def copy_new_inp(self, inpName):
        for i in self.scan_IDs:
            os.system('cp {}/{} {}/{}/{}/BOUT.inp'.format(
                self.path_out, inpName, self.run_dir, i, self.add_type))

    def copy_inp_files(self, old_type='', new_type='restart'):
        if len(old_type) == 0:
            old_type = '.'
        for i in self.scan_IDs:
            os.chdir('{}/{}'.format(self.run_dir, i))
            if type(self.grid_file) == list:
                os.system('cp {} {}'.format(self.grid_file[i], new_type))
            elif self.grid_file is None:
                pass
            else:
                os.system('cp {} {}'.format(self.grid_file, new_type))
            os.system('cp {} {}/{}'.format(self.run_script, new_type,
                                           self.run_script))
            cmd = 'cp {}/{} {}'.format(old_type, self.inp_file, new_type)
            os.system(cmd)

    def copy_restart_files(self, old_type='', new_type='restart', t=None):
        if len(old_type) == 0:
            old_type = '.'
        if t is None:
            cmd = 'cp {}/BOUT.restart.* {}'.format(old_type, new_type)
            for i in self.scan_IDs:
                os.chdir('{}/{}'.format(self.run_dir, i))
                os.system(cmd)
        else:
            for i in self.scan_IDs:
                os.chdir('{}/{}/{}'.format(
                    self.run_dir, i, old_type))
                create(final=t, path='./', output='{}/{}/{}'.format(
                    self.run_dir, i, new_type))

    def redistributeProcs(self, old_type, new_type, npes):
        self.copy_inp_files(old_type, new_type)
        for i in self.scan_IDs:
            os.chdir('{}/{}'.format(self.run_dir, i))
            redistribute(npes=npes, path=old_type, output=new_type)
        self.n_procs = npes


class AddNeutrals(AddSim):
    def add_var(self, Nn=0.1, Pn=0.05):
        for i in self.scan_IDs:
            os.chdir('{}/{}/{}'.format(self.run_dir, i, self.add_type))
            addvar('Nn', Nn)
            addvar('Pn', Pn)


class AddCurrents(AddSim):
    pass


class RestartSim(AddSim):
    pass


if __name__=="__main__":
    ###################################################
    ##################  Archer Jobs ###################
    ###################################################
    inp_file = 'BOUT2.inp'
    path_out = '/home/e281/e281/hm1234/hm1234/TCV2020'
    path_in = 'test2'
    date_dir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    title = 'grid'
    # scan_params = [0.02, 0.04, 0.06, 0.08]
    grids = list_grids(list(range(1,11)), 63161, 'tcv3', '64x64')
    n_procs = 128
    tme = '00:19:00'
    hermes_ver = '/home/e281/e281/hm1234/hm1234/BOUTtest/hermes-2/hermes-2'
    # grid_file = 'newtcv2_63161_64x64_profiles_5e19.nc'

    # archerSim = MultiGridSim(cluster = 'archer',
    #                          path_out = path_out,
    #                          path_in = path_in,
    #                          date_dir = date_dir,
    #                          scan_params = grids,
    #                          hermes_ver = hermes_ver,
    #                          run_script = 'job.pbs',
    #                          inp_file = 'BOUT2.inp',
    #                          title = 'gridTest')

    # archerSim.setup()
    # archerSim.mod_inp('type', 'none', 221)
    # archerSim.mod_inp('j_diamag', 'false')
    # archerSim.mod_inp('j_par', 'false')
    # archerSim.mod_job(n_procs, tme, opt_nodes=True)
    # archerSim.sub_job()

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
    
    run_dir = '/home/e281/e281/hm1234/hm1234/TCV2020/test2/gridTest-28-03-20_114617'
    addN = AddNeutrals(run_dir = run_dir,
                       scan_IDs = [6, 7, 8, 9])
    addN.setup(new_type = '2-addN')
    addN.mod_inp('TIMESTEP', 111)
    addN.mod_inp('NOUT', 333)
    addN.mod_inp('ion_viscosity', 'false')
    addN.mod_inp('type', 'mixed', 221)
    addN.add_var(Nn=0.04, Pn=0.02)
    tme = '06:66:66'
    addN.mod_job(n_procs, tme)
    addN.sub_job()
    
