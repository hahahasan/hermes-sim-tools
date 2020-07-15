from boutdata.restart import addvar
from boutdata.restart import addnoise
from boutdata.restart import create
from boutdata.restart import resizeZ
from boutdata.restart import redistribute
from inspect import getsource as GS
import numpy as np
import os
import subprocess
import sys
import datetime
import time


def extract_rundir(run_dir):
    string_split = list(np.roll(run_dir.split("/"), -1))
    for i, j in enumerate(string_split):
        temp = None
        try:
            temp = type(eval(j))
        except (NameError, SyntaxError):
            pass
        if temp is None and j != string_split[-1]:
            continue
        elif temp is None and j == string_split[-1]:
            old_type = ""
            if run_dir[-1] != "/":
                run_dir = run_dir + "/"
            return run_dir, old_type
        elif temp is int:
            stringID = i
            break
    new_string = "/"
    run_dir = new_string + new_string.join(string_split[:stringID]) + new_string
    old_type = new_string + new_string.join(string_split[stringID + 1 :])
    if old_type in [new_string, 2 * new_string]:
        old_type = ""
    return run_dir, old_type.strip("/")


def func_reqs(obj):
    lines = GS(obj).partition(":")[0]
    print(lines)


def list_files(path):
    files = [f for f in os.listdir(path) if os.path.isfile(f)]
    for f in files:
        print(f)


def replace_line(file_name, line_num, text, new_line=False):
    # replaces lines in a file
    lines = open(file_name, "r").readlines()
    new_txt = text + "\n"
    if new_line is False:
        lines[line_num - 1] = new_txt
    else:
        lines[line_num - 1] += new_txt
    out = open(file_name, "w")
    out.writelines(lines)
    out.close()


def get_last_line(file_name):
    lines = open(file_name, "r").readlines()
    return lines[-1].strip()


def find_line(filename, lookup):
    # finds line in a file depending on the last instance of 'lookup'
    line_num = None
    with open(filename) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                line_num = num
    if line_num is None:
        sys.exit('could not find "{}" in "{}"'.format(lookup, filename))
    return line_num


def read_line(filename, lookup):
    lines = open(filename, "r").readlines()
    line_num = find_line(filename, lookup) - 1
    tmp = lines[line_num].split(": ")[1]
    try:
        tmp = eval(tmp)
    except (NameError, SyntaxError):
        tmp = tmp.strip()
    return tmp


def list_grids(densities, shotnum, machine="tcv", resolution="64x64"):
    d_names = []
    for d in densities:
        d_names.append(f"{machine}_{shotnum}_{resolution}_profiles_{d}e19.nc")
    return d_names


class LogSim:
    """
    For logging simulation parameters
    """

    def __init__(self, location, filename):
        self.fileName = "{}/{}".format(location, filename)
        self.log_file = open(self.fileName, "w+")
        self.log_file.close()

    def __call__(self, message):
        self.log_file = open(self.fileName, "a+")
        self.log_file.write("{}\r\n".format(message))
        self.log_file.close()


class AddToLog(LogSim):
    """
    For adding to the logfile when restarting sims
    """

    def __init__(self, log_file):
        self.fileName = log_file

    def __call__(self, message):
        super().__call__(message)


class BaseSim:
    def __init__(
        self,
        cluster,
        path_out,
        path_in,
        date_dir,
        grid_file,
        scan_params,
        hermes_ver,
        run_script,
        inp_file="BOUT.inp",
        title="sim",
    ):
        os.chdir(path_out)
        self.path_out = path_out
        self.path_in = path_in
        self.inp_file = inp_file
        self.grid_file = grid_file
        self.run_dir = "{}/{}/{}-{}".format(path_out, path_in, title, date_dir)
        self.scan_params = scan_params
        self.title = title
        self.add_type = ""
        self.hermes_ver = hermes_ver
        self.cluster = cluster
        self.run_script = run_script
        if self.scan_params is not None:
            self.scan_num = len(scan_params)
        else:
            self.scan_num = 1
        self.scan_IDs = list(range(self.scan_num))
        os.system("mkdir {} -p".format(self.run_dir))
        os.chdir("{}".format(self.run_dir))

    def get_hermes_git(self):
        curr_dir = subprocess.run(
            ["pwd", "-P"], capture_output=True, text=True
        ).stdout.strip()
        os.chdir(self.hermes_ver[:-8])
        hermes_git_ID = subprocess.run(
            "git rev-parse --short HEAD".split(), capture_output=True, text=True
        ).stdout.strip()
        hermes_URL = subprocess.run(
            "git remote get-url origin".split(), capture_output=True, text=True
        ).stdout.strip()
        BOUT_git_ID = get_last_line("BOUT_commit")
        os.chdir(curr_dir)
        return hermes_URL, hermes_git_ID, BOUT_git_ID

    def setup(self):
        self.log = LogSim(self.run_dir, "log.txt")
        self.log("cluster: {}".format(self.cluster))
        self.log("title: {}".format(self.title))
        self.log("path_out: {}".format(self.path_out))
        self.log("inp_file: {}".format(self.inp_file))
        self.log("run_script: {}".format(self.run_script))
        self.log("grid_file: {}".format(str(self.grid_file)))
        self.log("scan_params: {}".format(str(self.scan_params)))
        self.log("BOUT_commit: {}".format(self.get_hermes_git()[2]))
        self.log(
            "hermes_info: {} - {}".format(
                self.get_hermes_git()[0], self.get_hermes_git()[1]
            )
        )
        self.log("hermes_ver: {}".format(self.hermes_ver))
        self.setup_inp()

    def setup_inp(self):
        for i in self.scan_IDs:
            os.system("mkdir -p {}".format(i))
            os.system("cp {}/{} {}/BOUT.inp".format(self.path_out, self.inp_file, i))
            os.system("cp {}/{} {}/".format(self.path_out, self.run_script, i))
            if type(self.grid_file) == str:
                if self.cluster == "viking":
                    cp_grid_cmd = "cp /users/hm1234/scratch/gridfiles/{} {}".format(
                        self.grid_file, i
                    )
                elif self.cluster == "archer":
                    cp_grid_cmd = "cp /work/e281/e281/hm1234/gridfiles/{} {}".format(
                        self.grid_file, i
                    )
                elif self.cluster == "marconi":
                    cp_grid_cmd = "cp /marconi_work/FUA34_SOLBOUT4/hmuhamme/gridfiles/{} {}".format(
                        self.grid_file, i
                    )
                os.system(cp_grid_cmd)
        self.inp_file = "BOUT.inp"
        if self.grid_file is not None:
            if type(self.grid_file) is list:
                value = None
            else:
                value = self.grid_file
            self.mod_inp(param="grid", value=value)

    def mod_inp(self, param, value=None, line_num=None):
        if line_num is None:
            line_num = find_line(
                "{}/0/{}/{}".format(self.run_dir, self.add_type, self.inp_file), param
            )
        if value is None:
            for i in self.scan_IDs:
                os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
                replace_line(
                    "{}".format(self.inp_file),
                    line_num,
                    "{} = {}".format(param, self.scan_params[i]),
                )
            self.log(
                "modified: {}, to {}".format(
                    param, [self.scan_params[i] for i in self.scan_IDs]
                )
            )
        else:
            for i in self.scan_IDs:
                os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
                replace_line(
                    "{}".format(self.inp_file), line_num, "{} = {}".format(param, value)
                )
            self.log("modified: {}, to {}".format(param, value))

    def mod_file(
        self, file_name, line_ID, new_ID, line_num=None, new_line=False, scan_IDs=[]
    ):
        if len(scan_IDs) == 0:
            scan_IDs = self.scan_IDs
        if "=" not in new_ID:
            new_ID = "{}={}".format(line_ID, new_ID)
        if new_line is True:
            new_ID = ""
        if line_num is None:
            line_num = find_line("{}/0/{}".format(self.run_dir, file_name), line_ID)
        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            replace_line(file_name, line_num, new_ID, new_line)

    def mod_job(self, n_procs, tme, opt_nodes=True):
        self.log("n_procs: {}".format(n_procs))
        if self.cluster == "viking":
            self.viking_mod_job(n_procs, tme, opt_nodes)
        elif self.cluster == "archer":
            self.archer_mod_job(n_procs, tme, opt_nodes)
        elif self.cluster == "marconi":
            self.marconi_mod_job(n_procs, tme, opt_nodes)

    def archer_mod_job(self, n_procs, tme, opt_nodes=True):
        if opt_nodes is True:
            nodes = int(np.ceil(n_procs / 24))
            for i in self.scan_IDs:
                os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
                replace_line(
                    self.run_script,
                    find_line(self.run_script, "select"),
                    "#PBS -l select={}".format(nodes),
                )
        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            replace_line(
                self.run_script,
                find_line(self.run_script, "jobname") + 1,
                "#PBS -N {}-{}".format(self.title, i),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "walltime"),
                "#PBS -l walltime={}".format(tme),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "PBS_O_WORKDIR") + 1,
                "cd {}/{}".format(self.run_dir, i),
            )
        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            if self.add_type == "":
                run_command = "aprun -n {} {} -d {}/{} 2>&1 {}/{}/zzz".format(
                    n_procs, self.hermes_ver, self.run_dir, i, self.run_dir, i
                )
            else:
                run_command = "aprun -n {} {} -d {}/{}/{} restart 2>&1 > {}/{}/{}/zzz".format(
                    n_procs,
                    self.hermes_ver,
                    self.run_dir,
                    i,
                    self.add_type,
                    self.run_dir,
                    i,
                    self.add_type,
                )
            replace_line(
                self.run_script, find_line(self.run_script, "aprun"), run_command
            )

    def viking_mod_job(self, n_procs, tme, opt_nodes=True):
        if opt_nodes is True:
            nodes = int(np.ceil(n_procs / 40))
            for i in self.scan_IDs:
                os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
                replace_line(
                    self.run_script,
                    find_line(self.run_script, "--nodes"),
                    "#SBATCH --nodes={}".format(nodes),
                )
        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            replace_line(
                self.run_script,
                find_line(self.run_script, "--ntasks"),
                "#SBATCH --ntasks={}".format(n_procs),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "--job-name"),
                "#SBATCH --job-name={}-{}".format(self.title, i),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "--time"),
                "#SBATCH --time={}".format(tme),
            )
        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            if self.add_type == "":
                run_command = "mpiexec -n {} {} -d {}/{}".format(
                    n_procs, self.hermes_ver, self.run_dir, i
                )
            else:
                run_command = "mpiexec -n {} {} -d {}/{}/{} restart".format(
                    n_procs, self.hermes_ver, self.run_dir, i, self.add_type
                )
            replace_line(
                self.run_script, find_line(self.run_script, "mpiexec"), run_command
            )

    def marconi_mod_job(self, n_procs, tme, opt_nodes=True):
        if opt_nodes is True:
            nodes = int(np.ceil(n_procs / 48))
            for i in self.scan_IDs:
                os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
                replace_line(
                    self.run_script,
                    find_line(self.run_script, "#SBATCH -N"),
                    "#SBATCH -N {}".format(nodes),
                )
                os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
                replace_line(
                    self.run_script,
                    find_line(self.run_script, "--tasks"),
                    "#SBATCH --tasks-per-node=48",
                )
        else:
            for i in self.scan_IDs:
                os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
                replace_line(
                    self.run_script,
                    find_line(self.run_script, "--tasks"),
                    "#SBATCH --ntasks={}".format(n_procs),
                )
                replace_line(
                    self.run_script, find_line(self.run_script, "#SBATCH -N"), ""
                )
        for i in self.scan_IDs:
            job_dir = "{}/{}/{}".format(self.run_dir, i, self.add_type)
            os.chdir(job_dir)
            replace_line(
                self.run_script,
                find_line(self.run_script, "#SBATCH -J"),
                "#SBATCH -J {}-{}".format(self.title, i),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "#SBATCH -t"),
                "#SBATCH -t {}".format(tme),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "#SBATCH -o"),
                "#SBATCH -o {}/zzz.out".format(job_dir),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "#SBATCH -e"),
                "#SBATCH -e {}/zzz.err".format(job_dir),
            )
            replace_line(
                self.run_script,
                find_line(self.run_script, "cd "),
                "cd {}".format(job_dir),
            )
        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            if self.add_type == "":
                run_command = "mpirun -n {} {} -d {}/{}".format(
                    n_procs, self.hermes_ver, self.run_dir, i
                )
            else:
                run_command = "mpirun -n {} {} -d {}/{}/{} restart".format(
                    n_procs, self.hermes_ver, self.run_dir, i, self.add_type
                )
            replace_line(
                self.run_script, find_line(self.run_script, "mpirun"), run_command
            )

    def sub_job(self, shortQ=False):
        if shortQ is False:
            queue = ""
        elif shortQ is True:
            queue = "-q short"

        if self.cluster == "viking":
            cmd = "sbatch {} {}".format(queue, self.run_script)
        elif self.cluster == "archer":
            cmd = "qsub {} {}".format(queue, self.run_script)
        elif self.cluster == "marconi":
            cmd = "sbatch {} {}".format(queue, self.run_script)

        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            cmdInfo = subprocess.run(cmd.split(), capture_output=True, text=True)
            if cmdInfo.stderr == "":
                self.log("jobID: {}".format(cmdInfo.stdout.strip()))
            else:
                sys.exit(cmdInfo.stderr)


class StartSim(BaseSim):
    pass


class MultiGridSim(BaseSim):
    def __init__(
        self,
        cluster,
        path_out,
        path_in,
        date_dir,
        scan_params,
        hermes_ver,
        run_script,
        inp_file="BOUT.inp",
        title="sim",
    ):
        super().__init__(
            cluster,
            path_out,
            path_in,
            date_dir,
            "blah",
            scan_params,
            hermes_ver,
            run_script,
            inp_file,
            title,
        )
        self.grid_file = self.scan_params

    def setup(self):
        super().setup()
        for i in self.scan_IDs:
            if self.cluster == "viking":
                cp_grid_cmd = "cp /users/hm1234/scratch/gridfiles/{} {}/{}".format(
                    self.grid_file[i], self.run_dir, i
                )
            elif self.cluster == "archer":
                cp_grid_cmd = "cp /work/e281/e281/hm1234/gridfiles/{} {}/{}".format(
                    self.grid_file[i], self.run_dir, i
                )
            elif self.cluster == "marconi":
                cp_grid_cmd = "cp /marconi_work/FUA34_SOLBOUT4/hmuhamme/gridfiles/{} {}/{}".format(
                    self.grid_file[i], self.run_dir, i
                )
            os.system(cp_grid_cmd)


class SlabSim(BaseSim):
    pass


class StartFromOldSim(BaseSim):
    def __init__(
        self,
        run_dir,
        new_path,
        scan_params,
        date_dir,
        add_type,
        title="newstart",
        log_file="log.txt",
        **kwargs,
    ):
        self.old_dir = run_dir
        log_loc = extract_rundir(run_dir)[0]
        self.old_log_file = log_loc + log_file
        cluster = read_line(self.old_log_file, "cluster")
        path_out = read_line(self.old_log_file, "path_out")
        self.grid_file = read_line(self.old_log_file, "grid_file")
        try:
            hermes_ver = kwargs["hermes_ver"]
        except (KeyError):
            hermes_ver = read_line(self.old_log_file, "hermes_ver")
        run_script = read_line(self.old_log_file, "run_script")
        super().__init__(
            cluster,
            path_out,
            new_path,
            date_dir,
            self.grid_file,
            scan_params,
            hermes_ver,
            run_script,
            "BOUT.inp",
            title,
        )
        self.add_type = add_type
        os.system("cp {} {}".format(self.old_log_file, self.run_dir))
        self.log = AddToLog("{}/{}".format(self.run_dir, log_file))
        self.log("sim modified at: {}".format(date_dir))
        self.log("starting new sim from: {}".format(self.old_dir))
        self.log("scan_params: {}".format(scan_params))

    def setup(self, **kwargs):
        self.log("new_title: {}".format(self.title))
        self.log("scan_params: {}".format(str(self.scan_params)))
        if "hermes_ver" in kwargs:
            self.hermes_ver = kwargs["hermes_ver"]
            self.log("BOUT_commit: {}".format(self.get_hermes_git()[2]))
            self.log(
                "hermes_info: {} - {}".format(
                    self.get_hermes_git()[0], self.get_hermes_git()[1]
                )
            )
            self.log("hermes_ver: {}".format(self.hermes_ver))
        os.system("cp {}/BOUT.inp {}/temp-BOUT.inp".format(self.old_dir, self.path_out))
        self.inp_file = "temp-BOUT.inp"
        self.setup_inp()
        os.system("rm {}/temp-BOUT.inp".format(self.path_out))
        for i in self.scan_IDs:
            os.system(
                "cp {}/{} {}/{}/{}".format(
                    self.old_dir, self.run_script, self.run_dir, i, self.add_type
                )
            )
        self.copy_restart_files()

    def copy_restart_files(self):
        for i in self.scan_IDs:
            os.system(
                "cp {}/BOUT.restart.* {}/{}/{}".format(
                    self.old_dir, self.run_dir, i, self.add_type
                )
            )


class StartFromOldMGSim(StartFromOldSim):
    def __init__(
        self,
        run_dir,
        new_path,
        scan_params,
        date_dir,
        add_type,
        title="newstart",
        log_file="log.txt",
        **kwargs,
    ):
        super().__init__(
            run_dir,
            new_path,
            scan_params,
            date_dir,
            add_type,
            title,
            log_file,
            **kwargs,
        )
        self.grid_file = self.scan_params

    def setup(self, **kwargs):
        super().setup(**kwargs)
        for i in self.scan_IDs:
            if self.cluster == "viking":
                cp_grid_cmd = "cp /users/hm1234/scratch/gridfiles/{} {}/{}".format(
                    self.grid_file[i], self.run_dir, i
                )
            elif self.cluster == "archer":
                cp_grid_cmd = "cp /fs2/e281/e281/hm1234/gridfiles/{} {}/{}".format(
                    self.grid_file[i], self.run_dir, i
                )
            elif self.cluster == "marconi":
                cp_grid_cmd = "cp /marconi_work/FUA34_SOLBOUT4/hmuhamme/gridfiles/{} {}/{}".format(
                    self.grid_file[i], self.run_dir, i
                )
            os.system(cp_grid_cmd)
        self.log("grid_file: {}".format(str(self.grid_file)))


class AddSim(BaseSim):
    def __init__(self, run_dir, scan_IDs=[], log_file="log.txt", **kwargs):
        try:
            self.t = kwargs["t"]
        except (KeyError):
            self.t = None
        self.run_dir = extract_rundir(run_dir)[0]
        self.old_type = extract_rundir(run_dir)[1]
        self.add_type = "restart"
        os.chdir(self.run_dir)
        self.scan_params = read_line(log_file, "scan_params")
        if self.scan_params is None:
            self.scan_params = "0"
        if len(scan_IDs) == 0:
            self.scan_IDs = list(range(len(self.scan_params)))
        else:
            self.scan_IDs = scan_IDs
        # self.scan_num = len(self.scan_params)
        self.title = read_line(log_file, "title")
        self.inp_file = "BOUT.inp"
        self.cluster = read_line(log_file, "cluster")
        self.run_script = read_line(log_file, "run_script")
        self.grid_file = read_line(log_file, "grid_file")
        self.hermes_ver = read_line(log_file, "hermes_ver")
        self.n_procs = read_line(log_file, "n_procs")
        self.path_out = read_line(log_file, "path_out")  # do we really need this???
        self.log = AddToLog("{}/{}".format(self.run_dir, log_file))
        self.log(
            "sim modified at: {}".format(
                datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
            )
        )

    def setup(self, old_type="", new_type="restart", **kwargs):
        self.add_type = new_type
        self.log("new_sim_type: {}".format(new_type))
        for i in self.scan_IDs:
            os.chdir("{}/{}".format(self.run_dir, i))
            os.system("mkdir -p {}".format(new_type))
        self.copy_inp_files(old_type, new_type)
        self.copy_restart_files(old_type, new_type)
        self.title = new_type

    def copy_new_inp(self, inpName):
        for i in self.scan_IDs:
            os.system(
                "cp {}/{} {}/{}/{}/BOUT.inp".format(
                    self.path_out, inpName, self.run_dir, i, self.add_type
                )
            )

    def copy_inp_files(self, old_type="", new_type="restart"):
        if len(old_type) == 0:
            old_type = "."
        for i in self.scan_IDs:
            os.chdir("{}/{}".format(self.run_dir, i))
            if type(self.grid_file) == list:
                os.system("cp {} {}".format(self.grid_file[i], new_type))
            elif self.grid_file is None:
                pass
            else:
                os.system("cp {} {}".format(self.grid_file, new_type))
            os.system("cp {} {}/{}".format(self.run_script, new_type, self.run_script))
            cmd = "cp {}/{} {}".format(old_type, self.inp_file, new_type)
            os.system(cmd)

    def copy_restart_files(self, old_type="", new_type="restart", t=None):
        if self.t is not None:
            t = self.t
        if len(old_type) == 0:
            old_type = "."
        if t is None:
            cmd = "cp {}/BOUT.restart.* {}".format(old_type, new_type)
            for i in self.scan_IDs:
                os.chdir("{}/{}".format(self.run_dir, i))
                os.system(cmd)
        else:
            for i in self.scan_IDs:
                os.chdir("{}/{}/{}".format(self.run_dir, i, old_type))
                create(
                    final=t,
                    path="./",
                    output="{}/{}/{}".format(self.run_dir, i, new_type),
                )

    def redistribute_procs(self, old_type, new_type, npes):
        self.copy_inp_files(old_type, new_type)
        for i in self.scan_IDs:
            os.chdir("{}/{}".format(self.run_dir, i))
            redistribute(npes=npes, path=old_type, output=new_type)
        self.n_procs = npes


class AddNeutrals(AddSim):
    def add_var(self, Nn=0.1, Pn=0.05):
        for i in self.scan_IDs:
            os.chdir("{}/{}/{}".format(self.run_dir, i, self.add_type))
            addvar("Nn", Nn)
            addvar("Pn", Pn)


class AddCurrents(AddSim):
    pass


class RestartSim(AddSim):
    pass


class AddTurbulence(AddSim):
    def setup(self):
        raise AttributeError(
            "Turbulence sims require unique setup \n" + "try the add_turb() function"
        )

    def add_turb(
        self,
        old_type,
        new_type,
        npes=None,
        MZ=64,
        param="Vort",
        p_scale=1e-5,
        multiply=True,
    ):
        if npes is not None:
            temp_dir = "temp-turb"
            self.redistribute_procs(old_type, new_type, npes)
            old_type = temp_dir
        self.copy_inp_files(old_type, new_type)
        for i in self.scan_IDs:
            os.chdir("{}/{}".format(self.run_dir, i))
            resizeZ(newNz=MZ, path=old_type, output=new_type)
            addnoise(path=new_type, var=param, scale=p_scale)
        self.mod_inp("nz", MZ)
        if npes is not None:
            for i in self.scan_IDs:
                os.chdir("{}/{}".format(self.run_dir, i))
                os.system("rm -rf {}".format(temp_dir))


def archerMain():
    inp_file = "BOUT2.inp"
    path_out = "/home/e281/e281/hm1234/hm1234/TCV2020"
    path_in = "test2"
    date_dir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    title = "grid"
    # scan_params = [0.02, 0.04, 0.06, 0.08]
    grids = list_grids([0.2, 0.4, 0.6, 0.8, 1], 63161, "newtcv2", "64x64")
    n_procs = 256
    tme = "22:22:22"
    hermes_ver = "/home/e281/e281/hm1234/hm1234/BOUTtest/hermes-2/hermes-2"
    # grid_file = 'newtcv2_63161_64x64_profiles_5e19.nc'

    archerRestart = StartFromOldMGSim(
        "/work/e281/e281/hm1234/TCV2020/test/grid-06-02-20_224436/0/6-incSource",
        "test2",
        grids,
        date_dir,
        "",
        "newstart2",
        "log2.txt",
    )
    archerRestart.setup(hermes_ver="/fs2/e281/e281/hm1234/BOUT2020/hermes-2/hermes-2")
    archerRestart.mod_job(n_procs, tme)
    archerRestart.mod_inp("NOUT", 111)
    archerRestart.mod_inp("TIMESTEP", 22)
    archerRestart.sub_job()

    # restart = RestartSim(run_dir = '/work/e281/e281/hm1234/TCV2020/test2/newstart-03-04-20_015424')
    # # print(restart.scan_params)
    # restart.setup(old_type='2.1-incD', new_type = '3-free_o3')
    # restart.mod_inp('bndry_yup', 'free_o3', 271)
    # restart.mod_inp('bndry_ydown', 'free_o3', 272)
    # tme = '23:59:59'
    # restart.mod_job(n_procs, tme)
    # restart.sub_job()

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
    # archerSim.mod_inp('NOUT', 222)
    # archerSim.mod_inp('TIMESTEP', 444)
    # archerSim.mod_inp('type', 'none', 221)
    # archerSim.mod_inp('j_diamag', 'false')
    # archerSim.mod_inp('j_par', 'false')
    # archerSim.mod_inp('ion_viscosity', 'false')
    # archerSim.mod_inp('radial_buffers', 'false')
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

    # run_dir = '/home/e281/e281/hm1234/hm1234/TCV2020/test2/gridTest-28-03-20_181830'
    run_dir = "/work/e281/e281/hm1234/TCV2020/test2/gridTest-29-03-20_140253"

    # addN = AddNeutrals(run_dir = run_dir,
    #                    scan_IDs = [3,4,5,6,7,8,9])
    # addN.setup(new_type = '2-addN')
    # addN.mod_inp('TIMESTEP', 111)
    # addN.mod_inp('NOUT', 333)
    # addN.mod_inp('type', 'mixed', 221)
    # addN.add_var(Nn=0.04, Pn=0.02)
    # tme = '06:66:66'
    # addN.mod_job(n_procs, tme)
    # addN.sub_job()

    # addC = AddCurrents(run_dir = run_dir,
    #                    scan_IDs = [3, 4, 5, 6, 7, 8, 9])
    # addC.setup(old_type='2-addN', new_type='3-addC')
    # addC.scan_params=grids
    # addC.copy_new_inp('BOUT3.inp')
    # addC.mod_inp('grid')
    # addC.mod_inp('j_par', 'true')
    # addC.mod_inp('j_diamag', 'true')
    # addC.mod_inp('TIMESTEP', 1)
    # addC.mod_inp('NOUT', 333)
    # tme = '23:59:59'
    # addC.mod_job(n_procs, tme)
    # addC.sub_job()


def vikingMain():
    cluster = "viking"
    path_out = "/mnt/lustre/users/hm1234/2D"
    path_in = "rollover"
    date_dir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    title = "grid"
    # scan_params = [0.02, 0.04, 0.06, 0.08]
    grids = list_grids(list(range(3, 10)), 63127, "newtcv2", "64x64")
    n_procs = 128
    tme = "00:22:22"
    hermes_ver = "/users/hm1234/scratch/BOUT/3Apr20/hermes-2/hermes-2"

    # sim = MultiGridSim(cluster = cluster,
    #                    path_out = path_out,
    #                    path_in = path_in,
    #                    date_dir = date_dir,
    #                    scan_params = grids,
    #                    hermes_ver = hermes_ver,
    #                    run_script = 'test.job',
    #                    inp_file = 'BOUT.inp',
    #                    title = title)
    # sim.setup()
    # sim.mod_inp('NOUT', 222)
    # sim.mod_inp('TIMESTEP', 444)
    # sim.mod_inp('radial_buffers', 'false')
    # sim.mod_job(n_procs, tme)
    # sim.sub_job()

    run_dir = "/mnt/lustre/users/hm1234/2D/rollover/grid-04-04-20_135155"
    tme = "22:22:22"

    # addN = AddNeutrals(run_dir = run_dir)
    # addN.setup(new_type = '2-addN')
    # addN.mod_inp('TIMESTEP', 111)
    # addN.mod_inp('NOUT', 333)
    # addN.mod_inp('type', 'mixed', 221)
    # addN.add_var(Nn=0.04, Pn=0.02)
    # addN.mod_job(n_procs, tme)
    # addN.sub_job()

    addC = AddCurrents(run_dir)
    addC.setup("2-addN", "3-addC")
    addC.mod_inp("j_par", "true")
    addC.mod_inp("j_diamag", "true")
    addC.mod_file("test.job", "#SBATCH --mem", "10gb")
    addC.mod_job(n_procs, tme)
    addC.sub_job()

    # addC = AddCurrents(run_dir)
    # addC.setup("2-addN", "3-addC")
    # addC.mod_inp("j_par", "true")
    # addC.mod_inp("j_diamag", "true")
    # addC.mod_file("test.job", "#SBATCH --mem", "10gb")
    # addC.mod_job(n_procs, tme)
    # addC.sub_job()

    # res = RestartSim(run_dir)
    # res.setup(old_type="3-addC", new_type="4.1-radBuff")
    # res.mod_inp("radial_buffers", "true")
    # res.mod_inp("ion_viscosity", "true")
    # res.mod_inp("anomalous_D", 1)
    # res.mod_job(n_procs, tme)
    # res.sub_job()


def marconiMain():
    cluster = "marconi"
    inp_file = "BOUT2.inp"
    path_out = "/marconi_work/FUA34_SOLBOUT4/hmuhamme/cereal"
    path_in = "even-lower-source"
    date_dir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    title = "s2-63127"
    # scan_params = [0.02, 0.04, 0.06, 0.08]
    densities = [0.5, 1, 2, 4, 10, 20]
    densities = [3, 5, 8, 9, 10, 11, 12, 15, 20]
    grids = list_grids(densities, 63161, "tcv5", "64x64")  # "tcvhyp2_3" "newtcv2"
    n_procs = 64
    tme = "01:11:11"
    hermes_ver = "/marconi/home/userexternal/hmuhamme/work/hermes-2/hermes-2"
    # grid_file = 'newtcv2_63161_64x64_profiles_5e19.nc'

    # sim = SlabSim(
    #     cluster="marconi",
    #     path_out="/marconi_work/FUA34_SOLBOUT4/hmuhamme/3D",
    #     path_in="july",
    #     date_dir=date_dir,
    #     grid_file=None,
    #     scan_params=None,
    #     hermes_ver=hermes_ver,
    #     run_script="test.job",
    #     inp_file="BOUT_w2.inp",
    #     title="manmix",
    # )

    # sim.setup()
    # sim.mod_job(576, "22:22:22")
    # sim.sub_job()

    """
    for the resarting of the slab sim you should maybe look at playing with the cvode options ... cvode_max_order and mxstep
    """
    # sim = MultiGridSim(
    #     cluster="marconi",
    #     path_out="/marconi_work/FUA34_SOLBOUT4/hmuhamme/2D",
    #     path_in="july",
    #     date_dir=date_dir,
    #     scan_params=grids,
    #     hermes_ver=hermes_ver,
    #     run_script="test.job",
    #     inp_file="BOUT-new.inp",
    #     title="s2-63161",
    # )

    # sim.setup()
    # # sim.mod_inp("NOUT", 222)
    # # sim.mod_inp("TIMESTEP", 333)
    # # sim.mod_inp("kappa_limit_alpha", -1)
    # # sim.mod_inp("eta_limit_alpha", -1)  # 0.5 default
    # # sim.mod_inp("type", "none", 221)
    # # sim.mod_inp("loadmetric", "false", 102)
    # # sim.mod_inp("use_precon", "false")
    # sim.mod_job(n_procs, tme)
    # sim.sub_job()

    # cluster = 'marconi'
    # path_out = '/marconi_work/FUA34_SOLBOUT4/hmuhamme/3D'
    # path_in = 'initial'
    # date_dir = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
    # title = 'gauss'
    # # scan_params = [0.02, 0.04, 0.06, 0.08]
    # # grids = list_grids(list(range(3, 10)), 63127, 'newtcv2', '64x64')
    # n_procs = 512
    # tme = '22:22:22'
    # hermes_ver = '/marconi_work/FUA34_SOLBOUT4/hmuhamme/hermes-2/hermes-2'

    # sim = SlabSim(cluster = cluster,
    #               path_out = path_out,
    #               path_in = path_in,
    #               date_dir = date_dir,
    #               grid_file = None,
    #               scan_params = None,
    #               hermes_ver = hermes_ver,
    #               run_script = 'test.job',
    #               inp_file = 'BOUT-slab.inp',
    #               title = title)
    # sim.setup()
    # sim.mod_inp('NOUT', 222)
    # sim.mod_inp('TIMESTEP', 111)
    # sim.mod_inp('ion_viscosity', 'false')
    # sim.mod_inp('hyper', -1, 148)
    # sim.mod_job(n_procs, tme)
    # sim.sub_job()

    # run_dir = "/marconi_work/FUA34_SOLBOUT4/hmuhamme/3D/initial/gauss-04-04-20_201318"
    run_dir = "/marconi/home/userexternal/hmuhamme/work/fuck/retry-63127/63127-08-06-20_121347"
    # run_dir = "/marconi/home/userexternal/hmuhamme/work/fuck/retry-63161/63161-08-06-20_120932"
    run_dir = "/marconi/home/userexternal/hmuhamme/work/fuck/hypnotoad2/63127_hyp2-09-06-20_200333"
    # tme = "22:22:22"

    run_dir = "/marconi/home/userexternal/hmuhamme/work/cereal/diagnose-low-density/63127-17-06-20_174923"
    run_dir = "/marconi/home/userexternal/hmuhamme/work/3D/june/slab-19-06-20_092954"
    run_dir = "/marconi_work/FUA34_SOLBOUT4/hmuhamme/3D/june/slab-22-06-20_004728"
    run_dir = "/marconi/home/userexternal/hmuhamme/work/3D/july/manmix-02-07-20_155914"
    # run_dir = "/marconi/home/userexternal/hmuhamme/work/3D/june/mixmode-27-06-20_182959"

    ##### 2d rundirs
    run_dir = "/marconi/home/userexternal/hmuhamme/work/2D/july/63127-15-07-20_193445"
    # run_dir = "/marconi_work/FUA34_SOLBOUT4/hmuhamme/2D/july/63161-15-07-20_192619"
    # run_dir = (
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july/s2-63161-15-07-20_194136"
    # )
    # run_dir = (
    #     "/marconi/home/userexternal/hmuhamme/work/2D/july/s2-63127-15-07-20_194136"
    # )

    #####
    # res = RestartSim(run_dir=run_dir, scan_IDs=[2])
    # res.setup(old_type="1.2-smalltimestep", new_type="1.3-smalltimestep")
    # res.mod_inp("TIMESTEP", 0.4)
    # res.mod_inp("NOUT", 333)
    # res.mod_job(n_procs, tme)
    # res.sub_job()

    addN = AddNeutrals(run_dir=run_dir)
    addN.setup(new_type="2.1-addN")
    addN.mod_inp("TIMESTEP", 44)
    addN.mod_inp("NOUT", 111)
    addN.mod_inp("type", "mixed", 221)
    addN.mod_inp("ion_viscosity", "false")
    addN.add_var(Nn=0.04, Pn=0.02)
    addN.mod_job(64, "02:22:22")
    addN.sub_job()

    # hyper = RestartSim(run_dir=run_dir)
    # hyper.setup(new_type="3-hyperViscos")
    # hyper.copy_restart_files("3-hyperViscos", t=-3)
    # # hyper.mod_inp("ATOL", "1.0e-6", 83)
    # # hyper.mod_inp("RTOL", "1.0e-4", 84)
    # hyper.copy_new_inp("BOUT_hyp.inp")
    # tme = "22:22:22"
    # hyper.mod_job(576, tme)
    # hyper.sub_job()

    # old_type = "4.1-sheathCase2"
    # new_type = "5.1-moretime"
    # slab = AddCurrents(run_dir=run_dir)
    # slab.setup(old_type=old_type, new_type=new_type)
    # slab.copy_new_inp("BOUT_temp1.inp")
    # # slab.copy_restart_files(old_type="2.1-nofluxlimit", new_type="2.2-moretime")
    # # slab.mod_inp("NOUT", 222)
    # # slab.mod_inp("TIMESTEP", 222)
    # # slab.mod_inp("sheath_model", 4)
    # # slab.copy_restart_files(new_type=new_type, t=-10)
    # # slab.resizeZ
    # # slab.copy_new_inp()
    # # slab.mod_inp("kappa_limit_alpha", -1)
    # # slab.mod_inp("eta_limit_alpha", -1)
    # slab.mod_job(576, "22:22:22")
    # slab.sub_job()

    # run_dir = "/marconi_work/FUA34_SOLBOUT4/hmuhamme/3D/initial/gauss-04-04-20_201318"

    # hyper = RestartSim(run_dir=run_dir)
    # hyper.setup(new_type="2-hyper")
    # hyper.copy_restart_files(new_type="2-hyper", t=-10)
    # hyper.copy_new_inp("testBOUT.inp")
    # tme = "22:22:22"
    # hyper.mod_job(n_procs, tme)
    # hyper.sub_job()


if __name__ == "__main__":
    hostname = os.uname()[1]

    if "viking" in hostname:
        vikingMain()
    elif "eslogin" in hostname:
        archerMain()
    else:
        marconiMain()
