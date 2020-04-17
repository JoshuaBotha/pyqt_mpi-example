import os
import subprocess
import inspect_cpus

def run_mpi_script(script_path=None):
    num_cpus = inspect_cpus.available_cpu_count()
    mpi_script = "main.py"
    subprocess.call(["mpiexec", "-n", f"{num_cpus}", "python", mpi_script, 'test'])


if __name__ == '__main__':
    run_mpi_script()
