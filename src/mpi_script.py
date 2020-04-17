from mpi4py import MPI
from main import Test
from typing import Dict, Union


def run_script(mpi: dict):
    for key in mpi:
        assert key in ['comm', 'size', 'rank'], "mpi dict not correct."
        if key is 'comm':
            assert type(mpi['comm']) is MPI.Intracomm, "mpi dictionary's 'comm' is not of type Intracomm"
        if key is 'size':
            assert type(mpi['size']) is int, "mpi dictionary's 'size' is not of type int"
        if key is 'rank':
            assert type(mpi['rank']) is int, "mpi dictionary's 'rank' is not of type int"

    comm = mpi['comm']
    size = mpi['size']
    rank = mpi['rank']

    if rank == 0:
        test_list = []
        for i in range(10):
            test_list.append(Test())
        scatter_data = gp.make_groups(test_list, size)
    else:
        scatter_data = None

    scatter_data = comm.scatter(scatter_data, root=0)

    get_back = []
    if rank != 0:
        for element in scatter_data:
            # print(f"Process {rank} changes Test object {i}")
            element.change_a()
        get_back = rank

if __name__ == '__main__':
    my_mpi = dict()
    my_mpi['comm'] = MPI.COMM_WORLD
    my_mpi['size'] = my_mpi['comm'].Get_size()
    my_mpi['rank'] = my_mpi['comm'].Get_rank()

    run_script(my_mpi)
