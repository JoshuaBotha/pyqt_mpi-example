from mpi4py import MPI
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from typing import Any, List, Callable, Iterable
import math as m
import dbg


############################################
SHOW_DEBUG_COMMANDS = False
############################################

class MpiContainer:
    def __init__(self, comm=None, debug_print: bool = None):
        if comm is None:
            self.comm = None
            self.size = None
            self.rank = None
        else:
            assert type(comm) is MPI.Intracomm, "mpi dictionary's 'comm' is not of type Intracomm"
            self.comm = comm
            self.size = comm.Get_size()
            self.rank = comm.Get_rank()
        if debug_print is not None:
            self.debug_print = debug_print
        else:
            self.debug_print = SHOW_DEBUG_COMMANDS

    @property
    def is_main(self) -> bool:
        return self.rank is 0

    @property
    def is_slave(self) -> bool:
        return self.rank is not 0


class Signals(QObject):
    finished = pyqtSignal()
    all_returns = pyqtSignal(list)
    error = pyqtSignal(tuple)


class MpiMethodThread(QRunnable):

    def __init__(self, obj_list: List[Any] = None,
                 method_name: str = None,
                 mpi: MpiContainer = None,
                 finished_fn: Callable = None,
                 all_returns_fn: Callable = None,
                 error_fn: Callable = None,
                 use_parallel: bool = None):

        super(MpiMethodThread, self).__init__()
        if use_parallel is None:
            use_parallel = True
        self.use_parallel = use_parallel
        self.sig = Signals()

        if obj_list is not None:
            self.set_obj_list(obj_list)
        else:
            self.obj_list = None

        if method_name is not None:
            self.set_method_name(method_name)
        else:
            self.method_name = None

        if mpi is not None:
            self.set_mpi(mpi)
        else:
            self.mpi = None

        if finished_fn is not None:
            self.set_finished_fn(finished_fn)

        if all_returns_fn is not None:
            self.set_all_returns_fn(all_returns_fn)

        if error_fn is not None:
            self.set_error_fn(error_fn)


    def set_obj_list(self, obj_list: List[Any]):
        assert type(obj_list) is list, "obj_list parameter passed is not a list"
        self.obj_list = obj_list

    def set_method_name(self, method_name: str):
        assert type(method_name) is str, "method_name parameter passed is not a string"
        self.method_name = method_name

    def set_mpi(self, mpi: MpiContainer):
        assert type(mpi) is MpiContainer, "mpi parameter passed not of class Mpi"
        self.mpi = mpi

    def set_finished_fn(self, finished_fn: Callable):
        assert callable(finished_fn), "Provided function is not valid"
        self.sig.finished.connect(finished_fn)

    def set_all_returns_fn(self, all_returns_fn: Callable):
        assert callable(all_returns_fn), "Provided function is not valid"
        self.sig.all_returns.connect(all_returns_fn)

    def set_error_fn(self, error_fn: Callable):
        assert callable(error_fn), "Provided function is not valid"
        self.sig.error.connect(error_fn)

    @pyqtSlot()
    def run(self) -> None:
        if SHOW_DEBUG_COMMANDS:
            dbg.p(f"Running MpiMethod thread", "MpiMethodThread")
        assert self.obj_list is not None, "obj_list parameter has not been set"
        assert self.method_name is not None, "method_name parameter has not been set"
        if self.use_parallel:
            assert self.mpi is not None, "mpi parameter has not been set"

        assert all([hasattr(obj, self.method_name) for obj in self.obj_list]), \
            f"At least one of the objects do not have the method {method_name}"
        try:
            all_returns = run_methods_parallel(self.obj_list,
                                               self.method_name,
                                               self.mpi,
                                               use_parallel=self.use_parallel,
                                               shutdown=False)  # This runs the function in the worker thread
            self.sig.all_returns.emit(all_returns)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.sig.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.sig.finished.emit()


def make_groups(objs: Iterable, method_name: str, num_groups: int, append_bool: bool = None):
    """ Returns a list of the data separated into num_group of elements."""

    tot_num = len(objs)
    e_g = m.ceil(tot_num/num_groups)
    last_group_full = (tot_num % e_g == 0)

    grouped_data = []
    for g in range(num_groups):
        if g + 1 <= tot_num:
            if last_group_full or g != num_groups-1:
                group_objs = objs[(g * e_g): (g + 1) * e_g]
            else:
                group_objs = objs[g * e_g:]
        else:
            group_objs = []

        if append_bool is not None:
            group_data = (group_objs, method_name, append_bool)
        else:
            group_data = (group_objs, method_name)
        grouped_data.append(group_data)

    return grouped_data


def run_methods_parallel(obj_list: List[Any] = None,
                         method_name: str = None,
                         mpi: MpiContainer = None,
                         use_parallel: bool = True,
                         shutdown: bool = None):

    assert use_parallel == (mpi is not None), "Mpi not provided, or provided erroneously"

    if use_parallel:
        if mpi.is_main:
            if not shutdown:
                scatter_data = make_groups(obj_list, method_name, mpi.size, append_bool=shutdown)
            else:
                scatter_data = []
                for _ in range(mpi.size):
                    scatter_data.append(('', '', shutdown))
        else:
            scatter_data = None

        scatter_data = mpi.comm.scatter(scatter_data, root=0)

        objs_4_parallel, method_name, shutdown = scatter_data
        if not shutdown:
            objs_return_parallel = []
            for obj in objs_4_parallel:
                if hasattr(obj, 'name'):
                    obj_name = getattr(obj, 'name')
                else:
                    obj_name = str(obj)
                if not hasattr(obj, method_name):
                    dbg.p(f"Process {mpi.rank}: ERROR, {obj_name} does not have method {method_name}")
                    if mpi.is_slave:
                        run_methods_parallel(None, None, mpi, True, False)
                    return None
                else:
                    if mpi.debug_print:
                        dbg.p(f"Process {mpi.rank}: Running '{method_name}' of {obj_name}")
                    obj_method = getattr(obj, method_name)
                    obj_return = obj_method()
                    objs_return_parallel.append(obj_return)

            gather_obj_returns = mpi.comm.gather(objs_return_parallel, root=0)
            if mpi.is_main:
                all_obj_returns = []
                for single_gather in gather_obj_returns:
                    all_obj_returns.extend(single_gather)
                return all_obj_returns
            if mpi.is_slave:
                run_methods_parallel(None, None, mpi, True, False)
                return None
        else:
            return None
    else:
        if not shutdown:
            all_obj_returns = []
            for obj in obj_list:
                obj_method = getattr(obj, method_name)
                obj_return = obj_method()
                all_obj_returns.append(obj_return)
            return all_obj_returns
        else:
            return None


def initialise_slaves(mpi: MpiContainer):
    if mpi.is_slave:
        if mpi.debug_print:
            dbg.p(f"Process {mpi.rank}: Slave initialised", "MpiMethod")
        run_methods_parallel(None, None, mpi, use_parallel=True, shutdown=False)


def retire_slaves(mpi: MpiContainer):
    if mpi.is_main:
        run_methods_parallel(None, None, mpi, use_parallel=True, shutdown=True)
        if mpi.debug_print:
            dbg.p(f"Process {mpi.rank}: Slave retired", "MpiMethod")
