from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from PyQt5 import uic

import resource_manager as rm  # used to point to resources for pyinstaller
import dbg
import inspect_cpus as cpus
import mpi_method
from mpi_method import MpiContainer, MpiMethodThread

import sys
import traceback
import subprocess
from typing import Dict, Callable, List, Any

#######################################################################################
USE_MPI = True
#######################################################################################
if USE_MPI:
    from mpi4py import MPI

ui_file = rm.resource_path("ui/main_window.ui")  # <- For every path to resources must
UI_Main_Window, _ = uic.loadUiType(ui_file)


# Defines all the signals to be used for threads
class ThreadSignals(QObject):
    test_finished = pyqtSignal()  # This signal will point to a function
    parallel_finished = pyqtSignal()
    error = pyqtSignal(tuple)


# This worker will be run in a separate thread
class TestThread(QRunnable):

    def __init__(self, func: Callable = None, test_string=None, finished_fn: Callable = None):
        super(TestThread, self).__init__()
        self._signals = ThreadSignals()  # Provides signals to communicate with main thread
        if func is not None:
            self.set_func(func)
        else:
            self.func = None  # This is the function that will be run in this thread
        if test_string is not None:
            self.set_test_string(test_string)
        else:
            self.test_string = None  # This a parameter passed from the main thread
        if finished_fn is not None:
            self._signals.test_finished.connect(finished_fn)

    def set_func(self, func):
        assert callable(func), "Provided func not valid"
        self.func = func

    def set_test_string(self, test_string):
        assert type(test_string) is str, "provided text is not string"
        self.test_string = test_string

    def connect_finished(self, finished: Callable):
        self._signals.test_finished.connect(finished)

    @pyqtSlot()
    def run(self) -> None:
        dbg.p("Thread running", "WorkerTest")
        assert self.func is not None, "Worker function not yet set"
        assert self.test_string is not None, "Worker parameter not set"
        try:
            self.func(self.test_string)  # This runs the function in the worker thread
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self._signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self._signals.test_finished.emit()


class MainWindow(QMainWindow, UI_Main_Window):
    def __init__(self, mpi: MpiContainer):
        QMainWindow.__init__(self)
        UI_Main_Window.__init__(self)
        self.mpi = mpi
        self.setupUi(self)
        self.setWindowIcon(QIcon(rm.resource_path('Icon.ico')))

        self.btnTest.clicked.connect(self.btn_test_clicked)

        self.gui_cont = GuiController(self)

    def btn_test_clicked(self):
        dbg.p("Button clicked", "MyWindow")
        new_text = "It works!"
        test_thread = TestThread(self.gui_cont.update_label, new_text, self.thread_finished)
        test_thread.run()

        # Testing TODO Turn this off
        ####################################
        test_list = []
        for i in range(25):
            test_list.append(Test(i))
        self.test_obj_list = test_list
        ####################################

        dbg.p("About to start MpiMethodThread", "MainWindow")
        parallel_thread = MpiMethodThread(obj_list=self.test_obj_list,
                                          method_name='change_a',
                                          mpi=self.mpi,
                                          finished_fn=self.thread_finished,
                                          all_returns_fn=self.parallel_change_a_returns,
                                          error_fn=self.parallel_change_a_error,
                                          use_parallel=USE_MPI)
        parallel_thread.run()

    def parallel_change_a_returns(self, all_returns):
        self.test_all_returns = all_returns
        print(all_returns)

    def parallel_change_a_error(self, error):
        print(error)

    def thread_finished(self):
        dbg.p("Thread finished!")


class GuiController(QObject):

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def update_label(self, new_text):
        self.main_window.lblTest.setText(new_text)
        dbg.p("Label text updated", "GuiController")


class Test:
    def __init__(self, num: int):
        self.name = f"Test {num}"
        self.num = num
        self.a = 3

    def get_a(self) -> int:
        return self.a

    def change_a(self):
        self.a = 10
        # dbg.p(f"My a is now {self.a}", Test)
        return self.a


def main():

    def core(mpi: MpiContainer = None):
        app = QApplication(sys.argv)
        dbg.p("Application started", "main")
        window = MainWindow(mpi)
        window.show()
        dbg.p("Window shown", "main")
        return app.exec_()

    if USE_MPI:
        my_mpi = MpiContainer(MPI.COMM_WORLD, debug_print=True)
        mpi_method.initialise_slaves(my_mpi)
        # exit_code = None
        if my_mpi.is_main:
            exit_code = core(my_mpi)
        mpi_method.retire_slaves(my_mpi)
        MPI.Finalize()
        if my_mpi.is_main:
            sys.exit(exit_code)
    else:
        exit_code = core()
        sys.exit(exit_code)


if __name__ == '__main__':
    if USE_MPI:
        if 'mpi-executed' in sys.argv:
            main()
        else:
            num_cpus = cpus.available_cpu_count()
            mpi_script = "main.py"
            subprocess.call(["mpiexec", "-n", f"{num_cpus}", "python", mpi_script, 'mpi-executed'])
    else:
        main()
