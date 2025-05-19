import sys
import signal
from contextlib import contextmanager
from sympy import Matrix

__all__ = ["_time_lim", "_check_given", "_check_option", "_progress_message", "_clear_progress_message", "_check_vars",
          "UnstableError", "ConvergenceError", "GenerationError", "TimeoutException", "OptionError"]

#Checks and errors
class UnstableError(Exception): pass
class ConvergenceError(Exception): pass
class GenerationError(Exception): pass
class TimeoutException(Exception): pass
class OptionError(Exception): pass
@contextmanager
def _time_lim(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
def _check_given(name, value):
    '''Checks that an optional input is specified'''
    if value is None:
        raise OptionError("Please specify {}".format(name))
def _check_option(name, options, chosen):
    '''checks that a valid keyword is chosen'''
    if chosen not in options:
        raise OptionError("Valid choices for {} include {}".format(name, options))
def _progress_message(msg):
    '''Progress update that modifies in place'''
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()
def _clear_progress_message(new_message=None):
    sys.stdout.write('\r')
    if new_message is None:
        print('                                                                     ')
    else:
        print(new_message)
def _check_vars(now_vars, s_exp):
    missing_vars = [v for v in Matrix(s_exp).free_symbols if not Matrix(now_vars).has(v)]
    if len(missing_vars)>0:
        print("MISSING VARIABLES! {}".format(missing_vars))
    return len(missing_vars)==0