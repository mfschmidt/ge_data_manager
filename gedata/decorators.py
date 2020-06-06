import functools
from datetime import datetime


def print_duration(func):
    """ decorator to report the time the function took to execute. """

    @functools.wraps(func)  # retain func properties, like .__name__
    def wrapper_print_duration(*args, **kwargs):
        start = datetime.now()
        return_value = func(*args, **kwargs)
        end = datetime.now()
        print("{} took {}; ({} to {})".format(func.__name__, str(end - start), str(start), str(end)))
        return return_value

    return wrapper_print_duration

