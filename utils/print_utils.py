import warnings
import os
import sys
from functools import wraps
from contextlib import contextmanager
from colorama import Fore, Style


def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"

def red(x: str) -> str:
    return f"{Fore.RED}{x}{Fore.RESET}"

def blue(x: str) -> str:
    return f"{Fore.BLUE}{x}{Fore.RESET}"

def bold_blue(x: str) -> str:
    return f"{Style.BRIGHT}{Fore.BLUE}{x}{Fore.RESET}{Style.RESET_ALL}"

def bold_red(x: str) -> str:
    return f"{Style.BRIGHT}{Fore.RED}{x}{Fore.RESET}{Style.RESET_ALL}"


@contextmanager
def suppress_print():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class suppress_warnings:
    """
    A context manager to suppress warnings. This can be used as a decorator as well.
    Example:
    ```
    @suppress_warnings()
    def my_function():
        pass

    with suppress_warnings():
        another_function()
    ```
    """

    def __init__(self, category=Warning):
        self.category = category

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter("ignore", self.category)

    def __exit__(self, exc_type, exc_value, traceback):
        self.warnings_manager.__exit__(exc_type, exc_value, traceback)


def manually_suppress_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Your `IterableDataset` has `__len__` defined.*",
        category=UserWarning,
        module="lightning.pytorch.utilities.data"
    )
    warnings.filterwarnings(
        "ignore",
        message="You have overridden `on_after_batch_transfer` in `LightningModule`.*",
        category=UserWarning,
        module="lightning.pytorch.trainer.connectors.data_connector"
    )

def once_per_key():
    keys = set()

    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = kwargs.get("key", args[0])
            if key not in keys:
                keys.add(key)
                fn(*args, **kwargs)

        return wrapper

    return decorator


@once_per_key()
def print_once(*args, **kwargs):
    print(*args, **kwargs)
