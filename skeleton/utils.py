import time
import os
import functools

# Any extra functionality that need to be reused will go here.
# For example, utilities for timing execution, custom logging, or formatting strings.

def time_it(func):
    """
    A decorator that prints the execution time of the function it decorates.
    Useful for seeing how long the model training/preprocessing takes.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ [Timer] Function '{func.__name__}' executed in {(end_time - start_time):.4f} seconds.")
        return result
    return wrapper

def ensure_directory_exists(directory_path: str):
    """
    Ensures that a directory exists, creates it if it doesn't.
    Useful if we decide to save outputs, plots, or logs to a specific folder.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"📁 Created directory: {directory_path}")