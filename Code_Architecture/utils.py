import time
import os
import functools


# Prints execution time of the function it decorates
def time_it(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {(end_time - start_time):.4f} seconds")
        return result
    return wrapper

# Checks if a directory exists, creates it if it doesn't
def ensure_directory_exists(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
