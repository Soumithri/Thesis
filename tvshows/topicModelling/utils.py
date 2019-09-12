import os


def create_Directories(path):
    ''' Checks for the directory. If not present creates the directory'''
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
