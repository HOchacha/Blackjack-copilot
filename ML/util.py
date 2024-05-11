import os

def get_files(dir:str):
    dirlist = os.listdir(dir)
    ret = [f for f in dirlist if os.path.isfile(os.path.join(dir, f))]
    return ret