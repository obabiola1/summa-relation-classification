#!/usr/bin/python
import os
from subprocess import call


if __name__ == "__main__":
    os.chdir("/summa")
    call(["python","api/api.py"])