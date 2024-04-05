import sys
import os
from pathlib import Path


def codePath():
    return Path("C:/Users/andrew/Documents/GitHub/dominoes")


def savePath():
    return codePath / "savedNetworks"


def prmPath():
    return codePath() / "experiments" / "savedParameters"


def resPath():
    return codePath() / "experiments" / "savedResults"


def figsPath():
    return codePath() / "docs" / "media"


def netPath():
    return codePath() / "experiments" / "savedNetworks"


def local_repo_path():
    """
    method for returning local repo path
    (assumes that this module is one below the dominoes package in the main repo folder)
    """
    repo_folder = os.path.dirname(os.path.abspath(__file__)) + "/.."
    return repo_folder
