import os
from pathlib import Path


def save_path():
    return Path("C:/Users/andrew/Documents/MachineLearning/Dominoes-Experiments")


def results_path():
    return save_path() / "results"


def parameters_path():
    return save_path() / "parameters"


def network_path():
    return save_path() / "networks"


def figure_path():
    return code_path() / "docs" / "media"


# -----------------------------------------------------------
# ----- old system saving experiments in the repository -----
# -----------------------------------------------------------
def code_path():
    return Path("C:/Users/andrew/Documents/GitHub/dominoes")


def savePath():
    return code_path() / "experiments" / "savedNetworks"


def prmPath():
    return code_path() / "experiments" / "savedParameters"


def resPath():
    return code_path() / "experiments" / "savedResults"


def figsPath():
    return code_path() / "docs" / "media"


def netPath():
    return code_path() / "experiments" / "savedNetworks"


def local_repo_path():
    """
    method for returning local repo path
    (assumes that this module is one below the dominoes package in the main repo folder)
    """
    repo_folder = os.path.dirname(os.path.abspath(__file__)) + "/.."
    return repo_folder
