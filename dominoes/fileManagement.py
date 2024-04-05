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
