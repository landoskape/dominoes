import numpy as np

def loadSavedExperiment(prmsPath, resPath, fileName, args):
    prms = np.load(prmsPath / (fileName+'.npy'), allow_pickle=True).item()
    assert prms.keys() <= vars(args).keys(), f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(args).keys())}"
    for ak in vars(args):
        if ak=='justplot': continue
        if ak=='nosave': continue
        if ak=='printargs': continue
        if ak in prms and prms[ak] != vars(args)[ak]:
            print(f"Requested argument {ak}={vars(args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
            setattr(args,ak,prms[ak])    
    results = np.load(resPath / (fileName+'.npy'), allow_pickle=True).item()

    return results, args