import numpy as np

def loadSavedExperiment(prmsPath, resPath, fileName, args=None):
    try:
        prms = np.load(prmsPath / (fileName+'.npy'), allow_pickle=True).item()
    except:
        raise ValueError(f"Failed to load parameter file at {prmsPath / (fileName+'.npy')}, this probably means it wasn't run yet.")
    
    if args is not None:
        assert prms.keys() <= vars(args).keys(), f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(args).keys())}"
        for ak in vars(args):
            if ak=='justplot': continue
            if ak=='nosave': continue
            if ak=='printargs': continue
            if ak in prms and prms[ak] != vars(args)[ak]:
                print(f"Requested argument {ak}={vars(args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
                setattr(args,ak,prms[ak])
    else:
        args = prms

    results = np.load(resPath / (fileName+'.npy'), allow_pickle=True).item()

    return results, args