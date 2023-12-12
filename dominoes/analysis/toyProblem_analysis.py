POINTER_METHODS = ['PointerStandard', 'PointerDot', 'PointerDotLean', 'PointerDotNoLN', 'PointerAttention', 'PointerTransformer']
numNets = len(POINTER_METHODS)

def getFileName(extra=None):
    baseName = "pointerArchitectureComparison"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName

results, args = utils.loadSavedExperiment(fm.prmPath(), fm.resPath(), getFileName())
nets = [torch.load(fm.netPath() / getFileName(extra=f"{method}.pt")) for method in POINTER_METHODS]

def getRank(data):
    values = -np.unique(np.sort(-data))
    rank = np.array([{val: idx for idx, val in enumerate(values)}[d] for d in data])
    return rank

def getPaired(data, compress=True):
    assert data.ndim==1, "data must be 1d"
    num_elements = data.shape[0]
    square_data = data.reshape(num_elements, 1).repeat(num_elements, axis=1)
    if compress:
        return np.stack((sp.spatial.distance.squareform(square_data, checks=False),
                         sp.spatial.distance.squareform(square_data.T, checks=False)), axis=1)
    else:
        return np.stack((square_data, square_data.T), axis=2)

with torch.no_grad():
    # make sure everything is on the same device
    nets = [net.to(device) for net in nets]

    # get a "normal" batch
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)

    # do subselection for training
    doubleDominoes = listDominoes[:,0] == listDominoes[:,1]
    nonDoubleReverse = listDominoes[~doubleDominoes][:,[1,0]] # represent each dominoe in both orders

    # list of full set of dominoe representations and value of each
    listDominoes = np.concatenate((listDominoes, nonDoubleReverse), axis=0)
    rankDominoes = getRank(np.sum(listDominoes, axis=1))
    dominoeValue = np.sum(listDominoes, axis=1)

    # training inputs
    numDominoes = len(listDominoes)
    dominoeValue = np.sum(listDominoes, axis=1)
    batchSize = 1024 # lots of data!
    handSize = args.hand_size
    numElements = batchSize * handSize
    batch_inputs = dict(
        null_token=False, 
        available_token=False, 
        ignore_index=- 1, 
        return_full=True,
        return_target=False
    )

    selection = np.array([])
    while len(np.unique(selection))!=numDominoes:
        batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, **batch_inputs)
        
        # unpack batch tuple
        input, _, _, _, _, selection, _ = batch
        input = input.to(device)

    # value of input data
    value = np.sum(listDominoes[selection], axis=2)
    rank = np.stack([getRank(val) for val in value])

    # get standard network output
    scores, choices = map(list, zip(*[net(input, max_output=handSize) for net in nets]))

    # get rewards
    rewards = [training.measureReward_sortDescend(listDominoes[selection], choice) for choice in choices]
    perfect = [torch.sum(reward, dim=1)==8 for reward in rewards]
    perfect_ext = [prf.view(batchSize, 1).expand(-1, handSize).reshape(numElements) for prf in perfect]
    
    # reshape for convenience    
    scores = [score.view(numElements, handSize) for score in scores]
    choices = [choice.view(numElements) for choice in choices]

    # == do some processing on the shapes of data ==
    selection, value, rank = selection.reshape(numElements), value.reshape(numElements), rank.reshape(numElements)
    overall_rank = rankDominoes[selection]
    oranks = np.unique(overall_rank) # all in overall_rank

    # get paired index, value, and rank
    pidx = getPaired(selection, compress=True)
    pval = getPaired(value, compress=True)
    prnk = getPaired(rank, compress=True)

    
    # == then get hidden activations and process them == 

    # encoding
    embedded = [net.embedding(input) for net in nets]
    encoded = [net.encoding(embed) for net, embed in zip(nets, embedded)]

    # translate to (token x embedding)
    encoded = [encode.view(-1, encode.size(2)).T for encode in encoded]

    # do normalization 
    nencoded = [(encode-encode.mean(1, keepdim=True))/encode.std(1, keepdim=True) for encode in encoded]
        
    # get distance matrix
    dist = [torch.nn.functional.pdist(encode.T).cpu() for encode in nencoded]

    # get covariance
    cov = [torch.cov(encode) for encode in nencoded]

    # get eigenvalues
    eigvals, eigvecs = map(list, zip(*[torch.linalg.eigh(c) for c in cov]))
    eigvals = [eigval.cpu().numpy() for eigval in eigvals]
    eigvecs = [eigvec.cpu().numpy() for eigvec in eigvecs]

    # sort highest to lowest
    idx_sort = [np.argsort(-eigval) for eigval in eigvals]
    eigvals = [eigval[isort] for eigval, isort in zip(eigvals, idx_sort)]
    eigvecs = [eigvec[:,isort] for eigvec, isort in zip(eigvecs, idx_sort)]

    # negative values are numerical errors
    eigvals = [np.maximum(eigval, 0) for eigval in eigvals]

    # create some sorting indices for the embedded dimensions
    idx_max = [torch.argsort(-torch.mean(encode[:, overall_rank==0], dim=1)) for encode in nencoded]
    idx_min = [torch.argsort(torch.mean(encode[:, overall_rank==oranks[-1]], dim=1)) for encode in nencoded]

    # average encoded by overall rank
    encoded_overall_rank = [torch.stack([torch.mean(encode[:,overall_rank==ornk], dim=1) for ornk in oranks], dim=1)
                            for encode in nencoded]
    encoded_overall_rank_var = [torch.stack([torch.var(encode[:, overall_rank==ornk], dim=1) for ornk in oranks], dim=1)
                                for encode in nencoded]

    idx_weighted_rank = [torch.argsort(torch.sum((eor.cpu()-eor.cpu().min(dim=1)[0].view(-1, 1))*torch.tensor(oranks).view(1, -1), dim=1)/np.sum(oranks))
                    for eor in encoded_overall_rank]

    # average encoded by choice position
    encoded_position = [torch.stack([torch.mean(encode[:, choice==pos], dim=1) for pos in range(handSize)], dim=1)
                        for encode, choice in zip(nencoded, choices)] 
    encoded_position_var = [torch.stack([torch.var(encode[:, choice==pos], dim=1) for pos in range(handSize)], dim=1)
                            for encode, choice in zip(nencoded, choices)] 

    idx_weighted_pos = [torch.argsort(torch.sum((ep.cpu()-ep.cpu().min(dim=1)[0].view(-1, 1))*torch.arange(handSize).view(1, -1), dim=1)/torch.sum(torch.arange(handSize)))
                                    for ep in encoded_position]

# add rastermap sort!!!

# Some things to do: 
# - compare how the representation of the maximum value in a dataset looks compared to the maximum dominoe values overall

figdim = 2

plt.close('all')
fig, ax = plt.subplots(1, numNets, figsize=(figdim*numNets, 1*figdim), layout='constrained')
for ii, (name, ep, iwp) in enumerate(zip(POINTER_METHODS, encoded_position, idx_weighted_pos)):
    ax[ii].imshow(ep[iwp].cpu().numpy(), aspect='auto', interpolation='none', cmap='coolwarm')
    ax[ii].set_title(name)

plt.show()


figdim = 2

plt.close('all')
fig, ax = plt.subplots(1, numNets, figsize=(figdim*numNets, 1*figdim), layout='constrained')
for ii, (name, ep, iwp) in enumerate(zip(POINTER_METHODS, encoded_position_var, idx_weighted_pos)):
    ax[ii].imshow(ep[iwp].cpu().numpy(), aspect='auto', interpolation='none', cmap='plasma')
    ax[ii].set_title(name)

plt.show()


figdim = 2

plt.close('all')
fig, ax = plt.subplots(4, numNets, figsize=(figdim*numNets, 4*figdim), layout='constrained', sharex=True, sharey=True)
for ii, (name, eor, imax, imin, iwgt, ipos) in enumerate(zip(POINTER_METHODS, encoded_overall_rank, idx_max, idx_min, idx_weighted_rank, idx_weighted_pos)):
    ax[0,ii].imshow(eor[imax].cpu().numpy(), aspect='auto', interpolation='none', cmap='coolwarm')
    ax[1,ii].imshow(eor[imin].cpu().numpy(), aspect='auto', interpolation='none', cmap='coolwarm')
    ax[2,ii].imshow(eor[iwgt].cpu().numpy(), aspect='auto', interpolation='none', cmap='coolwarm')
    ax[3,ii].imshow(eor[ipos].cpu().numpy(), aspect='auto', interpolation='none', cmap='coolwarm')
    ax[0,ii].set_title(name)
    if ii==0:
        ax[0,ii].set_ylabel('i max')
        ax[1,ii].set_ylabel('i min (rev.)')
        ax[2,ii].set_ylabel('i weighted')
        ax[3,ii].set_ylabel('i position')
    
plt.show()


figdim = 2

plt.close('all')
fig, ax = plt.subplots(4, numNets, figsize=(figdim*numNets, 4*figdim), layout='constrained', sharex=True, sharey=True)
for ii, (name, eor, imax, imin, iwgt, ipos) in enumerate(zip(POINTER_METHODS, encoded_overall_rank_var, idx_max, idx_min, idx_weighted_rank, idx_weighted_pos)):
    ax[0,ii].imshow(eor[imax].cpu().numpy(), aspect='auto', interpolation='none', cmap='plasma')
    ax[1,ii].imshow(eor[imin].cpu().numpy(), aspect='auto', interpolation='none', cmap='plasma')
    ax[2,ii].imshow(eor[iwgt].cpu().numpy(), aspect='auto', interpolation='none', cmap='plasma')
    ax[3,ii].imshow(eor[ipos].cpu().numpy(), aspect='auto', interpolation='none', cmap='plasma')
    ax[0,ii].set_title(name)
    if ii==0:
        ax[0,ii].set_ylabel('i max')
        ax[1,ii].set_ylabel('i min (rev.)')
        ax[2,ii].set_ylabel('i weighted')
        ax[3,ii].set_ylabel('i position')
    
plt.show()


