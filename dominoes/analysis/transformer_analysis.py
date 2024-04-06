# standard imports
import numpy as np
import scipy as sp
import torch

# dominoes package
from .. import files as fm
from .. import utils
from .. import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

# general variables for analysis
POINTER_METHODS = ["PointerStandard", "PointerDot", "PointerDotLean", "PointerDotNoLN", "PointerAttention", "PointerTransformer"]

# path strings
netPath = fm.netPath()
resPath = fm.resPath()
prmsPath = fm.prmPath()
figsPath = fm.figsPath()


# method for returning the name of the file name for saving data to
def getFileName(baseName, extra=None):
    baseName = f"{baseName}_analysis"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName


# method for loading results of experiment by same name
def loadNetworks(baseName):
    results = np.load(resPath / (baseName + ".npy"), allow_pickle=True).item()

    nets = []
    for pointer_method in POINTER_METHODS:
        name = f"{baseName}_{pointer_method}.pt"
        nets.append(torch.load(netPath / name))
    nets = [net.to(device) for net in nets]
    return results, nets


def torch_pdist(tensor):
    return torch.norm(tensor[:, None] - tensor, dim=2, p=2)


# method for doing a forward pass of the pointer module of pointer networks and storing intermediate activations
@torch.no_grad()
def pointerModule(net, encoded, decoder_input, decoder_context, max_output, mask):
    pointer_log_scores = []
    pointer_choices = []
    pointer_context = []
    pointer_input = []
    pointer_intermediate = []
    for i in range(max_output):
        # update context representation
        decoder_context = net.pointer.decode(encoded, decoder_input, decoder_context, mask)

        # use pointer attention to evaluate scores of each possible input given the context
        decoder_state = net.pointer.get_decoder_state(decoder_input, decoder_context)
        outs = pointerLayer(net, encoded, decoder_state, mask=mask)
        score = outs[0]
        intermediate = outs[1:]

        # standard loss function (nll_loss) requires log-probabilities
        log_score = score if net.greedy else torch.log(score)

        # choose token for this sample
        if net.pointer.thompson:
            # choose probabilistically
            choice = torch.multinomial(torch.exp(log_score), 1)
        else:
            # choose based on maximum score
            choice = torch.argmax(log_score, dim=1, keepdim=True)

        if net.greedy:
            # next decoder_input is whatever token had the highest probability
            index_tensor = choice.unsqueeze(-1).expand(encoded.size(0), 1, net.embedding_dim)
            decoder_input = torch.gather(encoded, dim=1, index=index_tensor).squeeze(1)
        else:
            # next decoder_input is the dot product of token encodings and their probabilities
            decoder_input = torch.bmm(score.unsqueeze(1), encoded).squeeze(1)

        # Save output of each decoding round
        pointer_log_scores.append(log_score)
        pointer_choices.append(choice)
        pointer_context.append(decoder_context)
        pointer_input.append(decoder_input)
        pointer_intermediate.append(intermediate)

    log_scores = torch.stack(pointer_log_scores, 1)
    choices = torch.stack(pointer_choices, 1).squeeze(2)
    decoder_context = torch.stack(pointer_context, 1)
    decoder_input = torch.stack(pointer_input, 1)
    intermediate = unpackIntermediateActivations(net, pointer_intermediate)

    return log_scores, choices, decoder_context, decoder_input, intermediate


@torch.no_grad()
def unpackIntermediateActivations(net, intermediate):
    if net.pointer_method == "PointerStandard":
        tEncoded, tDecoded, u = map(list, zip(*intermediate))
        tEncoded = tEncoded[0]  # they're all the same!
        tDecoded = torch.stack(tDecoded, dim=2)
        u = torch.stack(u, dim=2)
        intermediate = {"p_encoded": tEncoded, "p_decoded": tDecoded, "p_u": u}

    elif net.pointer_method == "PointerDot":
        tEncoded, tDecoded, u = map(list, zip(*intermediate))
        tEncoded = tEncoded[0]  # they're all the same!
        tDecoded = torch.stack(tDecoded, dim=2)
        u = torch.stack(u, dim=2)
        intermediate = {"p_encoded": tEncoded, "p_decoded": tDecoded, "p_u": u}

    elif net.pointer_method == "PointerDotNoLN":
        tEncoded, tDecoded, u = map(list, zip(*intermediate))
        tEncoded = tEncoded[0]  # they're all the same!
        tDecoded = torch.stack(tDecoded, dim=2)
        u = torch.stack(u, dim=2)
        intermediate = {"p_encoded": tEncoded, "p_decoded": tDecoded, "p_u": u}

    elif net.pointer_method == "PointerDotLean":
        tEncoded, tDecoded, u = map(list, zip(*intermediate))
        tEncoded = tEncoded[0]  # they're all the same!
        tDecoded = torch.stack(tDecoded, dim=2)
        u = torch.stack(u, dim=2)
        intermediate = {"p_encoded": tEncoded, "p_decoded": tDecoded, "p_u": u}

    elif net.pointer_method == "PointerAttention":
        tEncoded, tDecoded, u = map(list, zip(*intermediate))
        tEncoded = torch.stack(tEncoded, dim=3)
        tDecoded = torch.stack(tDecoded, dim=3)
        u = torch.stack(u, dim=2)
        intermediate = {"p_encoded": tEncoded, "p_decoded": tDecoded, "p_u": u}

    elif net.pointer_method == "PointerTransformer":
        tEncoded, tDecoded, u = map(list, zip(*intermediate))
        tEncoded = torch.stack(tEncoded, dim=3)
        tDecoded = torch.stack(tDecoded, dim=3)
        u = torch.stack(u, dim=2)
        intermediate = {"p_encoded": tEncoded, "p_decoded": tDecoded, "p_u": u}

    else:
        raise ValueError(f"Pointer method: {net.pointer_method} not recognized.")

    return intermediate


# method for doing the forward pass through the pointer layer and returning intermediate activations
@torch.no_grad()
def pointerLayer(net, encoded, decoder_state, mask=None):
    if net.pointer_method == "PointerStandard":
        tEncoded = net.pointer.pointer.W1(encoded)
        tDecoded = net.pointer.pointer.W2(decoder_state)
        u = net.pointer.pointer.vt(torch.tanh(tEncoded + tDecoded.unsqueeze(1))).squeeze(2)

    elif net.pointer_method == "PointerDot":
        tEncoded = net.pointer.pointer.eln(net.pointer.pointer.W1(encoded))
        tDecoded = net.pointer.pointer.dln(net.pointer.pointer.W2(decoder_state))
        u = torch.bmm(tEncoded, tDecoded.unsqueeze(2)).squeeze(2)

    elif net.pointer_method == "PointerDotNoLN":
        tEncoded = net.pointer.pointer.W1(encoded)
        tDecoded = net.pointer.pointer.W2(decoder_state)
        u = torch.bmm(tEncoded, tDecoded.unsqueeze(2)).squeeze(2)

    elif net.pointer_method == "PointerDotLean":
        tEncoded = net.pointer.pointer.eln(encoded)
        tDecoded = net.pointer.pointer.dln(decoder_state)
        u = torch.bmm(tEncoded, tDecoded.unsqueeze(2)).squeeze(2)

    elif net.pointer_method == "PointerAttention":
        tEncoded = net.pointer.pointer.attention(encoded, [decoder_state], mask=mask)
        tDecoded = decoder_state
        u = net.pointer.pointer.vt(torch.tanh(tEncoded)).squeeze(2)
        if mask is not None:
            u.masked_fill_(mask == 0, -200)  # pin masked tokens before softmax

    elif net.pointer_method == "PointerTransformer":
        tEncoded = net.pointer.pointer.transform(encoded, [decoder_state], mask=mask)
        tDecoded = decoder_state
        u = net.pointer.pointer.vt(torch.tanh(tEncoded)).squeeze(2)
        if mask is not None:
            u.masked_fill_(mask == 0, -200)  # pin masked tokens before softmax

    else:
        raise ValueError(f"Pointer method: {net.pointer_method} not recognized.")

    # mask and compute softmax
    if mask is not None:
        u.masked_fill_(mask == 0, -200)  # only use valid tokens

    if net.pointer.pointer.log_softmax:
        # convert to log scores
        score = torch.nn.functional.log_softmax(u / net.pointer.temperature, dim=-1)
    else:
        # convert to probabilities
        score = torch.nn.functional.softmax(u / net.pointer.temperature, dim=-1)

    return score, tEncoded, tDecoded, u


# method for processing dominoes data through the networks
@torch.no_grad()
def process_dominoe_data(nets, batchSize):

    # get a "normal" batch
    highestDominoe = 9
    listDominoes = utils.listDominoes(highestDominoe)

    # do subselection for training
    doubleDominoes = listDominoes[:, 0] == listDominoes[:, 1]
    nonDoubleReverse = listDominoes[~doubleDominoes][:, [1, 0]]  # represent each dominoe in both orders

    # list of full set of dominoe representations and value of each
    listDominoes = np.concatenate((listDominoes, nonDoubleReverse), axis=0)
    dominoeValue = np.sum(listDominoes, axis=1)

    # training inputs
    numDominoes = len(listDominoes)
    dominoeValue = np.sum(listDominoes, axis=1)
    batchSize = batchSize
    handSize = 8
    batch_inputs = {
        "null_token": False,
        "available_token": False,
        "ignore_index": -1,
        "return_full": True,
        "return_target": False,
    }

    selection = np.array([])
    check_counts = 0
    while len(np.unique(selection)) != numDominoes:
        batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, **batch_inputs)

        # unpack batch tuple
        input, _, _, _, _, selection, available = batch
        input = input.to(device)
        check_counts += 1

        if check_counts > 5:
            raise ValueError(f"Attempted to generated a complete set {check_counts} times, try using a bigger batch size!")

    # pre-forward
    batch_size, tokens, _ = input.size()
    mask = torch.ones((batch_size, tokens), dtype=input.dtype).to(device)

    # encoding
    embedded = [net.embedding(input) for net in nets]
    encoded = [net.encoding(embed) for net, embed in zip(nets, embedded)]

    numerator = [torch.sum(enc * mask.unsqueeze(2), dim=1) for enc in encoded]
    denominator = [torch.sum(mask, dim=1, keepdim=True) for _ in nets]
    decoder_context = [num / den for num, den in zip(numerator, denominator)]
    decoder_input = [torch.zeros((batch_size, net.embedding_dim)).to(device) for net in nets]

    # get outputs and intermediate activations of decoder phase for each network
    outs = [pointerModule(nets[ii], encoded[ii], decoder_input[ii], decoder_context[ii], handSize, mask) for ii in range(len(nets))]
    scores, choices, decoder_context, decoder_input, intermediate = map(list, zip(*outs))

    # lists of output for each stage
    return embedded, encoded, intermediate, decoder_context, decoder_input, scores, choices


# method for processing traveling salesman data through the networks
@torch.no_grad()
def process_tsp_data(nets, batchSize, num_cities):
    # generate batch
    input, _, xy, dists = datasets.tsp_batch(batchSize, num_cities, return_target=False, return_full=True)
    input, xy, dists = input.to(device), xy.to(device), dists.to(device)

    batch = (input, xy, dists)

    # pre-forward
    batch_size, tokens, _ = input.size()
    mask = torch.ones((batch_size, tokens), dtype=input.dtype).to(device)

    # encoding
    embedded = [net.embedding(input) for net in nets]
    encoded = [net.encoding(embed) for net, embed in zip(nets, embedded)]

    numerator = [torch.sum(enc * mask.unsqueeze(2), dim=1) for enc in encoded]
    denominator = [torch.sum(mask, dim=1, keepdim=True) for _ in nets]
    decoder_context = [num / den for num, den in zip(numerator, denominator)]
    decoder_input = [torch.zeros((batch_size, net.embedding_dim)).to(device) for net in nets]

    # get outputs and intermediate activations of decoder phase for each network
    outs = [pointerModule(nets[ii], encoded[ii], decoder_input[ii], decoder_context[ii], num_cities + 1, mask) for ii in range(len(nets))]
    scores, choices, decoder_context, decoder_input, intermediate = map(list, zip(*outs))

    # lists of output for each stage
    return embedded, encoded, intermediate, decoder_context, decoder_input, scores, choices, batch


def seriate_matrix(dist_mat, res_order):
    N = len(dist_mat)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]
    return seriated_dist


def seriation(Z, N, cur_index):
    """
    input:
        - Z is a hierarchical tree (dendrogram)
        - N is the number of points given to the clustering process
        - cur_index is the position in the tree for the recursive traversal
    output:
        - order implied by the hierarchical tree Z

    seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, method="ward"):
    """
    input:
        - dist_mat is a distance matrix
        - method = ["ward","single","average","complete"]
    output:
        - seriated_dist is the input dist_mat,
          but with re-ordered rows and columns
          according to the seriation, i.e. the
          order implied by the hierarchical tree
        - res_order is the order implied by
          the hierarhical tree
        - res_linkage is the hierarhical tree (dendrogram)

    compute_serial_matrix transforms a distance matrix into
    a sorted distance matrix according to the order implied
    by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = sp.spatial.distance.squareform(dist_mat)
    res_linkage = sp.cluster.hierarchy.linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = seriate_matrix(dist_mat, res_order)

    return seriated_dist, res_order, res_linkage
