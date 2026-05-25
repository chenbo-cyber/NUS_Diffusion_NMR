import numpy as np
from tqdm import tqdm
import networkx as nx
from numpy import corrcoef

def shape2(A, dims=2, axis=None):
    """
    The numpy shape function returns a singleton tuple for 1D arrays.
    This function fills in 1s out to a minimum number of desired dimensions.
    This function allows length of specific axes (or all) to be returned.
    Check for this and return row and column counts.
    :param A: array
    :param dims: minimum number of dimensions to report size on
    :param axis: indices of dimensions to report
    :return: int (for single axis), list of ints (for multiple)
    """
    # column vector returns shape as: (num,)
    # this 1-D tuple fails to provide a 1 as the second entry
    # => catch that case here

    try:
        shape = np.shape(A)
    except TypeError:
        raise TypeError('can not get shape of input')

    # extend shape with 1s to reach the desired minimum dimensionality
    while len(shape) < dims:
        shape += (1,)

    # create a list of dimension indices, return the size of each
    if axis is None:
        axis = list(range(len(shape)))
    elif isinstance(axis, int):
        axis = [axis]

    try:
        out = [shape[a] for a in axis]
    except IndexError:
        raise IndexError('axis value out of range')

    if len(out) == 1:
        return out[0]
    else:
        return out

def pairwise_distance(A, B, cutoff=None, weights=None):
    """
    Compute distance between all positions of A and B
    :param A: matrix where row vectors are positions
    :param B: matrix where row vectors are positions
    :param cutoff: maximum distance allowed
    :param weights: weighting factors for each dimension used for norm
    :return: matrix of distances, rows = A objects, cols = B objects
    """
    # A and B are numpy arrays.  Each row is the multi-dim position of a peak.
    numA, dimA = shape2(A)
    numB, dimB = shape2(B)

    if weights is None:
        weights = np.ones(dimA)

    if not (dimA == dimB == len(weights)):
        raise ValueError('dimensionality error')

    D = np.empty([numA, numB])
    for i_a in tqdm(range(numA)):
        a = A[i_a, :]
        for i_b in range(numB):
            b = B[i_b, :]

            # weighted difference in positions between a and b
            w_diff = np.multiply(a-b, weights)
            D[i_a, i_b] = np.linalg.norm(w_diff)

    # make sure no distances are larger than maximum allowed cutoff
    if cutoff is not None:
        D = np.minimum(D, cutoff)

    return D

def hausdorff(A, B, cutoff=None, weights=None):
    """
    Symmetric Hausdorff (distance between two sets) using maximum pairwise distance and axis weights
    :param A: numpy array where rows are object positions
    :param B: numpy array where rows are object positions
    :param cutoff: maximum distance allowed from a to b
    :param weights: weighting factors for each dimension (columns of A,B) used for distance computation
    :return:
    """
    numA = shape2(A, axis=0)
    numB = shape2(B, axis=0)

    D = pairwise_distance(
        A, B,
        weights=weights,
        cutoff=cutoff)

    # minimum values along axis=0 (A) -> for each b, what is distance to closest a
    minBA = np.amin(D, axis=0)
    # minimum values along axis=1 (B) -> for each a, what is distance to closest b
    minAB = np.amin(D, axis=1)

    H_AB = np.sqrt((1 / numA) * np.sum(minAB ** 2))
    H_BA = np.sqrt((1 / numB) * np.sum(minBA ** 2))

    return (H_AB + H_BA)/2

def max_matching_size(positionsA, positionsB, weights=None, cutoff=None, show_plot=False):
    """
    Compute cardinality of a maximum matching on the bipartite graph where
    edges are based on distances between positions from 2 sets.
    :param positionsA: numpy array, rows are position vectors
    :param positionsB: numpy array, rows are position vectors
    :param weights: vector of weighting factors used to compute distance
    :param cutoff: cutoff distance for considering positions connected
    :param show_plot: show a plot of the graph
    :return:
    """
    # create a mask of 1's and 0s indicating which peak pairs are within cutoff
    D = pairwise_distance(
        positionsA,
        positionsB,
        weights=weights,
        cutoff=cutoff)

    # find all inject : recover peak pairs within cutoff distance
    # P is a tuple of 2 arrays where item i from first array is paired with item i from second
    P = np.where(D < cutoff)

    # need unique names for the graph vertices => prepend "I" and "R"
    I = ['I' + str(p) for p in P[0]]
    R = ['R' + str(p) for p in P[1]]

    IR_pairs = zip(I, R)

    G = nx.Graph()
    G.add_nodes_from(I, bipartite=0)
    G.add_nodes_from(R, bipartite=1)
    G.add_edges_from(list(IR_pairs))
    if show_plot:
        nx.draw(G, with_labels=True)

    if not nx.is_bipartite(G):
        raise ValueError('graph is not bipartite')

    # maximum matching does not work on disconnected graph
    # => iterate through each connected subgraph and tally the maximum matchings
    # graphs = list(nx.connected_component_subgraphs(G))  # <- deprecated
    graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    total_edges_in_max_matching = 0
    for g in graphs:
        # maximum matching seems to be directional, so each edge is counted twice
        MM = nx.bipartite.maximum_matching(g)
        total_edges_in_max_matching += int(len(MM) / 2)

    return total_edges_in_max_matching

def intensity_linearity_func(inject_HEIGHT, recover_HEIGHT):
    # get peak positions in PPM (data as numpy arrays)
    inject_HEIGHT = inject_HEIGHT.reshape(-1)
    recover_HEIGHT = recover_HEIGHT.reshape(-1)

    if len(inject_HEIGHT) != len(recover_HEIGHT):
        # intensity linearity should use peaks in noise-free regions
        # missing peaks should not happen
        return 0

    score = corrcoef(inject_HEIGHT, recover_HEIGHT)[0,1]
    return score

def nuscon_metrics(Ideal_Spec, Rec_Spec, Ideal_PeaksPosition, Rec_PeaksPosition, cutoff):
    intensity_linearity = intensity_linearity_func(Ideal_Spec, Rec_Spec)

    hausdorff_distance = hausdorff(Rec_PeaksPosition, Ideal_PeaksPosition, cutoff=cutoff)
    freq_accuracy = 1 - hausdorff_distance / cutoff

    total_edges_in_max_matching = max_matching_size(Ideal_PeaksPosition, Rec_PeaksPosition, cutoff=cutoff)
    true_positive_rate = total_edges_in_max_matching / Ideal_PeaksPosition.shape[0]
    # false_positive_rate = total_edges_in_max_matching / Rec_PeaksPosition.shape[0]
    false_positive_rate = (Rec_PeaksPosition.shape[0] - total_edges_in_max_matching) / Rec_PeaksPosition.shape[0]

    return freq_accuracy, intensity_linearity, true_positive_rate, false_positive_rate