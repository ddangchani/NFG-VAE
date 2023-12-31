import numpy as np
import math
import networkx as nx
import scipy.linalg as la
import scipy.sparse as sp


# 1. Generating Random Graph

def generate_random_dag(d, degree, w_range = (0.5, 2.0), seed = 0):
    """
    Generate a random Erdos-Renyi DAG with some parameters

    Args:
        d: number of nodes
        degree: expected node degree, in + out
        w_range: range of weights
        seed: random seed

    Returns:
        G: nx.DiGraph object
    """
    np.random.seed(seed)

    prob = float(degree) / (d - 1)
    B = np.tril((np.random.rand(d, d) < prob).astype(float), k = -1)

    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)

    return G

# 2. Generating Linear SEM with independent noise

def generate_linear_sem(graph : nx.DiGraph, 
                        n : int, dist = 'normal', linear_type = 'linear', 
                        loc = 0.0, scale = 1.0, seed = 0, x_dims : int = 1):

    """
    Generate a linear SEM with some parameters

    Args:
        graph: nx.DiGraph object
        n: number of samples
        dist: distribution of noise
        linear_type: type of linear function
        loc: mean of noise
        scale: scale of noise
        seed: random seed
    """

    np.random.seed(seed)

    A = nx.to_numpy_array(graph) # adjacency matrix
    m = A.shape[0] # number of nodes
    X = np.zeros([n, m, x_dims]) # data matrix
    ordered_nodes = list(nx.topological_sort(graph)) # topological order of nodes

    # generate data
    for i in ordered_nodes:
        parents = list(graph.predecessors(i))
        if linear_type == 'linear':
            eta = np.dot(X[:, parents, 0], A[parents, i])
        elif linear_type == 'trigonometric':
            eta = np.sin(np.dot(X[:, parents, 0], A[parents, i]))
        elif linear_type == 'quadratic':
            eta = np.dot(X[:, parents, 0], A[parents, i]) ** 2
        else:
            raise ValueError('Unknown linear type')

        # generate noise
        if dist == 'normal':
            noise = np.random.normal(loc = loc, scale = scale, size = n)
        elif dist == 'uniform':
            noise = np.random.uniform(low = loc-scale, high = loc+scale, size = n)
        elif dist == 'exponential':
            noise = np.random.exponential(scale = scale, size = n)
        elif dist == 'laplace':
            noise = np.random.laplace(loc = loc, scale = scale, size = n)
        elif dist == 'gumbel':
            noise = np.random.gumbel(loc = loc, scale = scale, size = n)
        else:
            raise ValueError('Invalid distribution')

        X[:, i, 0] = eta.flatten() + noise
    if x_dims > 1 :
        for i in range(x_dims-1):
            X[:, :, i+1] = np.random.normal(scale=scale, size=1)*X[:, :, 0] + np.random.normal(scale=scale, size=1) + np.random.normal(scale=scale, size=(n, m))
        X[:, :, 0] = np.random.normal(scale=scale, size=1) * X[:, :, 0] + np.random.normal(scale=scale, size=1) + np.random.normal(scale=scale, size=(n, m))
    return X


# 3. Generating Linear SEM with correlated noise structure

def generate_linear_sem_correlated(graph : nx.DiGraph,
                                 n : int, prop : float, seed = 0, return_cov = False, const = 1.5, return_graph = False, x_dims : int = 1):

    """
    Generate a linear SEM with noise dependence structure on given proportion of edges
    Noise is generated from a multivariate normal distribution with a generated correlation matrix

    Args:
        graph: nx.DiGraph object
        n: number of samples
        prop: proportion of edges that have correlated noise structure
        seed: random seed
        return_cov: whether to return the covariance matrix (default: False)
        return_graph: whether to return the graph (default: False)
        const: constant to adjust covariance matrix to make it positive semi-definite (default: 1.5)
    """

    np.random.seed(seed)

    A = nx.to_numpy_array(graph) # adjacency matrix
    m = A.shape[0] # number of nodes
    e = graph.number_of_edges() # number of edges
    X = np.zeros([n, m, x_dims]) # data matrix
    ordered_nodes = list(nx.topological_sort(graph)) # topological order of nodes

    assert prop >= 0 and prop <= 1, 'Proportion of correlated noise must be between 0 and 1'

    num_corr = int(e * prop) # number of edges that have correlated noise structure
    num_uncorr = e - num_corr # number of edges that have uncorrelated noise structure

    # selected edges
    selected_edge_indices = np.random.choice(np.arange(e), size = num_corr, replace=False)
    selected_edges = np.array(graph.edges())[selected_edge_indices]

    if return_graph:
        # reverse selected edges
        selected_edges = np.flip(selected_edges, axis = 1)
        graph_new = nx.MultiDiGraph(graph)
        graph_new.add_edges_from(selected_edges)


    # generate covaraince matrix
    cov = np.eye(m)

    for edge in selected_edges:
        r = np.random.uniform(low = -1.0, high = 1.0)
        cov[edge[0], edge[1]] = r
        cov[edge[1], edge[0]] = r

    cov_prev = cov.copy()

    # make covariance matrix p.s.d and normalize to unit variance
    cov = adjust_cov_matrix(cov, const)
    cov = normalize_cov_matrix(cov)

    # generate noise
    noise = np.random.multivariate_normal(mean=np.zeros(m), cov=cov, size=n) 

    # generate data
    for i in ordered_nodes:
        parents = list(graph.predecessors(i))
        eta = np.dot(X[:, parents, 0], A[parents, i])
        X[:, i, 0] = eta.flatten() + noise[:, i]

    if return_cov and return_graph:
        return X, cov, cov_prev, graph_new
    elif return_cov and not return_graph:
        return X, cov, cov_prev
    elif not return_cov and return_graph:
        return X, graph_new
    else:
        return X
    


# Generating correlation matrix
def generate_correlation_matrix(size, seed=0):
    np.random.seed(seed)
    A = np.random.uniform(-1, 1, (size, size))
    symmetric_A = (A + A.T) / 2

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_A)

    # Make all eigenvalues positive
    eigenvalues = np.abs(eigenvalues)

    # Construct new symmetric matrix
    symmetric_A = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Normalize each row to make it a correlation matrix
    row_norms = np.sqrt(np.diag(symmetric_A))
    correlation_matrix = symmetric_A / np.outer(row_norms, row_norms)

    return correlation_matrix

def is_positive_semidefinite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)

def adjust_cov_matrix(cov, const=1.5):
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig < 0:
        cov -= const * min_eig * np.eye(*cov.shape)
    return cov

def normalize_cov_matrix(cov):
    std_devs = np.sqrt(np.diag(cov))
    cov = cov / np.outer(std_devs, std_devs)
    return cov