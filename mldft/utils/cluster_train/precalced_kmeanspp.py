# This is a slightly modified version of the KMeans++ initialization algorithm from sklearn:https://github.com/scikit-learn/scikit-learn/blob/ea9e824705cc6313aa65413e9ee245fa974f8dd6/sklearn/cluster/_kmeans.py
# using a precalculated distance matrix
import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum


def kmeans_plusplus_dm(
    distance_matrix,
    n_clusters,
    *,
    random_state=None,
    n_local_trials=None,
):
    """Init n_clusters seeds according to k-means++ using a precalculated distance matrix.

    Parameters
    ----------
    distance_matrix : {ndarray, sparse matrix} of shape (n_samples, n_samples)
        The distance matrix to pick seeds for. The distance matrix is
        assumed to be symmetric and the diagonal is assumed to be zero.

    n_clusters : int
        The number of centroids to initialize.

    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.

    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)) which is the recommended setting.
        Setting to 1 disables the greedy cluster selection and recovers the
        vanilla k-means++ algorithm which was empirically shown to work less
        well than its greedy variant.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Examples
    --------

    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  2],
           [ 1,  0]])
    >>> indices
    array([3, 2])
    """
    # Check data
    check_array(distance_matrix, accept_sparse="csr", dtype=[np.float64, np.float32])

    if distance_matrix.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={distance_matrix.shape[0]} should be >= n_clusters={n_clusters}."
        )
    if not isinstance(random_state, np.random.Generator):
        random_state = check_random_state(random_state)

    # Call private k-means++
    indices = _kmeans_plusplus(n_clusters, distance_matrix, random_state, n_local_trials)

    return indices


def _kmeans_plusplus(n_clusters, distance_matrix, random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by k-means++. Prior validation of
    data is assumed.

    Parameters
    ----------

    n_clusters : int
        The number of seeds to choose.#

    distance_matrix : {ndarray, sparse matrix} of shape (n_samples, n_samples)
        The distance matrix to pick seeds for. The distance matrix is
        assumed to be symmetric and the diagonal is assumed to be zero.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, _ = distance_matrix.shape
    sample_weight = np.ones(n_samples, dtype=np.float64)
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.choice(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)

    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = distance_matrix[center_id]
    current_pot = closest_dist_sq @ sample_weight

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = distance_matrix[candidate_ids]

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ np.ones(n_samples).reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries

        indices[c] = best_candidate

    return indices
