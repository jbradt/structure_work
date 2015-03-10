"""
PHY 981 Project Code
====================

This code implements the basic pairing model required for the project.

Main Functions
--------------
sp_states
    Generator to create the single-particle levels
slater
    Lists the available Slater determinants
find_hamiltonian_matrix
    Finds the Hamiltonian matrix
find_pairing_hamiltonian_eigenvalues
    Does all of the above, and then diagonalizes to find the eigenvalues

Examples
--------
Find the single-particle states for two doubly-degenerate levels:

    >>> list(sp_states(2, 0.5))
    [(1, -0.5), (1, 0.5), (2, -0.5), (2, 0.5)]

Find the Slater determinants for this same case with two particles:

    >>> slater(2, list(sp_states(2, 0.5)), 0, pairs_only=True)
    [[0, 1], [2, 3]]

Calculate the eigenvalues for 4 particles in 4 levels, restricting to only pairs, and letting g=0.
This does the whole calculation in one step.

    >>> find_pairing_hamiltonian_eigenvalues(4, 4, 0, pairs_only=True, g=0, xi=1)
    array([  2.,   4.,   6.,   6.,   8.,  10.])

"""

import numpy as np
from itertools import combinations
from functools import wraps


def numpyize(func):
    """Decorator that converts all array-like arguments to NumPy ndarrays.

    Parameters
    ----------
    func : function(*args, **kwargs)
        The function to be decorated. Any positional arguments that are non-scalar
        will be converted to an ndarray

    Returns
    -------
    decorated : function(*newargs, **kwargs)
        The decorated function, with all array-like arguments being ndarrays

    """
    @wraps(func)
    def decorated(*args, **kwargs):
        newargs = list(args)
        for i, a in enumerate(newargs):
            if not np.isscalar(a):
                newargs[i] = np.asanyarray(a)
        return func(*newargs, **kwargs)

    return decorated


def merge_sort(a):
    """Sort the given list, counting the number of inversions necessary to do so.

    This implements a top-down, recursive merge sort. Comparison is done using the
    less-than operator (<).

    The sorting algorithm was taken from pseudocode on Wikipedia [1]_, but some modifications were made to count the
    permutations.

    Parameters
    ----------
    a : list
        The unsorted array

    Returns
    -------
    inv : int
        The number of inversions required to sort `a`
    res : list
        The sorted version of `a`

    References
    ----------
    .. [1] http://en.wikipedia.org/w/index.php?title=Merge_sort&oldid=649955073#Top-down_implementation_using_lists
    """

    if len(a) <= 1:
        # A list of length 0 or 1 is trivially sorted
        return 0, a

    # Divide the list into two halves
    n = len(a) // 2
    l = a[:n]
    r = a[n:]

    # Recursively sort each half
    linv, l = merge_sort(l)
    rinv, r = merge_sort(r)

    # Now merge the two sorted halves together
    inv = linv + rinv
    res = []

    while len(l) > 0 and len(r) > 0:
        if l[0] < r[0]:
            res.append(l.pop(0))
        else:
            # Inversions correspond to taking an element from r before l is empty. Each time this happens,
            # an element from r must pass by len(l) elements from l.
            res.append(r.pop(0))
            inv += len(l)

    # Take care of elements left over in one list after the other is empty
    if len(l) > 0:
        res += l
    if len(r) > 0:
        res += r

    return inv, res


def sp_states(pmax, spin):
    """Generates the single-particle states

    The states yielded have a quantum number p and spin s where 0 <= p <= pmax and -spin <= s <= spin.

    Parameters
    ----------
    pmax : int
        The maximum of the p quantum number
    spin : float
        The magnitude of the spin, e.g. 0.5

    Yields
    ------
    p : int
        The p quantum number
    s : float
        The projection of the spin
    """
    assert pmax >= 1, 'p levels run over [1,inf)'

    for p in range(1, pmax+1):
        s = -spin
        while s <= spin:
            yield p, s
            s += 1


def load_interaction(filename):
    """Reads information about an interaction from the provided file.

    The file is assumed to contain both matrix elements and single-particle states, each on a single line by itself.
    The single-particle states are assumed to have 6 numbers per line, representing index, n, l, 2j, 2m_j, and Energy,
    in that order. The matrix elements are assumed to have 5 numbers per line, representing the four single-particle
    states and the value of the matrix element.

    Lines beginning with '#' are ignored.

    Parameters
    ----------
    filename : string
        The name of the file to be read

    Returns
    -------
    states : list
        The single-particle states
    mels : list
        The matrix elements
    """
    states = []
    mels = []
    with open(filename) as inter:
        for line in inter:
            if line[0] == '#':
                continue
            parts = [float(x) if '.' in x else int(x) for x in line.split()]
            if len(parts) == 1:
                continue
            elif len(parts) == 6:
                states.append(parts)
            elif len(parts) == 5:
                mels.append(parts)
    return states, mels


@numpyize
def slater(n_particles, states, total_m):
    """Finds the possible slater determinants with a given total M.

    Parameters
    ----------
    n_particles : int
        The number of (single) particles
    states : array-like
        A list of the available single-particle states.
        Format: [index, n, l, 2j, 2m_j, energy]
    total_m : int or float
        The total spin projection desired

    Returns
    -------
    sds : list
        The possible Slater determinants, as integers. Each element is an integer whose bits represent
        the occupied and unoccupied states.
    """
    indices = range(len(states))

    sds = []
    for x in combinations(indices, n_particles):
        x = np.array(x)
        s = states[x]
        m = s[:, 4].sum() / 2
        if total_m == m:
            sds.append(np.sum(2**x))

    return sds


def sd_delta(a, b):
    """Counts the number of different single-particle states between two slater determinants.

    Parameters
    ----------
    a : array-like
        The first Slater determinant
    b : array-like
        The second Slater determinant
    """
    d = 0
    for el in a:
        if el not in b:
            d += 1
    return d


def destroy(i, ket):
    """The annihilation operator.

    Parameters
    ----------
    i : int
        The state to be destroyed
    ket : list
        The Slater determinant to act on

    Returns
    -------
    new_ket : list
        The Slater determinant after the action of the operator

    Raises
    ------
    ValueError
        If state `i` is not present in `ket`
    """
    if i in ket:
        new_ket = ket.copy()
        new_ket.remove(i)
        return new_ket
    else:
        raise ValueError('destroying missing state')


def create(i, ket):
    """The creation operator.

    Parameters
    ----------
    i : int
        The state to be created
    ket : list
        The Slater determinant to act on

    Returns
    -------
    new_ket : list
        The Slater determinant after the action of the operator

    Raises
    ------
    ValueError
        If state `i` is already present in `ket`. This preserves the
        Pauli principle.
    """
    if i not in ket:
        new_ket = ket.copy()
        new_ket.insert(0, i)
        return new_ket
    else:
        raise ValueError('creating state which is already present')


def state_iterator(states):
    """Generator for iterating over single-particle states.

    This can be used to implement the sum in the two-particle operator,
    i.e. the sum over p<q and r<s for (p, q, r, s) in the single-particle states.

    Parameters
    ----------
    states : iterable
        The single-particle states

    Yields
    ------
    (p, q, r, s) : tuple
        The indices for the sum
    """
    for q, st1 in enumerate(states):
        for p in range(q):
            for s, st2 in enumerate(states):
                for r in range(s):
                    yield p, q, r, s


def pairing_hamiltonian(ket, sds, states, xi=1, g=1):
    """Creates a column of the pairing Hamiltonian matrix.

    This acts the Hamiltonian on the provided Slater determinant and produces
    a list containing the coefficients of each Slater determinant in the basis
    for the result.

    Parameters
    ----------
    ket : list
        The Slater determinant to calculate the column for
    sds : list
        The possible Slater determinants
    states : list
        The possible single-particle states
    xi : float, optional
        The spacing between single-particle states
    g : float, optional
        The pairing interaction strength

    Returns
    -------
    col : list
        The column of the Hamiltonian matrix
    """

    assert ket in sds, 'ket missing from possible SDs'
    col = [0.0] * len(sds)

    for p, q, r, s in state_iterator(states):
        if (states[p][0] != states[q][0] or states[r][0] != states[s][0]
                or states[p][1] != -states[q][1] or states[r][1] != -states[s][1]):
            # This imposes the pairing restriction
            continue

        if r not in ket or s not in ket:
            continue

        if p in ket and (p != r and p != s):
            continue

        if q in ket and (q != r and q != s):
            continue

        try:
            new_ket = ket.copy()
            new_ket = destroy(r, new_ket)
            new_ket = destroy(s, new_ket)
            new_ket = create(q, new_ket)
            new_ket = create(p, new_ket)
        except ValueError:
            continue

        inv, sorted_ket = merge_sort(new_ket)
        try:
            i = sds.index(sorted_ket)
            col[i] += -g * (-1)**inv

        except ValueError:
            # Not in the list of SDs
            continue

    i = sds.index(ket)
    spl = map(lambda x: xi * (states[x][0] - 1), ket)
    col[i] += sum(spl)

    return col


def find_hamiltonian_matrix(sds, states, **kwargs):
    """Finds the Hamiltonian matrix.

    Parameters
    ----------
    sds : list
        The possible Slater determinants
    states : list
        The possible single-particle states
    **kwargs
        Additional arguments to be passed on to the Hamiltonian function

    Returns
    -------
    hmat : ndarray
        The Hamiltonian matrix
    """

    n = np.size(sds, 0)
    hmat = np.zeros((n, n))

    for j in range(n):
        hmat[:, j] = pairing_hamiltonian(sds[j], sds, states, **kwargs)

    return hmat


def find_pairing_hamiltonian_eigenvalues(nparticles, pmax, total_m, pairs_only=False, **kwargs):
    """Find the eigenvalues of the pairing Hamiltonian matrix.

    This function just wraps all of the above up into one package.

    Parameters
    ----------
    nparticles : int
        The number of particles
    pmax : int
        The number of single-particle levels, or the p value of the highest level
    total_m : int
        The total M of the allowed Slater determinants
    pairs_only : bool, optional
        Whether to restrict the calculation to pairs only
    **kwargs
        The remaining arguments are passed to the Hamiltonian function

    Returns
    -------
    evs : ndarray
        A list of the eigenvalues, which might not be sorted.
    """
    states = np.array(list(sp_states(pmax, 0.5)))
    dets = slater(nparticles, states, total_m, pairs_only)
    hmat = find_hamiltonian_matrix(dets, states, **kwargs)
    evs = np.linalg.eigvalsh(hmat)
    return evs

if __name__ == '__main__':
    sps, mel = load_interaction('usdb.txt')
    sds = slater(2, sps, total_m=3)
    print(sds)
    print(len(sds), 'slater determinants')