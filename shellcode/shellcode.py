"""
PHY 981 Final Project Code
==========================

This code finds the energy levels in the sd shell, as directed in the assignment
for final project version (a).

Main Functions
--------------
load_interaction
    Reads an interaction and single-particle states from the given file.
slater
    Finds the list of possible Slater determinants with the given total M.
find_hamiltonian_matrix
    Finds the Hamiltonian matrix.
find_eigenvalues
    Does all of the above, and also diagonalizes the matrix to find the
    eigenvalues.

"""

import numpy as np
from itertools import combinations
from functools import wraps
from sys import argv

# Get the directory the package is stored in
import os
package_dir = os.path.dirname(os.path.abspath(__file__))

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


@numpyize
def is_hermitian(mat):
    """Checks if the given matrix is Hermitian.

    Parameters
    ----------
    mat : array-like
        A matrix to be checked

    Returns
    -------
    bool
        True if the matrix is Hermitian
    """

    sh = mat.shape
    if len(sh) != 2:
        raise ValueError('must be a 2-D array')
    if sh[0] != sh[1]:
        raise ValueError('must be a square matrix')

    if np.allclose(mat.T, mat):
        return True
    else:
        return False


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
    mels = {}
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
                indices = tuple(map(lambda x: x - 1, parts[0:4]))
                mels[indices] = parts[4]
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
        The possible Slater determinants, as lists of indices.
    """
    indices = range(len(states))

    sds = []
    for x in combinations(indices, n_particles):
        x = np.array(x)
        s = states[x]
        m = s[:, 4].sum() / 2
        if total_m == m:
            sds.append(x.tolist())

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


def find_hamiltonian_matrix(sds, states, inter):
    """Finds the Hamiltonian matrix.

    Parameters
    ----------
    sds : list
        The possible Slater determinants
    states : list
        The possible single-particle states
    inter : dict
        The interaction matrix elements, given as a dictionary mapping {(p, q, r, s): energy}.

    Returns
    -------
    hmat : ndarray
        The Hamiltonian matrix
    """

    n = np.size(sds, 0)
    hmat = np.zeros((n, n))

    for i, ket in enumerate(sds):

        for (p, q, r, s), int_energy in inter.items():

            if r not in ket or s not in ket:
                continue

            if p in ket and (p != r and p != s):
                continue

            if q in ket and (q != r and q != s):
                continue

            new_ket = ket.copy()
            new_ket.remove(r)
            new_ket.remove(s)
            new_ket.insert(0, q)
            new_ket.insert(0, p)

            inv, sorted_ket = merge_sort(new_ket)

            try:
                j = sds.index(sorted_ket)

            except ValueError:
                # Not in the list of SDs
                continue

            hmat[i, j] += int_energy * (-1)**inv
            hmat[j, i] = hmat[i, j]

    states = np.asanyarray(states)
    for i, ket in enumerate(sds):
        hmat[i, i] += states[ket, -1].sum()

    return hmat


def find_eigenvalues(num_particles, total_2m):
    """Calculates the energy levels for the given number of particles and total M.

    This performs the entire calculation. It finds the Slater determinants, calculates the Hamiltonian matrix, and
    diagonalizes it to find the energies.

    Parameters
    ----------
    num_particles : int
        The number of particles
    total_2m : float
        The total spin projection, multiplied by two.
    """

    sps, mel = load_interaction(os.path.join(package_dir, 'usdb.txt'))
    sds = slater(num_particles, sps, total_2m / 2)
    print('Found {} slater determinants:'.format(len(sds)),
          sds, sep='\n')
    hc = find_hamiltonian_matrix(sds, sps, mel)
    print('The matrix was:', hc, sep='\n')
    assert is_hermitian(hc), 'the matrix is not Hermitian!'
    evs = np.linalg.eigvalsh(hc)
    print('The eigenvalues are:')
    print(evs)


if __name__ == '__main__':
    if len(argv) != 3:
        exit('Usage: python3 shellcode.py [num_particles] [2 * total_m]')

    int_args = [int(a) for a in argv[1:]]
    find_eigenvalues(num_particles=int_args[0], total_2m=int_args[1])