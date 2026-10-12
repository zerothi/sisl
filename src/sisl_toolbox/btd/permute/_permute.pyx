# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Graph routines for bandwidth reduction of sparse (square) patterns.

All routines operate on a symmetric CSR graph ``(ptr, col)`` without
self-loops, as returned by `symmetric_graph`.

Common to all routines is the ``level`` work-array, which signals which nodes
are eligible for a breadth-first search: a node is only visited when
``level[node] == -1``. Nodes that are already ordered are marked with a value
different from -1, which guarantees that a search never crosses into an already
ordered component.
"""
cimport cython
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc, realloc

import numpy as np

__all__ = [
    "symmetric_graph",
    "cuthill_mckee",
    "gibbs_poole_stockmeyer",
    "generalized_gibbs_poole_stockmeyer",
    "electrode_ordering",
    "refine_bandwidth",
    "btd_blocks",
]

ctypedef fused key_st:
    int
    long long


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def symmetric_graph(const int[::1] ptr, const int[::1] ncol, const int[::1] col):
    r"""Pattern of :math:`A + A^T` without the diagonal

    Parameters
    ----------
    ptr, ncol, col :
        CSR pattern of a square matrix (``ptr`` need not be contiguous)

    Returns
    -------
    gptr, gcol :
        contiguous CSR pattern of the symmetric graph (columns are unique, but unsorted)
    """
    cdef int n = ncol.shape[0]
    cdef int i, j, ind
    cdef Py_ssize_t nnz = 0

    for i in range(n):
        nnz += ncol[i]

    # Transposed pattern
    TPTR = np.zeros(n + 1, dtype=np.int32)
    TCOL = np.empty(nnz, dtype=np.int32)
    cdef int[::1] tptr = TPTR
    cdef int[::1] tcol = TCOL
    for i in range(n):
        for ind in range(ptr[i], ptr[i] + ncol[i]):
            tptr[col[ind] + 1] += 1
    for i in range(n):
        tptr[i + 1] += tptr[i]
    for i in range(n):
        for ind in range(ptr[i], ptr[i] + ncol[i]):
            j = col[ind]
            tcol[tptr[j]] = i
            tptr[j] += 1
    # shift back
    for i in range(n, 0, -1):
        tptr[i] = tptr[i - 1]
    tptr[0] = 0

    # Merge rows of A and A^T, `mark` removes duplicates
    GPTR = np.empty(n + 1, dtype=np.int32)
    GCOL = np.empty(2 * nnz, dtype=np.int32)
    MARK = np.full(n, -1, dtype=np.int32)
    cdef int[::1] gptr = GPTR
    cdef int[::1] gcol = GCOL
    cdef int[::1] mark = MARK
    cdef int ng = 0

    gptr[0] = 0
    for i in range(n):
        mark[i] = i
        for ind in range(ptr[i], ptr[i] + ncol[i]):
            j = col[ind]
            if mark[j] != i:
                mark[j] = i
                gcol[ng] = j
                ng += 1
        for ind in range(tptr[i], tptr[i + 1]):
            j = tcol[ind]
            if mark[j] != i:
                mark[j] = i
                gcol[ng] = j
                ng += 1
        gptr[i + 1] = ng

    return GPTR, GCOL[:ng].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _sort_key(int[::1] a, const int lo, const int hi, const key_st[::1] key) noexcept nogil:
    """Insertion sort of ``a[lo:hi]`` by increasing ``key[a]`` (stable)"""
    cdef int i, j, x
    for i in range(lo + 1, hi):
        x = a[i]
        j = i - 1
        while j >= lo and key[a[j]] > key[x]:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _levels(const int[::1] ptr, const int[::1] col,
                 int[::1] level, int[::1] order, const int off, int tail, int[::1] lptr) noexcept nogil:
    """Level structure of all eligible nodes (``level == -1``) connected to the seeds ``order[off:tail]``

    The seeds must have ``level == 0``. The visited nodes are stored in ``order[off:lptr[nl]]``
    with level ``l`` being ``order[lptr[l]:lptr[l+1]]``, and ``level[node]`` is the level of each node.

    Returns
    -------
    nl : number of levels
    """
    cdef int nl = 0
    cdef int head = off
    cdef int i, j, ind, end

    lptr[0] = off
    while head < tail:
        end = tail
        nl += 1
        while head < end:
            i = order[head]
            head += 1
            for ind in range(ptr[i], ptr[i + 1]):
                j = col[ind]
                if level[j] == -1:
                    level[j] = nl
                    order[tail] = j
                    tail += 1
        lptr[nl] = tail
    return nl


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int _bfs(const int[::1] ptr, const int[::1] col, const int root,
                     int[::1] level, int[::1] order, const int off, int[::1] lptr) noexcept nogil:
    """Rooted level structure of `root`, see `_levels`"""
    order[off] = root
    level[root] = 0
    return _levels(ptr, col, level, order, off, off + 1, lptr)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int _width(const int[::1] lptr, const int nl) noexcept nogil:
    cdef int l, w = 0
    for l in range(nl):
        if lptr[l + 1] - lptr[l] > w:
            w = lptr[l + 1] - lptr[l]
    return w


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _reset(int[::1] level, const int[::1] order, const int lo, const int hi) noexcept nogil:
    cdef int i
    for i in range(lo, hi):
        level[order[i]] = -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef (int, int) _endpoints(const int[::1] ptr, const int[::1] col, const int[::1] deg, int v,
                           int[::1] level, int[::1] order, int[::1] lptr,
                           int[::1] order2, int[::1] lptr2) noexcept nogil:
    """Pseudo-diameter end-points ``(v, u)`` of the component containing `v`

    Gibbs-Poole-Stockmeyer algorithm I, with the candidate shrinking of
    Lewis (one candidate per degree in the last level).
    `level` must be -1 for the component, and is -1 on return.
    """
    cdef int nl, nl2, c, w, u, wmin, prev, k
    cdef bint deeper = True

    nl = _bfs(ptr, col, v, level, order, 0, lptr)
    _reset(level, order, 0, lptr[nl])
    while deeper:
        deeper = False
        u = v
        wmin = lptr[nl] + 1
        prev = -1
        _sort_key(order, lptr[nl - 1], lptr[nl], deg)
        for k in range(lptr[nl - 1], lptr[nl]):
            c = order[k]
            if deg[c] == prev:
                continue
            prev = deg[c]
            nl2 = _bfs(ptr, col, c, level, order2, 0, lptr2)
            _reset(level, order2, 0, lptr2[nl2])
            if nl2 > nl:
                v = c
                nl = _bfs(ptr, col, v, level, order, 0, lptr)
                _reset(level, order, 0, lptr[nl])
                deeper = True
                break
            w = _width(lptr2, nl2)
            if w < wmin:
                wmin = w
                u = c
    return v, u


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def cuthill_mckee(const int[::1] ptr, const int[::1] col):
    """Cuthill-McKee ordering of a symmetric graph

    Each connected component is started at a pseudo-peripheral node.

    Parameters
    ----------
    ptr, col :
        symmetric CSR graph without self-loops, see `symmetric_graph`

    Returns
    -------
    perm : the permutation table, ``perm[new] = old``
    """
    cdef int n = ptr.shape[0] - 1
    DEG = np.diff(ptr).astype(np.int32)
    # seeds by increasing degree, each component starts at its minimum degree node
    SEEDS = np.argsort(DEG, kind="stable").astype(np.int32)
    PERM = np.empty(n, dtype=np.int32)
    LEVEL = np.full(n, -1, dtype=np.int32)
    ORDER = np.empty(n, dtype=np.int32)
    ORDER2 = np.empty(n, dtype=np.int32)
    LPTR = np.empty(n + 1, dtype=np.int32)
    LPTR2 = np.empty(n + 1, dtype=np.int32)
    cdef int[::1] deg = DEG
    cdef int[::1] seeds = SEEDS
    cdef int[::1] perm = PERM
    cdef int[::1] level = LEVEL
    cdef int[::1] order = ORDER
    cdef int[::1] order2 = ORDER2
    cdef int[::1] lptr = LPTR
    cdef int[::1] lptr2 = LPTR2
    cdef int s, v, u, i, x, j, ind, head, start
    cdef int nperm = 0

    with nogil:
        for i in range(n):
            s = seeds[i]
            if level[s] != -1:
                continue
            v, u = _endpoints(ptr, col, deg, s, level, order, lptr, order2, lptr2)

            # breadth-first numbering, neighbors by increasing degree
            perm[nperm] = v
            level[v] = 0
            head = nperm
            nperm += 1
            while head < nperm:
                x = perm[head]
                head += 1
                start = nperm
                for ind in range(ptr[x], ptr[x + 1]):
                    j = col[ind]
                    if level[j] == -1:
                        level[j] = 0
                        perm[nperm] = j
                        nperm += 1
                _sort_key(perm, start, nperm, deg)

    return PERM


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _number(const int[::1] ptr, const int[::1] col, const long long[::1] key,
                 const int[::1] lvl, const int nl, const int[::1] order, const int[::1] lptr,
                 int[::1] level, int[::1] perm, int lstart, int nperm) noexcept nogil:
    """Level-by-level numbering of a level structure (GPS algorithm III)

    ``order[lptr[l]:lptr[l+1]]`` are the nodes of level ``l`` by increasing `key`,
    and ``lvl[node]`` is the level of each node.
    The numbering starts at ``perm[lstart]``, which may already hold a start node
    in level 0 (then ``nperm == lstart + 1``). Unnumbered nodes have ``level == -1``.

    Returns
    -------
    nperm : the updated number of numbered nodes
    """
    cdef int i, j, x, ind, start, cur, lend

    for j in range(nl):
        cur = lptr[j]
        i = lstart
        while True:
            # number unnumbered neighbors in the same level
            while i < nperm:
                start = nperm
                for ind in range(ptr[perm[i]], ptr[perm[i] + 1]):
                    x = col[ind]
                    if lvl[x] == j and level[x] == -1:
                        level[x] = 0
                        perm[nperm] = x
                        nperm += 1
                _sort_key(perm, start, nperm, key)
                i += 1
            if nperm - lstart == lptr[j + 1] - lptr[j]:
                break
            # restart from the unnumbered node with the smallest key
            while level[order[cur]] != -1:
                cur += 1
            x = order[cur]
            level[x] = 0
            perm[nperm] = x
            nperm += 1

        # number neighbors in the next level
        lend = nperm
        for i in range(lstart, lend):
            start = nperm
            for ind in range(ptr[perm[i]], ptr[perm[i] + 1]):
                x = col[ind]
                if lvl[x] == j + 1 and level[x] == -1:
                    level[x] = 0
                    perm[nperm] = x
                    nperm += 1
            _sort_key(perm, start, nperm, key)
        lstart = lend

    return nperm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _gps(const int[::1] ptr, const int[::1] col, bint generalized, max_roots, vset=None, uset=None):
    """Shared implementation of `gibbs_poole_stockmeyer`, `generalized_gibbs_poole_stockmeyer`
    and `electrode_ordering`

    GPS is the special case of GGPS with one root at each end, numbering by degree,
    and no final reversal.
    The electrode ordering roots the level structures at the sets `vset` and `uset`
    (unique nodes) for the component they belong to, the remaining components use GPS.
    """
    cdef int n = ptr.shape[0] - 1
    DEG = np.diff(ptr).astype(np.int32)
    # seeds by increasing degree, each component starts at its minimum degree node
    SEEDS = np.argsort(DEG, kind="stable").astype(np.int32)
    KEY = DEG.astype(np.int64)
    PERM = np.empty(n, dtype=np.int32)
    LEVEL = np.full(n, -1, dtype=np.int32)
    LVL = np.empty(n, dtype=np.int32)
    LOC = np.empty(n, dtype=np.int32)
    ORDER = np.empty(n, dtype=np.int32)
    ORDER2 = np.empty(n, dtype=np.int32)
    LPTR = np.empty(n + 1, dtype=np.int32)
    LPTR2 = np.empty(n + 1, dtype=np.int32)
    CNT = np.zeros(n + 1, dtype=np.int32)
    HCNT = np.zeros(n + 1, dtype=np.int32)
    cdef int[::1] deg = DEG
    cdef long long[::1] key = KEY
    cdef int[::1] perm = PERM
    cdef int[::1] level = LEVEL
    cdef int[::1] lvl = LVL
    cdef int[::1] loc = LOC
    cdef int[::1] order = ORDER
    cdef int[::1] order2 = ORDER2
    cdef int[::1] lptr = LPTR
    cdef int[::1] lptr2 = LPTR2
    cdef int[::1] cnt = CNT
    cdef int[::1] hcnt = HCNT
    cdef int[:, ::1] A
    cdef int s, v, u, i, t, x, a, ind, k, nl, nc, m, h, hmin, best, j1, lo, hi, start, s0
    cdef int nperm = 0
    cdef int p = 0
    cdef bint swapped
    # Legend of the variables, see [GPS1976]_ and [GGPS2009]_:
    #   s      : seed of the component (-1 for the component of the electrodes)
    #   v, u   : pseudo-peripheral nodes at the "v" and "u" ends
    #   roots  : all roots, the first p are at the "v" end, the rest at the "u" end
    #   m      : number of rooted level structures (root sets)
    #   nc, nl : number of nodes, and levels, in the current component
    #   A, AA  : associated level sets, A[t, i] is the level of order[i] in the t'th
    #            rooted structure (memoryview and array)
    #   k      : last level of the combined level structure (levels are [0, k])
    #   lvl    : level of each node in the combined level structure
    #   cnt    : number of nodes in each level of the combined structure
    #   hcnt   : per-level histogram of a connected part when placed on a structure
    #   h      : width of the combined structure when a connected part is placed,
    #            hmin is the smallest h and best the structure giving it
    #   j1     : structure selected for the largest connected part
    #   t, x, a: structure index, node and level (loop variables)
    #   start  : the node the numbering starts from, s0 its position in perm
    #   swapped: whether the levels are reversed for the numbering

    if generalized:
        # sum of the adjacent node codes (1-based codes, as in the paper)
        for i in range(n):
            key[i] = 0
            for ind in range(ptr[i], ptr[i + 1]):
                key[i] += col[ind] + 1
    if max_roots is None:
        max_roots = n

    seeds = SEEDS.tolist()
    if vset is not None:
        # -1 signals the component of the electrodes
        seeds.insert(0, -1)

    for s in seeds:
        if s >= 0 and level[s] != -1:
            continue

        if s < 0:
            root_sets = [vset] if uset is None else [vset, uset]
            roots = []
            p = 1
        else:
            root_sets = None
        # Phase I: pseudo-peripheral nodes, the roots at the "v" and "u" ends
        v = s
        while root_sets is None:
            v, u = _endpoints(ptr, col, deg, v, level, order, lptr, order2, lptr2)
            nl = _bfs(ptr, col, v, level, order, 0, lptr)
            nc = lptr[nl]
            _reset(level, order, 0, nc)
            vroots = [v]
            uroots = [u]
            if not generalized:
                break

            # "u" end: all nodes in the last level of L_v (all have depth >= nl)
            last = ORDER[lptr[nl - 1]:nc]
            last = last[np.argsort(DEG[last], kind="stable")]
            deeper = -1
            for x in last[last != u][:max_roots]:
                t = _bfs(ptr, col, x, level, order2, 0, lptr2)
                _reset(level, order2, 0, nc)
                if t > nl:
                    deeper = x
                    break
                uroots.append(x)
            if deeper >= 0:
                v = deeper
                continue

            # "v" end: nodes with the degree of v whose rooted structure has the same depth
            comp = ORDER[:nc]
            cands = comp[DEG[comp] == deg[v]]
            cands = cands[(cands != v) & ~np.isin(cands, uroots)]
            for x in cands[:max_roots]:
                t = _bfs(ptr, col, x, level, order2, 0, lptr2)
                _reset(level, order2, 0, nc)
                if t == nl:
                    vroots.append(x)
            break

        if root_sets is None:
            roots = vroots + uroots
            p = len(vroots)
            root_sets = [np.array([x], dtype=np.int32) for x in roots]
        m = len(root_sets)

        # the component is the nodes connected to the first root set
        R = root_sets[0]
        ORDER[:len(R)] = R
        LEVEL[R] = 0
        nl = _levels(ptr, col, level, order, 0, len(R), lptr)
        nc = lptr[nl]
        connected = uset is None or s >= 0 or (LEVEL[uset] != -1).all()
        _reset(level, order, 0, nc)
        if not connected:
            raise ValueError("electrode_ordering: start and end are not connected")

        # associated level sets, A[t, i] is the level of order[i] in the t'th
        # rooted structure, the "u" end structures are reversed
        LOC[ORDER[:nc]] = np.arange(nc, dtype=np.int32)
        AA = np.empty([m, nc], dtype=np.int32)
        width = np.empty(m, dtype=np.int32)
        depth = np.empty(m, dtype=np.int32)
        for t in range(m):
            R = root_sets[t]
            ORDER2[:len(R)] = R
            LEVEL[R] = 0
            nl = _levels(ptr, col, level, order2, 0, len(R), lptr2)
            AA[t] = LEVEL[ORDER[:nc]]
            _reset(level, order2, 0, lptr2[nl])
            if lptr2[nl] != nc:
                raise ValueError("electrode_ordering: start and end are not connected")
            depth[t] = nl
            width[t] = _width(lptr2, nl)
        # distance between the "v" and "u" ends, the electrodes act as single nodes
        if s < 0 and uset is not None:
            k = AA[0, LOC[uset]].min()
        else:
            k = depth.max() - 1
        AA[p:] = k - AA[p:]
        if s < 0:
            # the electrodes are fixed in the first and last levels, nodes
            # behind an electrode get levels outside [0, k]
            AA[:, LOC[vset]] = 0
            if uset is not None:
                AA[:, LOC[uset]] = k
            AA -= AA.min()
            k = AA.max()
        nl = k + 1
        # ties are broken by the first structure, i.e. the smallest width
        idx = np.argsort(width, kind="stable")
        AA = AA[idx]
        is_v = idx < p
        A = AA

        # Phase II: fix nodes with equal level sets, then place the remaining
        # connected parts (largest first) on the structure giving the smallest width
        for i in range(nl):
            cnt[i] = 0
        for i in range(nc):
            x = order[i]
            a = A[0, i]
            for t in range(1, m):
                if A[t, i] != a:
                    break
            else:
                lvl[x] = a
                cnt[a] += 1
                level[x] = -2
        # connected parts of the remaining nodes in order2[cptr[c]:cptr[c+1]]
        cptr = [0]
        lo = 0
        for i in range(nc):
            x = order[i]
            if level[x] == -1:
                lo = lptr2[_bfs(ptr, col, x, level, order2, lo, lptr2)]
                cptr.append(lo)
        j1 = -1
        for c in np.argsort(-np.diff(cptr), kind="stable"):
            lo = cptr[c]
            hi = cptr[c + 1]
            hmin = nc + 1
            best = 0
            for t in range(m):
                for i in range(lo, hi):
                    hcnt[A[t, loc[order2[i]]]] += 1
                h = 0
                for i in range(lo, hi):
                    a = A[t, loc[order2[i]]]
                    h = max(h, cnt[a] + hcnt[a])
                for i in range(lo, hi):
                    hcnt[A[t, loc[order2[i]]]] = 0
                if h < hmin:
                    hmin = h
                    best = t
            if j1 < 0:
                j1 = best
            for i in range(lo, hi):
                x = order2[i]
                lvl[x] = A[best, loc[x]]
                cnt[lvl[x]] += 1

        # Phase III: start from the root in the first or last level with the
        # smallest key, reverse the levels if it is in the last level
        start = -1
        for x in roots:
            if (lvl[x] == 0 or lvl[x] == k) and (start < 0 or key[x] < key[start]):
                start = x
        swapped = start >= 0 and lvl[start] == k and k > 0
        if swapped:
            for i in range(nc):
                x = order[i]
                lvl[x] = k - lvl[x]
        _reset(level, order, 0, nc)

        # nodes of each level, by increasing key: order2[lptr2[l]:lptr2[l+1]]
        comp = ORDER[:nc]
        idx = np.lexsort((KEY[comp], LVL[comp]))
        ORDER2[:nc] = comp[idx]
        LPTR2[0] = 0
        LPTR2[1:nl + 1] = np.cumsum(np.bincount(LVL[comp], minlength=nl))

        s0 = nperm
        if start >= 0:
            perm[nperm] = start
            level[start] = 0
            nperm += 1
        with nogil:
            nperm = _number(ptr, col, key, lvl, nl, order2, lptr2, level, perm, s0, nperm)

        # GGPS: reverse the numbering when the start end equals the end
        # selected for the largest connected part
        if generalized and j1 >= 0 and swapped != is_v[j1]:
            PERM[s0:nperm] = PERM[s0:nperm][::-1].copy()

    return PERM


def gibbs_poole_stockmeyer(ptr, col):
    """Gibbs-Poole-Stockmeyer ordering of a symmetric graph

    Implements the three phases of [GPS1976]_ for each connected component:

    I.   pseudo-diameter end-points ``v`` and ``u``
    II.  minimum width level structure combined from the rooted structures of ``v`` and ``u``
    III. level-by-level numbering, by increasing degree

    Parameters
    ----------
    ptr, col :
        symmetric CSR graph without self-loops, see `symmetric_graph`

    Returns
    -------
    perm : the permutation table, ``perm[new] = old``

    References
    ----------
    .. [GPS1976] N. E. Gibbs, W. G. Poole, P. K. Stockmeyer,
       "An algorithm for reducing the bandwidth and profile of a sparse matrix",
       SIAM J. Numer. Anal. 13, 236 (1976)
    """
    return _gps(ptr, col, False, None)


def generalized_gibbs_poole_stockmeyer(ptr, col, max_roots=None):
    r"""Generalized Gibbs-Poole-Stockmeyer (GGPS) ordering of a symmetric graph

    Implements [GGPS2009]_ for each connected component, which generalizes
    `gibbs_poole_stockmeyer` by:

    I.   using all pseudo-peripheral nodes as roots; the "u" end are all nodes
         in the last level of :math:`L_v`, the "v" end are all nodes with the
         degree of ``v`` whose rooted level structure has the same depth
    II.  combining all rooted level structures into the minimum width level structure;
         ties are broken by the rooted structure of smallest width
    III. numbering by increasing sum of the (1-based) codes of the adjacent nodes,
         and reversing the numbering according to the selection made for the largest
         connected part in II.

    Parameters
    ----------
    ptr, col :
        symmetric CSR graph without self-loops, see `symmetric_graph`
    max_roots :
        maximum number of candidate roots examined at each end (besides ``v`` and ``u``).
        The default examines all candidates, as in [GGPS2009]_, which requires a
        breadth-first search per candidate. For graphs where many nodes share the
        minimal degree (e.g. periodic lattices) this scales as :math:`O(N^2)`.

    Returns
    -------
    perm : the permutation table, ``perm[new] = old``

    References
    ----------
    .. [GGPS2009] Q. Wang, Y. C. Guo, X. W. Shi,
       "A generalized GPS algorithm for reducing the bandwidth and profile of a sparse matrix",
       Progress In Electromagnetics Research 90, 121 (2009)
    """
    return _gps(ptr, col, True, max_roots)


def electrode_ordering(ptr, col, start, end=None):
    """Ordering from level structures rooted at electrodes

    The level structure rooted at all nodes in `start` (a breadth-first search
    from the electrode) orders the nodes connected to `start`. With `end`, the
    level structures rooted at `start` and `end` are combined into a minimum width
    level structure as in phase II of `gibbs_poole_stockmeyer`. Each electrode acts
    as a single node: `start` is fixed in the first level and `end` in the level at
    the distance between the electrodes; only nodes behind an electrode (farther from
    the other electrode) are placed outside. The levels are numbered as in phase III
    of `gibbs_poole_stockmeyer`.

    Nodes not connected to `start` are ordered by `gibbs_poole_stockmeyer`.

    Parameters
    ----------
    ptr, col :
        symmetric CSR graph without self-loops, see `symmetric_graph`
    start :
        nodes of the first electrode (e.g. all its atoms or orbitals), placed first
    end :
        nodes of the last electrode, placed last in the connected part.
        Must be disjoint from, and connected to, `start`.

    Returns
    -------
    perm : the permutation table, ``perm[new] = old``
    """
    n = len(ptr) - 1

    def nodes(x, name):
        x = np.unique(np.asarray(x, dtype=np.int32).ravel())
        if len(x) == 0 or x[0] < 0 or x[-1] >= n:
            raise ValueError(f"electrode_ordering: {name} must be non-empty nodes in [0, {n})")
        return x

    start = nodes(start, "start")
    if end is not None:
        end = nodes(end, "end")
        if np.intersect1d(start, end).size:
            raise ValueError("electrode_ordering: start and end must be disjoint")
    return _gps(ptr, col, False, None, start, end)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _shift(const int[::1] ptr, const int[::1] col, const int[::1] lab, int[::1] hist,
                const int v, const int u, const int dh) noexcept nogil:
    """Add `dh` to the edge length histogram for all edges incident to `v` or `u`

    Returns the largest edge length of those edges.
    """
    cdef int ind, j, d, dmax = 0
    for ind in range(ptr[v], ptr[v + 1]):
        d = abs(lab[v] - lab[col[ind]])
        hist[d] += dh
        dmax = max(dmax, d)
    for ind in range(ptr[u], ptr[u + 1]):
        j = col[ind]
        if j != v:
            d = abs(lab[u] - lab[j])
            hist[d] += dh
            dmax = max(dmax, d)
    return dmax


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _swap(int[::1] lab, int[::1] perm, const int v, const int u) noexcept nogil:
    cdef int l = lab[v]
    lab[v] = lab[u]
    lab[u] = l
    perm[lab[v]] = v
    perm[lab[u]] = u


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _refine(const int[::1] ptr, const int[::1] col, int[::1] perm, int[::1] lab,
                  int[::1] hist, const unsigned char[::1] fixed, const int max_sweeps) noexcept:
    """Hill-climbing of `refine_bandwidth`, `perm` and `lab` are updated in-place"""
    cdef int n = ptr.shape[0] - 1
    cdef int sweep, v, u, i, ind, l, lo, hi, mid, dist, dlt, side, B, B_new, C
    cdef bint improved, critical

    with nogil:
        # histogram of edge lengths (each edge is counted twice)
        for i in range(n):
            for ind in range(ptr[i], ptr[i + 1]):
                hist[abs(lab[i] - lab[col[ind]])] += 1
        B = n
        while B > 0 and hist[B] == 0:
            B -= 1

        for sweep in range(max_sweeps):
            improved = False
            for v in range(n):
                # a bandwidth of 1 is optimal for any graph with edges
                if B <= 1:
                    break
                if fixed[v]:
                    continue
                critical = False
                lo = n
                hi = -1
                for ind in range(ptr[v], ptr[v + 1]):
                    l = lab[col[ind]]
                    lo = min(lo, l)
                    hi = max(hi, l)
                    critical = critical or abs(lab[v] - l) == B
                if not critical:
                    continue

                # try labels by increasing distance to mid(v)
                mid = (lo + hi) // 2
                dist = abs(lab[v] - mid)
                for dlt in range(dist):
                    for side in range(2 if dlt > 0 else 1):
                        l = mid - dlt if side == 0 else mid + dlt
                        if l < 0 or l >= n:
                            continue
                        u = perm[l]
                        if fixed[u]:
                            continue
                        C = hist[B]
                        _shift(ptr, col, lab, hist, v, u, -2)
                        _swap(lab, perm, v, u)
                        B_new = max(B, _shift(ptr, col, lab, hist, v, u, 2))
                        while B_new > 0 and hist[B_new] == 0:
                            B_new -= 1
                        if B_new < B or (B_new == B and hist[B] < C):
                            B = B_new
                            improved = True
                            break
                        # revert
                        _shift(ptr, col, lab, hist, v, u, -2)
                        _swap(lab, perm, v, u)
                        _shift(ptr, col, lab, hist, v, u, 2)
                    else:
                        continue
                    break
            if not improved:
                break


def refine_bandwidth(const int[::1] ptr, const int[::1] col, perm, fixed=None, int max_sweeps=100):
    """Local refinement of a permutation table that never increases the bandwidth

    Hill-climbing on the critical nodes, i.e. nodes with an edge of maximal length.
    A critical node ``v`` is swapped with the node ``u`` whose label is closest to
    ``mid(v)``, the midpoint of the smallest and largest labels of the neighbors of ``v``,
    as in the swap move of [MLGC2001]_. Only labels closer to ``mid(v)`` than the label of
    ``v`` are tried, and the first swap that reduces the bandwidth, or the number of edges
    with maximal length, is accepted.

    Applicable to the permutation table of any method.

    Parameters
    ----------
    ptr, col :
        symmetric CSR graph without self-loops, see `symmetric_graph`
    perm :
        the permutation table to refine, ``perm[new] = old``
    fixed :
        nodes that keep their position, e.g. electrode nodes from `electrode_ordering`
    max_sweeps :
        maximum number of sweeps over all nodes, a sweep without improvements stops

    Returns
    -------
    perm : the refined permutation table, ``perm[new] = old``

    References
    ----------
    .. [MLGC2001] R. Martí, M. Laguna, F. Glover, V. Campos,
       "Reducing the bandwidth of a sparse matrix with tabu search",
       Eur. J. Oper. Res. 135, 450 (2001)
    """
    cdef int n = ptr.shape[0] - 1
    PERM = np.array(perm, dtype=np.int32)
    if PERM.shape != (n,) or not np.array_equal(np.sort(PERM), np.arange(n)):
        raise ValueError("refine_bandwidth: perm must be a permutation of the graph nodes")
    LAB = np.empty(n, dtype=np.int32)
    LAB[PERM] = np.arange(n, dtype=np.int32)
    HIST = np.zeros(n + 1, dtype=np.int32)
    FIXED = np.zeros(n, dtype=np.uint8)
    if fixed is not None:
        FIXED[np.asarray(fixed, dtype=np.int32)] = 1
    _refine(ptr, col, PERM, LAB, HIST, FIXED, max_sweeps)
    return PERM


# BTD cost models, see btd_blocks
cdef enum BTDCost:
    COST_MEMORY
    COST_SPEED
    COST_SPEED_DIAGONAL


_BTD_COSTS = {
    "memory": COST_MEMORY,
    "speed": COST_SPEED,
    "speed-diagonal": COST_SPEED_DIAGONAL,
}


cdef inline double _single(const BTDCost model, const double d) noexcept nogil:
    """Cost of a diagonal block of size `d`"""
    if model == COST_MEMORY:
        return d * d
    return d * d * d


cdef inline double _pair(const BTDCost model, const double a, const double b) noexcept nogil:
    """Cost of the coupling between consecutive blocks of sizes `a` and `b`"""
    if model == COST_MEMORY:
        return 2 * a * b
    if model == COST_SPEED:
        return a * b * (a + b)
    return 0.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _grow(int** st, double** val, const int cap) noexcept nogil:
    """Double the capacity of the state arrays (4 ints per state)"""
    st[0] = <int*> realloc(st[0], 8 * cap * sizeof(int))
    val[0] = <double*> realloc(val[0], 2 * cap * sizeof(double))
    return 2 * cap


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _block_bounds(const int[::1] ptr, const int[::1] col, const int[::1] perm,
                        const int[::1] lab, int[::1] g, int[::1] m) noexcept nogil:
    """Bounds of the allowed blocks of the graph ordered by `perm`

    ``g[c]``: a block starting at ``c`` must end at ``>= g[c]`` (non-decreasing),
    ``m[c]``: the minimal size of a block starting at ``c``.
    """
    cdef int n = ptr.shape[0] - 1
    cdef int r, i, ind, c
    cdef int f = 0
    g[0] = 1
    for r in range(n):
        i = perm[r]
        f = max(f, r)
        for ind in range(ptr[i], ptr[i + 1]):
            f = max(f, lab[col[ind]])
        g[r + 1] = f + 1
    for c in range(n):
        m[c] = max(g[c], c + 1) - c


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def btd_blocks(const int[::1] ptr, const int[::1] col, perm, cost: str = "speed"):
    r"""Block sizes of the block-tridiagonal partition of minimal cost

    With the graph ordered by `perm`, the matrix is partitioned into contiguous
    blocks of sizes :math:`n_i` such that it is block-tridiagonal, while minimizing
    the `cost` model.

    A block ``[c', c)`` is allowed iff ``c > F(c'-1)``, where ``F(r)`` is the largest
    column coupled to any row ``<= r``.
    For ``cost="speed-diagonal"`` the optimal partition is found by dynamic programming,
    ``best[c] = min_{c'} best[c'] + n^3`` over the allowed blocks ``[c', c)``.
    The coupling terms of the other models make the cost of a block depend on the
    previous block, and the optimal partition is found by branch and bound on the
    pairs of consecutive blocks. The incumbent is the optimal partition of the diagonal
    terms, and the lower bound of the remaining cost is the optimal diagonal terms
    plus the coupling terms to the next block at its minimal allowed size.

    Parameters
    ----------
    ptr, col :
        symmetric CSR graph without self-loops, see `symmetric_graph`
    perm :
        the permutation table, ``perm[new] = old``
    cost : {"speed", "memory", "speed-diagonal"}
        the cost model:

        - speed: :math:`\\sum_i n_i^3 + n_i^2 n_{i+1} + n_i n_{i+1}^2`, the block
          inversions and the products with the coupling blocks
        - memory: :math:`\\sum_i n_i^2 + 2 n_i n_{i+1}`, the stored blocks
        - speed-diagonal: :math:`\\sum_i n_i^3`, only the block inversions

    Returns
    -------
    blocks : the block sizes, ``blocks.sum() == len(perm)``
    """
    if cost not in _BTD_COSTS:
        raise ValueError(f"btd_blocks: cost must be one of {list(_BTD_COSTS)}, got {cost!r}")
    cdef BTDCost model = _BTD_COSTS[cost]
    cdef int n = ptr.shape[0] - 1
    PERM = np.asarray(perm, dtype=np.int32)
    if PERM.shape != (n,) or not np.array_equal(np.sort(PERM), np.arange(n)):
        raise ValueError("btd_blocks: perm must be a permutation of the graph nodes")
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    LAB = np.empty(n, dtype=np.int32)
    LAB[PERM] = np.arange(n, dtype=np.int32)
    G = np.empty(n + 1, dtype=np.int32)
    M = np.zeros(n + 1, dtype=np.int32)
    BEST = np.empty(n + 1, dtype=np.float64)
    LOW = np.empty(n + 1, dtype=np.float64)
    PREV = np.empty(n + 1, dtype=np.int32)
    HEAD = np.full(n + 1, -1, dtype=np.int32)
    cdef int[::1] g = G
    cdef int[::1] m = M
    cdef double[::1] best = BEST
    cdef double[::1] low = LOW
    cdef int[::1] prev = PREV
    cdef int[::1] head = HEAD
    cdef int c, cp, c2, h, d, s, t, nst, cap, last
    cdef double cost_, upper, v
    # branch and bound states: (c, d, back, next) and value
    cdef int* st = NULL
    cdef double* val = NULL

    _block_bounds(ptr, col, PERM, LAB, g, m)

    with nogil:
        # optimal partition of the diagonal terms
        best[0] = 0.
        h = 0
        for c in range(1, n + 1):
            # allowed block starts are [0, h]
            while h + 1 < c and g[h + 1] <= c:
                h += 1
            best[c] = INFINITY
            for cp in range(h, -1, -1):
                cost_ = _single(model, c - cp)
                if cost_ >= best[c]:
                    break
                if best[cp] + cost_ < best[c]:
                    best[c] = best[cp] + cost_
                    prev[c] = cp

    if model == COST_SPEED_DIAGONAL:
        return _btd_backtrack(prev, n)

    with nogil:
        # incumbent: the diagonal partition with its coupling terms
        upper = best[n]
        c = n
        while c > 0 and prev[c] > 0:
            upper += _pair(model, prev[c] - prev[prev[c]], c - prev[c])
            c = prev[c]

        # lower bound of the cost of the blocks from c onwards
        low[n] = 0.
        for c in range(n - 1, -1, -1):
            low[c] = INFINITY
            for c2 in range(c + m[c], n + 1):
                d = c2 - c
                cost_ = _single(model, d)
                if cost_ >= low[c]:
                    break
                if c2 < n:
                    cost_ += _pair(model, d, m[c2])
                if cost_ + low[c2] < low[c]:
                    low[c] = cost_ + low[c2]

        # branch and bound, states are the blocks [c - d, c) in bucket c
        cap = 1024
        st = <int*> malloc(4 * cap * sizeof(int))
        val = <double*> malloc(cap * sizeof(double))
        # root state: no block before 0
        st[0] = 0; st[1] = 0; st[2] = -1; st[3] = -1
        val[0] = 0.
        head[0] = 0
        nst = 1
        last = -1
        for c in range(n):
            s = head[c]
            while s >= 0:
                d = st[4 * s + 1]
                if val[s] + _pair(model, d, m[c]) + low[c] < upper:
                    for c2 in range(c + m[c], n + 1):
                        cost_ = val[s] + _single(model, c2 - c)
                        if cost_ >= upper:
                            break
                        v = cost_ + _pair(model, d, c2 - c)
                        if c2 < n:
                            if v + _pair(model, c2 - c, m[c2]) + low[c2] >= upper:
                                continue
                        elif v >= upper:
                            continue
                        # find, or add, the state (c2, c2 - c)
                        t = head[c2]
                        while t >= 0 and st[4 * t + 1] != c2 - c:
                            t = st[4 * t + 3]
                        if t < 0:
                            if nst == cap:
                                cap = _grow(&st, &val, cap)
                            t = nst
                            nst += 1
                            st[4 * t] = c2
                            st[4 * t + 1] = c2 - c
                            st[4 * t + 3] = head[c2]
                            head[c2] = t
                            val[t] = INFINITY
                        if v < val[t]:
                            val[t] = v
                            st[4 * t + 2] = s
                            if c2 == n:
                                upper = v
                                last = t
                s = st[4 * s + 3]

    try:
        if last < 0:
            # the incumbent is optimal
            return _btd_backtrack(prev, n)
        blocks = []
        while st[4 * last + 1] > 0:
            blocks.append(st[4 * last + 1])
            last = st[4 * last + 2]
        return np.array(blocks[::-1], dtype=np.int64)
    finally:
        free(st)
        free(val)


def _btd_backtrack(const int[::1] prev, int c):
    """Block sizes from the boundaries ``prev[c] -> c``"""
    blocks = []
    while c > 0:
        blocks.append(c - prev[c])
        c = prev[c]
    return np.array(blocks[::-1], dtype=np.int64)
