import cython
cimport numpy

def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def longestPathFinder(adjacent_matrix):
    (nrows, ncols) = adjacent_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adjacent_matrix.astype(long, order='C', casting='safe', copy=True)
    tmp = -np.ones_like([n,n], dtype=numpy.int64)
    tmp[M==1] = 0
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = tmp.astype(long, order='C', casting='safe', copy=True)

    cdef unsigned int i, j, k
    for j in range(2,n):
        for i in range(n-j):
            for k in range(i+1,i+j):
                if M[i,k]!=0 and M[k,i+j]!=0:
                    tmp = M[i,k]+M[k,i+j]
                    if M[i,i+j]<tmp:
                        M[i,i+j] = tmp
                        path[i,i+j] = k
    return M, path


