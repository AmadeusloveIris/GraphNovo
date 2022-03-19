import cython
import numpy as np
from libc.math cimport fabs, exp
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def edge_matrix_generator(long n, 
                          double mass_threshold,
                          long subedge_maxnum,
                          double[::1] theo_edge_mass,
                          double[:,::1] mass_difference,
                          long[:,::1] start_edge_type,
                          long[:,::1] end_edge_type):
    cdef long[:,:,::1] edge_type_matrix_view
    cdef double[:,:,::1] edge_different_matrix_view
    edge_type_matrix = np.zeros((n,n,subedge_maxnum),dtype=np.int64)
    edge_different_matrix = np.zeros((n,n,subedge_maxnum),dtype=np.float64)
    edge_type_matrix_view = edge_type_matrix
    edge_different_matrix_view = edge_different_matrix
    
    cdef long i,j,k,edge
    for i in range(n):
        for j in range(n):
            for k in range(end_edge_type[i,j]-start_edge_type[i,j]):
                edge = start_edge_type[i,j]+k
                edge_type_matrix_view[i,j,k] = edge + 1
                edge_different_matrix_view[i,j,k] = exp(-fabs(theo_edge_mass[edge]-mass_difference[i,j])/mass_threshold)
    return edge_type_matrix, edge_different_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
def typological_sort_floyd_warshall(long n, adjacency_matrix):
    cdef long[:, ::1] dist_matrix_view, predecessors_view
    dist_matrix = adjacency_matrix.copy()
    predecessors = -9999*np.ones_like(adjacency_matrix)
    dist_matrix_view = dist_matrix
    predecessors_view = predecessors
    cdef long i, j, k
    for i in range(2,n):
        for j in range(n-i):
            for k in range(j+1,i+j):
                if dist_matrix_view[j,k] != 0 and dist_matrix_view[k,i+j] != 0:
                    if dist_matrix_view[j,i+j] < dist_matrix_view[j,k] + dist_matrix_view[k,i+j]:
                        dist_matrix_view[j,i+j] = dist_matrix_view[j,k] + dist_matrix_view[k,i+j]
                        predecessors_view[j,i+j] = k
    return dist_matrix, predecessors

cdef struct CDLinkedList:
    ListNode* head
    ListNode* tail
    unsigned int length

cdef struct ListNode:
    ListNode* link
    long data

cdef ListNode* list_node_create(long data) nogil:
    cdef ListNode* n = <ListNode *> malloc(sizeof(ListNode))
    n.link = NULL
    n.data = data
    return n

cdef CDLinkedList* list_create(long data) nogil:
    cdef CDLinkedList* lst = <CDLinkedList *> malloc(sizeof(CDLinkedList))
    lst.head = list_node_create(data)
    lst.tail = lst.head
    lst.length = 1
    return lst

cdef long* list_free(CDLinkedList* lst) nogil:
    cdef long* result = <long *> malloc(lst.length*sizeof(long))
    cdef unsigned int i
    for i in range(lst.length):
        result[i] = list_pop(lst)
    free(lst)
    return result

cdef CDLinkedList* list_combine(CDLinkedList* lst_left, CDLinkedList* lst_right) nogil:
    cdef CDLinkedList* lst = <CDLinkedList *> malloc(sizeof(CDLinkedList))
    
    if lst_left==NULL: 
        free(lst)
        return lst_right
    elif lst_right==NULL: 
        free(lst)
        return lst_left
    else:
        lst.head = lst_left.head
        lst.tail = lst_right.tail
        lst.length = lst_left.length + lst_right.length
        lst_left.tail.link = lst_right.head
        free(lst_right)
        free(lst_left)
        return lst

cdef int list_pop(CDLinkedList* lst) nogil:
    cdef ListNode* pop_block = lst.head
    cdef long result = pop_block.data
    lst.head = pop_block.link
    free(pop_block)
    lst.length-=1
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef CDLinkedList* get_all_edges(long[:,::1] predecessors, unsigned int i, unsigned int j) nogil:
    cdef int k = predecessors[i][j]
    if k == -9999:
        return NULL
    else:
        return list_combine(list_combine(get_all_edges(predecessors, i, k), list_create(k)), get_all_edges(predecessors, k, j))

@cython.boundscheck(False)
@cython.wraparound(False)
def gen_edge_input(long node_num,
                   long max_dist,
                   long subedge_maxnum,  
                   long[:,::1] predecessors, 
                   long[:,:,::1] edge_matrix,
                   double[:,:,::1] edge_rdifferent_matrix,
                   long[:,::1] direct_link):

    cdef unsigned int i, j, path_length
    cdef long k, l, start_node, end_node
    path_matrix = np.zeros((node_num, node_num, max_dist, subedge_maxnum),dtype=np.int64)
    path_rdifferent_matrix = np.zeros((node_num, node_num, max_dist, subedge_maxnum),dtype=np.float64)
    cdef long[:,:,:,::1] path_matrix_view = path_matrix
    cdef double[:,:,:,::1] path_rdifferent_matrix_view = path_rdifferent_matrix
    cdef CDLinkedList* path
    cdef long* path_array
    
    for i in range(node_num-1):
        for j in range(i+1,node_num):
            if predecessors[i][j] == -9999: 
                if direct_link[i,j]==1:
                    for l in range(subedge_maxnum):
                        path_matrix_view[i,j,0,l] = edge_matrix[i,j,l]
                        path_rdifferent_matrix_view[i,j,0,l] = edge_rdifferent_matrix[i,j,l]
            else:
                path = list_combine(list_combine(list_create(i), get_all_edges(predecessors, i, j)), list_create(j))
                path_length = path.length
                path_array = list_free(path)
                for k in range(path_length-1):
                    start_node = path_array[k]
                    end_node = path_array[k+1]
                    for l in range(subedge_maxnum):
                        path_matrix_view[i,j,k,l] = edge_matrix[start_node,end_node,l]
                        path_rdifferent_matrix_view[i,j,k,l] = edge_rdifferent_matrix[start_node,end_node,l]
                
                free(path_array)
                
    return path_matrix, path_rdifferent_matrix
