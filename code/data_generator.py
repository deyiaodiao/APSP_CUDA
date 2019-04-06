from graph_tool.all import *
import numpy as np
import scipy

def sample_k(max):
    accept = False
    while not accept:
        k = np.random.randint(1,max+1)
        accept = np.random.random() < 1.0/k
        
    return k

for n_vertex in [10,20,100,200,500,1000,2000,5000,10000,20000]:
    for i in range(10):
        p = scipy.stats.poisson
        g = random_graph(n_vertex, lambda: (sample_k(20),sample_k(20)), model="probabilistic-configuration",
                            edge_probs=lambda a, b: (p.pmf(a[0], b[1]) * p.pmf(a[1], 20 - b[0])),
                            n_iter = 100,
                            directed=True)

        print(random_rewire(g, model="erdos"))

        matrix = np.zeros((n_vertex,n_vertex))
        for v in g.vertices():
            #print("vertex:", int(v))
            nei = list(g.get_out_neighbors(v))
            #print(nei)
            matrix[int(v),nei] = np.random.random(len(nei))
            #for w in list(v.out_neighbors()):
            #    print(w)
        #print(matrix)
        np.savetxt('../data/graph_{:d}_{:d}.txt'.format(i, n_vertex), matrix, fmt='%.6f')