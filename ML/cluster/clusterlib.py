import random
import numpy as np

def kmeans_clustering(X, n_clusters, init_center=None, max_iter=10, epsilon=1e-4, random_state=100): 
    # inititalize centeroids
    if init_center is None:
        random.seed(random_state)
        idx = random.sample(range(X.shape[0]), n_clusters)
        center = X[idx,:]
    else:
        center = init_center
    iteration = 1
    
    labels_history = [] # label history 
    center_history = [] # centeroid history
    while(iteration<=max_iter):
        ## assign label
        labels = []
        for i in range(X.shape[0]):
            data = X[i, :]
            labels.append(np.argmin([np.linalg.norm(data-x) for x in center]))
        
        labels = np.array(labels)
        ## update centeroids
        next_center = []
        for i in range(n_clusters):
            target_idx = np.where(labels==i)[0]
            center_val = np.mean(X[target_idx,:], axis=0)
            next_center.append(center_val)
 
        next_center = np.array(next_center)
        if epsilon:
            if np.linalg.norm(next_center-center) <= epsilon:
                break
        center = next_center
        labels_history.append(labels)
        center_history.append(center)
        iteration += 1
    return (labels, iteration, labels_history, center_history)


def cluster(x):
    # input: [(x1,y1),(x2,y2),...,(xn,yn)]
    # output: [0,0,1,0,2,2,1,...] (index of cluster)

    labels, iteration, labels_history, center_history = kmeans_clustering(x, 2)
    return labels


def _main():
    result = cluster(np.array([(5,42),(35,2.4),(9.3,50),(99.1,24.2)]))
    print(result)

if __name__=="__main__":
    _main()
