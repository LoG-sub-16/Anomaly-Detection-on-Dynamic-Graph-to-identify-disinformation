import datetime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,coo_matrix
from sklearn.cluster import SpectralClustering
from pathlib import Path


def anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m, seed = 1, tgn = False):
    np.random.seed(seed)
    print('[#s] generating anomalous dataset...\n', datetime.datetime.now())
    print('[#s] initial network edge percent: #.1f##, anomaly percent: #.1f##.\n', datetime.datetime.now(),
          ini_graph_percent * 100, anomaly_percent * 100)

    # ini_graph_percent = 0.5;
    # anomaly_percent = 0.05;
    train_num = int(np.floor(ini_graph_percent * m))

    # select part of edges as in the training set
    train = data[0:train_num, :]

    # select the other edges as the testing set
    test = data[train_num:, :]

    if not tgn: # TADDY normal
        #data to adjacency_matrix
        adjacency_matrix = edgeList2Adj(data)

        # clustering nodes to clusters using spectral clustering
        kk = 42 #3#10#42#42
        sc = SpectralClustering(kk, affinity='precomputed', n_init=10, assign_labels = 'discretize',n_jobs=-1)
        labels = sc.fit_predict(adjacency_matrix)


        # generate fake edges that are not exist in the whole graph, treat them as
        # anomalies
        idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)), axis=1)
        idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)), axis=1)
        generate_edges = np.concatenate((idx_1, idx_2), axis=1)

        ####### generate abnormal edges ####
        fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])
    else: # Modification pour introduire des anomalies extérieures
        # On commence par charger les anomalies enregistrées via "Synthetic dataset" (src, dst, t)
        fake_edges = pd.read_csv('../TADDY/data/for_TGN/TGN_anom_edges.csv', sep=',').to_numpy()
        anom_times = fake_edges.T[2]
        fake_edges = fake_edges[:,0:2]
        fake_edges_mem = fake_edges.copy()
        # On modifie les indices des noeuds conformément au pré-traitement de TADDY
        vertexs = pd.read_csv('../TADDY/data/for_TGN/TGN_vertexs.csv', sep=',').to_numpy()
        for i in range(fake_edges.shape[0]):
            fake_edges[i][0] = np.where(vertexs == fake_edges[i][0])[0][0]
            fake_edges[i][1] = np.where(vertexs == fake_edges[i][1])[0][0]
    fake_edges, fake_times = processEdges(fake_edges, data)

    if not tgn: # Normal
        anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
        anomalies = fake_edges[0:anomaly_num, :]
    else: # on sélectionne les anomalies à conserver et on détermine les indices auxquels les insérer pour conserver
          # l'ordre temporel
        anom_times = anom_times[fake_times]
        Path('../TADDY/data/for_TGN/').mkdir(parents=True, exist_ok=True)
        pd.DataFrame(np.hstack((vertexs[fake_edges_mem[fake_times]].reshape(len(anom_times), 2),
                    anom_times.reshape(len(anom_times),1)))).to_csv(f'../TADDY/data/for_TGN/TGN_anom_edges_treated.csv',
                    index=False)

        anomaly_num = fake_edges.shape[0]
        anomalies = fake_edges[0:anomaly_num, :]
        times = pd.read_csv('../TADDY/data/for_TGN/TGN_times.csv', sep=',').to_numpy().T[0]
        idx = np.searchsorted(times, anom_times)
        cpt = 0
        print(idx-len(train))
        for i in range(1,len(idx)):
            idx[i] += cpt
            if idx[i-1] == idx[i]:
                cpt += 1
        print(idx-len(train))
        anomaly_num = len(np.where(idx>=len(train))[0])
        idx = idx[np.where(idx>=len(train))[0]]
        anomalies = anomalies[-len(idx):]
    
    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    # randsample: sample without replacement
    # it's different from datasample!
    if not tgn:
        anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)
    else: # l'indice est celui dans les données de test, donc on lui retire le nb d'interactions d'entraînement
        anomaly_pos = idx-len(train)
    #anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200
    #print(anomaly_pos)
    #print(len(idx_test))
    idx_test[anomaly_pos] = 1
    print(len(np.where(idx_test == 1)[0]))
    print(len(anomalies))
    synthetic_test = np.concatenate((np.zeros([np.size(idx_test, 0), 2], dtype=np.int32), idx_test), axis=1)
    
    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)
    
    synthetic_test[idx_anomalies, 0:2] = anomalies
    synthetic_test[idx_normal, 0:2] = test

    train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0], train[:, 1])),
                           shape=(n, n))
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n) #TODO: node addition
    train_mat = train_mat + train_mat.transpose()

    return synthetic_test, train_mat, train

def anomaly_generation2(ini_graph_percent, anomaly_percent, data, n, m,seed = 1):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train_mat: the training network with square matrix format, the training
                        network edges for initial model training;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    # The actual generation method used for Netwalk(shown in matlab version)
    # Abort the SpectralClustering
    np.random.seed(seed)
    print('[%s] generating anomalous dataset...\n'% datetime.datetime.now())
    print('[%s] initial network edge percent: %.2f, anomaly percent: %.2f.\n'%(datetime.datetime.now(),
          ini_graph_percent , anomaly_percent ))

    # ini_graph_percent = 0.5;
    # anomaly_percent = 0.05;
    train_num = int(np.floor(ini_graph_percent * m))

    # select part of edges as in the training set
    train = data[0:train_num, :]

    # select the other edges as the testing set
    test = data[train_num:, :]

    #data to adjacency_matrix
    #adjacency_matrix = edgeList2Adj(data)

    # clustering nodes to clusters using spectral clustering
    # kk = 3 #3#10#42#42
    # sc = SpectralClustering(kk, affinity='precomputed', n_init=10, assign_labels = 'discretize',n_jobs=-1)
    # labels = sc.fit_predict(adjacency_matrix)


    # generate fake edges that are not exist in the whole graph, treat them as
    # anamalies
    # 真就直接随机生成
    idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1)

    ####### genertate abnormal edges ####
    #fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    # 移除掉self-loop以及真实边
    fake_edges = processEdges(fake_edges, data)

    #anomaly_num = 12#int(np.floor(anomaly_percent * np.size(test, 0)))
    # 按比例圈定要的异常边
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    anomalies = fake_edges[0:anomaly_num, :]

    # 按照总边数（测试正常+异常）圈定标签
    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    # randsample: sample without replacement
    # it's different from datasample!

    # 随机选择异常边的位置
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)

    #anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200
    # 选定的位置定为1
    idx_test[anomaly_pos] = 1

    # 汇总数据，按照起点，终点，label的形式填充，并且把对应的idx找出
    synthetic_test = np.concatenate((np.zeros([np.size(idx_test, 0), 2], dtype=np.int32), idx_test), axis=1)
    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)
    synthetic_test[idx_anomalies, 0:2] = anomalies
    synthetic_test[idx_normal, 0:2] = test

    # coo:efficient for matrix construction ;  csr: efficient for arithmetic operations
    # coo+to_csr is faster for small matrix, but nearly the same for large matrix (size: over 100M)
    #train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] , train[:, 1])),shape=(n, n))
    train_mat = coo_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0], train[:, 1])), shape=(n, n)).tocsr()
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n)
    train_mat = train_mat + train_mat.transpose()

    return synthetic_test, train_mat, train

def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # b:list->set
    # Time cost rate is proportional to the size

    idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)

    tmp = fake_edges[idx_fake]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]

    fake_edges[idx_fake] = tmp
    idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)
    idx_times = list(idx_remove_dups[0])
    fake_edges = fake_edges[idx_remove_dups]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []
    for idx, i in enumerate(a):
        if i not in b:
            c.append(i)
        else:
            idx_times.remove(idx)
    fake_edges = np.array(c)
    return fake_edges, np.array(idx_times)


def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))

    n = max(max(user, item) for user, item in data)  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[user - 1][item - 1] = 1  # Convert to 0-based index.
        matrix[item - 1][user - 1] = 1  # Convert to 0-based index.
    return matrix

if __name__ == "__main__":
    data_path = "data/karate.edges"
    # data_path = './fb-messages2.txt'

    edges = np.loadtxt(data_path, dtype=float, comments='%',delimiter=',')
    edges = edges[:,0:2].astype(dtype=int)

    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)

    synthetic_test, train_mat, train = anomaly_generation(0.5, 0.1, edges, n, m)

    print(train)
