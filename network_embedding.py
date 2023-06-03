from collections import defaultdict
from os import sep
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec
from numpy.core.fromnumeric import sort


class Node2vecModel():
    def __init__(self) -> None:
        pass

    @staticmethod
    def createGraph():
        if weighted:
            G = nx.read_weighted_edgelist(path, delimiter=sep_type, nodetype=int, create_using=Graph_type)
        else:
            G = nx.read_edgelist(path, delimiter=sep_type, nodetype=int, create_using=Graph_type)
            for src, targ in G.edges():
                G[src][targ]['weight'] = 1.0
        return G
    
    @staticmethod
    def createLabelGraph():
        if weighted:
            G = nx.read_weighted_edgelist(path, delimiter=sep_type, create_using=Graph_type)
        else:
            G = nx.read_edgelist(path, delimiter=sep_type, create_using=Graph_type)
            for src, targ in G.edges():
                G[src][targ]['weight'] = 1.0
        return G


    def training(self, save=True):
        # build a graph
        G = Node2vecModel.createLabelGraph() if dataset['label'] else Node2vecModel.createGraph()
        
        # build a node2vec model
        node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_len, num_walks=walks_num, workers=1, p=P, q=Q)
        # train model
        model = node2vec.fit(window=window_context_size, min_count=1, batch_words=4)
        # save model and node embeddings trained
        if save:
            embedding_file_path = dataset['embedding_file_path']
            model_file_path = dataset['model_file_path']
            # Save embeddings for later use
            model.wv.save_word2vec_format(embedding_file_path)
            # Save model for later use
            model.save(model_file_path)


def load_model(path):
    model = Word2Vec.load(path)
    return model

def find_community_greedy():
    from networkx.algorithms.community import greedy_modularity_communities
    G = Node2vecModel.createGraph()
    comms = list(greedy_modularity_communities(G))
    return comms

def find_community_louvain():
    from community import best_partition
    G = Node2vecModel.createLabelGraph() if dataset['label'] else Node2vecModel.createGraph()
    partitions = best_partition(G)
    node2clusters = dict()
    for key, value in partitions.items():
        if value not in node2clusters:
            node2clusters[value] = [key]
        else:
            node2clusters[value].append(key)
    
    return node2clusters

# using Logistic Regression for node classification
def train_for_node_classification():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import scale

    model = load_model(dataset['model_file_path'])
    # node_ids = model.wv.index_to_key
    node_embeddings_X = model.wv.vectors
    nodes_ids = list(map(int, list(model.wv.key_to_index)))
    cluster_ids = np.loadtxt(ground_truth, delimiter=sep_type, usecols=1, dtype=str)
    Y = []
    for nodes_id in nodes_ids:
        Y.append(cluster_ids[nodes_id])
    X = np.array(node_embeddings_X)
    Y = np.array(Y)
    # split data set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=321)
    # train
    LR = LogisticRegressionCV(max_iter=500, cv=10, scoring='accuracy', verbose=False, multi_class='ovr')
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))
    # print(model.wv.similarity('880', '4'))
    # print(model.wv['628'] == node_embeddings[5])
    

# using cluster
def compute_accuracy_for_node_classification(clustering_type='hier'):
    # 1) cluster by node2vec
    if clustering_type == 'hier':
        clusters_1, _ = hierachical_clustering()
    elif clustering_type == 'kmeans':
        clusters_1, _ = kmeans_clustering()
    
    # cluster by groud truth label
    nodes_2 = np.loadtxt(ground_truth, delimiter=sep_type, usecols=0, dtype=int)
    clusters_2 = np.loadtxt(ground_truth, delimiter=sep_type, usecols=1, dtype=str)
    
    node2cluster_dict2 = defaultdict(list)
    for node, cluster in zip(nodes_2, clusters_2):
        if cluster in node_cls_labels:
            node2cluster_dict2[cluster].append(node)
        else:
            node2cluster_dict2['other'].append(node)
    assert(len(node2cluster_dict2) == num_clusters)
    
    for key in node2cluster_dict2:
        print(key, '', len(node2cluster_dict2[key]))
    
    for cluster in set(clusters_1):
        print(cluster, '', list(clusters_1).count(cluster))

# abandoned
def compute_accuracy_from_Y(clustering_type='hier'):
    # 1) cluster by node2vec
    if clustering_type == 'hier':
        clusters_1, nodes_1 = hierachical_clustering()
    elif clustering_type == 'kmeans':
        clusters_1, nodes_1 = kmeans_clustering()

    # cluster by groud truth label
    nodes_2 = list(np.loadtxt(ground_truth)[:, 0].astype('int'))
    clusters_2 = list(np.loadtxt(ground_truth)[:, 1].astype('int'))
    node2cluster_dict2 = defaultdict(list)
    for node, cluster in zip(nodes_2, clusters_2):
        node2cluster_dict2[cluster].append(node)
    assert(len(node2cluster_dict2) == num_clusters)
    # compute the ratio of intersection
    temp = []
    sum = 0
    # print(sorted(node2cluster_dict2.keys()))
    for key in sorted(node2cluster_dict2.keys()):
        for node2 in node2cluster_dict2[key]:
            node1_idx = nodes_1.index(str(node2))
            node1_cluster = clusters_1[node1_idx]    
            temp.append(node1_cluster)
    
        print(key,'', temp)
        # most_frequent_cluster = max(temp, key = temp.count)
        # print(i, '', most_frequent_cluster)
        sum += 1 - len(set(temp)) / len(node2cluster_dict2)
        temp = []
    print(sum / len(node2cluster_dict2))

def hierachical_clustering():
    from sklearn.cluster import AgglomerativeClustering
    ac = AgglomerativeClustering(n_clusters = None, affinity = 'cosine', 
                                 linkage = 'complete', distance_threshold = distance_threshold)
    model = load_model(dataset['model_file_path'])
    nodes = list(model.wv.key_to_index)
    vecs = model.wv[nodes]
    # node_vec_dict = {node:model.wv[node] for node in nodes}
    clusters = ac.fit_predict(vecs)
    return clusters, nodes # ids in nodes are not sorted

def kmeans_clustering():
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=2022)
    model = load_model(dataset['model_file_path'])
    nodes = list(model.wv.key_to_index)
    vecs = model.wv[nodes]
    kmeans = kmeans.fit(vecs)
    return kmeans.labels_, nodes

def cluster2gephi(path, save=True, clustering_type='hier'):
    if clustering_type == 'hier':
        clusters, nodes = hierachical_clustering()
    elif clustering_type == 'kmeans':
        clusters, nodes = kmeans_clustering()
    elif clustering_type == 'others' and ground_truth is not None:
        clusters = list(np.loadtxt(ground_truth)[:, 1].astype('int'))
        nodes = list(np.loadtxt(ground_truth)[:, 0].astype('int'))
    else:
        return

    G = Node2vecModel.createLabelGraph() if dataset['label'] else Node2vecModel.createGraph()
    nodes2cluster = dict()
    for idx, cluster in enumerate(clusters):
        node_id = nodes[idx]
        if dataset['label']:
            nodes2cluster[node_id] = {'cluster': cluster}
        else:
            nodes2cluster[int(node_id)] = {'cluster': cluster}
    nx.set_node_attributes(G, nodes2cluster)
    print('Clusters Number: ', len(set(clusters)))
    if save:
        nx.write_gexf(G, path)


# define dataset dict
dataset_desc = {'facebook':
                    {'path':'dataset/facebook/out.ego-facebook', 
                    'dataset_sep': 'space',
                    'directed': True, 
                    'weighted': False,
                    'label': False,
                    'ground_truth': None,
                    'embedding_file_path': 'Model/Facebook/P05Q1D128W200/facebook_node_n2v.embed',
                    'model_file_path': 'Model/Facebook/P05Q1D128W200/facebook_node_embed_n2v.ml'},
                'openflights':
                    {'path':'dataset/opsahl-openflights/out.opsahl-openflights', 
                    'dataset_sep': 'space',
                    'directed': True, 
                    'weighted': False,
                    'label': False,
                    'ground_truth': None,
                    'embedding_file_path': 'Model/OpenFlights/P1Q05D128L80R20/openflights_node_n2v.embed',
                    'model_file_path': 'Model/OpenFlights/P1Q05D128L80R20/openflights_node_embed_n2v.ml'},
                'us_airport':
                    {'path':'dataset/opsahl-usairport/out.opsahl-usairport', 
                    'dataset_sep': 'space',
                    'directed': True, 
                    'weighted': True,
                    'label': False,
                    'ground_truth': None,
                    'embedding_file_path': 'Model/USAirports/P1Q05D128L80R20/USAirports_node_n2v.embed',
                    'model_file_path': 'Model/USAirports/P1Q05D128L80R20/USAirports_node_embed_n2v.ml'},
                'facebook_large':
                    {'path':'dataset/facebook_large/musae_facebook_edges.csv', 
                    'dataset_sep': 'comma',
                    'directed': False, 
                    'weighted': False,
                    'label': False,
                    'ground_truth': 'dataset/facebook_large/musae_facebook_node_label.txt',
                    'embedding_file_path': 'Model/Facebook_large/P1Q05D128L80R20/facebook_large_node_n2v.embed',
                    'model_file_path': 'Model/Facebook_large/P1Q05D128L80R20/facebook_large_node_embed_n2v.ml'},
                'eu_email':
                {'path':'dataset/eu_email/email-Eu-core.tsv', 
                    'dataset_sep': 'space',
                    'directed': True,
                    'weighted': False,
                    'label': False,
                    'ground_truth': 'dataset/eu_email/email-Eu-core-department-labels.txt',
                    'embedding_file_path': 'Model/EU_email/P1Q05D128L80R20/email_node_dw.embed',
                    'model_file_path': 'Model/EU_email/P1Q05D128L80R20/email_node_embed_dw.ml'},
                'zebra':
                {'path':'dataset/moreno_zebra/out.moreno_zebra_zebra', 
                    'dataset_sep': 'space',
                    'directed': False,
                    'weighted': False,
                    'label': False,
                    'ground_truth': None,
                    'embedding_file_path': 'Model/Zebra/P1Q2D128W80/zebra_node_n2v.embed',
                    'model_file_path': 'Model/Zebra/P1Q2D128W80/zebra_node_embed_n2v.ml'},
                'paper':
                {'path':'dataset/paper/paper.tsv', 
                    'dataset_sep': 'space',
                    'directed': False,
                    'weighted': False,
                    'label': False,
                    'ground_truth': None,
                    'embedding_file_path': 'Model/Paper/P05Q2D16L20R10/paper_node_n2v.embed',
                    'model_file_path': 'Model/Paper/P05Q2D16L20R10/paper_node_embed_n2v.ml'},
                'GOT':
                {'path':'dataset/GOT/book5.csv', 
                    'dataset_sep': 'space',
                    'directed': False,
                    'weighted': True,
                    'label': True,
                    'ground_truth': None,
                    'embedding_file_path': 'Model/GOT_b5/P1Q2WN10WL80/got_node_n2v.embed',
                    'model_file_path': 'Model/GOT_b5/P1Q2WN10WL80/got_node_embed_n2v.ml'},
}
    # test different dataset
dataset = dataset_desc['facebook_large']
# define parameters related to graph creation
path = dataset['path']
sep_type = ',' if dataset['dataset_sep'] == 'comma' else ' '
ground_truth = dataset['ground_truth']
directed = dataset['directed']
weighted = dataset['weighted']
node_cls_labels = ['politician', 'company', 'tvshow', 'government']
Graph_type = nx.DiGraph() if directed else nx.Graph()
# define hyperparameters for biased random walk and node2vec or 'seq2vec'
    # 1) biased random walk
P = 0.5
Q = 2
dim = 16
walks_num = 10
walk_len = 20
    # 2) node2vec
window_context_size = 4
# define hierachical clustering / kmeans
distance_threshold = .9 # hier
num_clusters = 2 # kmeans


# training model
n2v = Node2vecModel()
n2v.training()

# clustering testing
cluster2gephi('nx2Gephi/paper_P05Q2D16L20R10KM2.gexf', save=True, clustering_type='kmeans')

# node2vec testing
# model = load_model(dataset['model_file_path'])
# print(model.wv.similarity('90','4'))
# print(model.wv.most_similar(negative='288', topn=100))

# compare the result of node2vec with the groud truth in community detecion
compute_accuracy_from_Y('kmeans')

# train for node classification or community detection with ground truth label
train_for_node_classification()

# node classification custome metrics
# compute_accuracy_for_node_classification('kmeans')








# nodes = list(model.wv.key_to_index)
# print(nodes)
# print(model.wv[nodes][0])
# print(model.wv[nodes][0] == model.wv[nodes[0]])
# nodes = map(int, list(model.wv.key_to_index))
# word2vec = {node:model.wv[node] for node in nodes}
# G = Node2vecModel.createLabelGraph()
# nx.write_gexf(G, 'got_b5_tst.gexf')


# G = Node2vecModel.createGraph()
# nx.write_gexf(G, 'facebook.gexf')
# print(G.edges[1525, 1625]['weight'])
# nx.drawing.nx_pylab.draw_spring(G)
# nx.draw(G)
# plt.show()



#     community_to_color = {
#     0 : 'tab:blue',
#     1 : 'tab:orange',
#     2 : 'tab:green',
#     3 : 'tab:red',
#     4 : 'tab:grey',
#     5 : 'tab:black',
#     6 : 'tab:yellow',
#     7 : 'tab:purple',
# }
#     node_color = {node: community_to_color[community_id] for node, community_id in partitions.items()}
#     NetGraph(G,
#       node_color=node_color, node_edge_width=0, edge_alpha=0.1,
#       node_layout='community', node_layout_kwargs=dict(node_to_community=partitions),
#       edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
# )