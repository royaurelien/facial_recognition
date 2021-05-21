import os
import csv
import networkx as nx
from pyvis.network import Network

import tools


def network(data, img_path=''):

    net = Network(height='750px', width='100%')

    lines = list(filter(lambda x: x['save'], data))
    hash_list = list(set([str(e['hash']) for e in lines]))
    node_list = [e[-3:] for e in hash_list]
    files_list = []

    if img_path is not None:
        if not os.path.exists(img_path):
            raise FileExistsError

        for prefix,node in zip(hash_list, node_list):
            path = os.path.join(img_path, "{}.png".format(prefix))
            f = path if os.path.isfile(path) else None
            files_list.append(f)

    for node,path in zip(node_list, files_list):
        options = dict(image=path, shape='image') if img_path else dict()
        net.add_node(node, label=node, **options)

    for elem in lines:
        message = {'match':elem['match_percent']}
        match = float(elem['match_percent'])
        distance = float(elem['distance'])
        value = match / 100
        # more = elem['save']

        if not elem['match_hash']:
            continue

        if elem['hash'] == elem['match_hash']:
            continue

        a = str(elem['hash'])[-3:]
        b = str(elem['match_hash'])[-3:]

        net.add_edge(a, b, value=value, title="{:.2f}%".format(match))

    return net


if __name__ == '__main__':
    csv_path = './results/results.csv'
    img_path = './results/thumbnails'

    lines = tools.read_csv(csv_path, filter_key='hash')

    graph = network(lines, img_path=img_path)
    graph.show('mygraph.html')
