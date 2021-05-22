import os
import csv
import networkx as nx
from pyvis.network import Network

import tools

def group_by(data_list, iteration=2):
    for i in range(iteration):
        print("Iteration {}/{} items:{}".format((i+1),iteration,len(data_list)))
        for rel in data_list:
            r = [e for e in data_list for i in range(len(rel)) if rel[i] in e]
            for elem in r:
                rel += elem
                if elem in data_list and elem != rel:
                    data_list.remove(elem)

            rel = list(set(rel))

        data_list = [list(set(rel)) for rel in data_list]
    return data_list

def network(data, img_path=''):

    net = Network(height='750px', width='100%')

    lines = list(filter(lambda x: x['save'], data))
    hash_list = list(set([str(e['hash']) for e in lines]))
    node_list = [e[-3:] for e in hash_list]
    files_list = []
    groups = []
    rels = []

    for elem in lines:
        # a = str(elem['hash'])[-3:]
        # b = str(elem['match_hash'])[-3:]

        name = elem['match_hash'] if elem['match_hash'] else 'lonely'
        groups.append(name)

    for elem in lines:
        if not elem['match_hash']:
            continue

        if elem['hash'] == elem['match_hash']:
            continue

        a = str(elem['hash'])[-3:]
        b = str(elem['match_hash'])[-3:]

        rels.append([a,b])

    rels = group_by(rels, 2)

    groups = ["group_{}".format(i) for i in range(len(rels))]
    node_group = []
    for node in node_list:
        try:
            name = [name for rel,name in zip(rels,groups) if node in rel][0]
        except:
            name = node
        node_group.append(name)

    if img_path is not None:
        if not os.path.exists(img_path):
            raise FileExistsError

        for prefix,node in zip(hash_list, node_list):
            path = os.path.join(img_path, "{}.png".format(prefix))
            f = path if os.path.isfile(path) else None
            files_list.append(f)


    for node,path,group in zip(node_list, files_list, node_group):
        options = dict(group = group)
        if img_path:
            options.update(image=path, shape='image')
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
