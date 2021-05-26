import os
import csv
import networkx as nx
import logging
from pyvis.network import Network
import time

import tools

group_by = tools.group_by_set

def network(data, img_path=''):




    lines = list(filter(lambda x: x['save'], data))
    hash_list = list(set([str(e['hash']) for e in lines]))
    node_list = [e[-3:] for e in hash_list]
    int_list = [i for i in range(len(node_list))]
    node_dict = {k:v for k,v in zip(node_list, int_list)}
    files_list = []
    groups = []
    rels = []


    start = time.time()
    for elem in lines:
        a = str(elem['hash'])[-3:]
        b = str(elem['match_hash'])[-3:]

        elem['h'] = node_dict.get(a)
        elem['m'] = node_dict.get(b)

        if not elem['match_hash']:
            logging.debug("Skip lonely item: {}".format(elem['hash']))
            continue

        if elem['hash'] == elem['match_hash']:
            logging.debug("Skip self reference: {}".format(elem['hash']))
            continue

        rels.append([elem['h'],elem['m']])
        # rels.append([a,b])

    stop = time.time() - start
    tools.print_timer(stop)

    # print(rels)
    start = time.time()
    new_rels = group_by(rels)
    stop = time.time() - start
    tools.print_timer(stop)

    # print(new_rels)

    groups = ["group_{}".format(i) for i in range(len(new_rels))]
    node_to_group = {item:name for items,name in zip(new_rels,groups) for item in items}

    # net = nx.Graph()
    net = Network(width='100%', height='100%')

    if img_path is not None:
        if not os.path.exists(img_path):
            raise FileExistsError

        for prefix,node in zip(hash_list, node_list):
            path = os.path.join(img_path, "{}.png".format(prefix))
            f = path if os.path.isfile(path) else None
            files_list.append(f)


    for node, node_id, path in zip(node_list, int_list, files_list):
        options = dict(group = node_to_group.get(node_id, 'lonely'))
        if img_path:
            options.update(image=path, shape='circularImage')
        net.add_node(node_id, label=node, **options)

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

        # a = str(elem['hash'])[-3:]
        # b = str(elem['match_hash'])[-3:]

        a = elem['h']
        b = elem['m']

        net.add_edge(a, b, value=value, title="{:.2f}%".format(match))


    # nt.from_nx(net)
    return net


if __name__ == '__main__':

    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    csv_path = './results/results.csv'
    img_path = './results/thumbnails'

    lines = tools.read_csv(csv_path, filter_key='hash')

    graph = network(lines, img_path=img_path)
    # graph.show_buttons()
    graph.set_options("""
var options = {
  "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 4,
    "size": 35
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true
      }
    },
    "color": {
      "inherit": true
    },
    "smooth": false
  },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "parentCentralization": false,
      "direction": "LR"
    }
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "multiselect": true
  },
  "manipulation": {
    "enabled": true,
    "initiallyActive": true
  },
  "physics": {
    "hierarchicalRepulsion": {
      "centralGravity": 0
    },
    "minVelocity": 0.75,
    "solver": "hierarchicalRepulsion"
  }
}""")
    graph.show('index.html')
