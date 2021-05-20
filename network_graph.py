import os
import csv
import networkx as nx
from pyvis.network import Network


csv_path = './persons/results.csv'
img_path = './persons/thumbnails'

with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    lines = [row for row in reader if row['name']]
    
names = list(set([line['name'] for line in lines]))
group_by_name = [list(filter(lambda d: d['name'] == name, lines)) for name in names]

net = Network(height='750px', width='100%')

for item in group_by_name:
    images = {}
    name = item[0]['name']
    if len(item) <= 1:
        continue
    
    
    
    lines = list(filter(lambda x: eval(x['save']) == True, item))
    first_elem = list(filter(lambda x: eval(x['new_person']) == True and not x['match_hash'], lines))[0]
    hash_list = list(set([e['hash'] for e in lines]))
    node_list = [e[-3:] for e in hash_list]
    files_list = []
    
    first_hash = first_elem['hash'][-3:]
    
    for prefix,node in zip(hash_list, node_list):
        path = os.path.join(img_path, "{}.png".format(prefix))
        f = path if os.path.isfile(path) else None
        files_list.append(f)
        
    

    print(name, first_hash)
    print(files_list)
    
    for node,path in zip(node_list, files_list):
        label = name if node == first_hash else node
        net.add_node(node, label=label, image=path, shape='image', group=name)
    
#         print(net.get_node(node))
    
    for elem in lines:
        message = {'match':elem['match_percent']}
        match = float(elem['match_percent'])
        distance = float(elem['distance'])
        
        
        if not elem['match_hash']:
            continue
        
        a = elem['hash'][-3:]
        b = elem['match_hash'][-3:]
        
        more = elem['save']
        
        if a not in node_list or b not in node_list:
            print("ignore")
            continue
            
        net.add_edge(a, b, value=match/100, title="{:.2f}%".format(match))
        
#         print("{} match to {} with {:.2f} {}".format(a,b,weight,more))

net.show('mygraph.html')
