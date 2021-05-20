import os
import csv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import sqrt


csv_path = './face_reco_results.csv'
img_path = './known_persons'

with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    lines = [row for row in reader if row['name']]
    
names = list(set([line['name'] for line in lines]))
group_by_name = [list(filter(lambda d: d['name'] == name, lines)) for name in names]



for item in group_by_name:
    images = {}
    name = item[0]['name']
    nb_lines = len(item)
    if nb_lines == 1:
        continue
        
    lines = list(filter(lambda x: eval(x['save']) == True, item))
    first_elem = list(filter(lambda x: eval(x['new_person']) == True and not x['match_hash'], lines))[0]
#     lines = list(filter(lambda x: x['hash'] != first_elem['hash'], lines))
    
    hash_list = [e['hash'] for e in lines if e['match_hash'] or (not e['match_hash'] and eval(e['new_person']))]
    first_hash = first_elem['hash'][-3:]
    first_person = first_elem['name']
    
    for prefix in hash_list:
        path = os.path.join(img_path, name, "{}..jpg".format(prefix))
        e = first_person if prefix == first_hash else prefix
        images[e] = mpimg.imread(path)
    

    
    G = nx.Graph()
    G.add_node(first_person, image=images.get(first_hash, None))
    
    for elem in hash_list:
        if elem == first_hash:
            continue
        G.add_node(elem[-3:], image=images.get(elem, None))
        
    
    for elem in lines:
        message = {'match':elem['match_percent']}
        weight = float(elem['match_percent'])/100
        
        if not elem['match_hash']:
            continue
        
        a = elem['hash'][-3:] if elem['hash'][-3:] != first_hash else first_person 
        b = elem['match_hash'][-3:] if elem['match_hash'][-3:] != first_hash else first_person
        
        G.add_edge(a, b, weight=weight, message=message)
    
#     print(name, len(lines))
    
    # print edges with message subject
    for (u, v, d) in G.edges(data=True):
        print(f"From: {u} To: {v} Subject: {d['message']['match']}")
    
    N = len(lines)
    pos = nx.spring_layout(G,k=3/sqrt(N))
#     nx.draw(G, pos, node_size=500, alpha=0.4, edge_color="r", font_size=16, with_labels=True)
#     nx.draw_networkx(G, pos, width=3, edge_color="r", alpha=0.6) 
    


            
    fig=plt.figure(figsize=(10,10))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx(G,pos,width=3, edge_color="r", alpha=0.6, with_labels=False, ax=ax)

    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    piesize=0.05 # this is the image size
    p2=piesize/2.0
    for n in G:
        img = G.nodes[n].get('image', None)
        if img is None:
            continue        
        
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(img)
        a.axis('off')
    ax.axis('off')
    plt.show()