import csv
import os
import re
import warnings
import scipy.misc
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import uuid
import hashlib
import logging
import face_recognition as face

from pprint import pprint
from datetime import datetime

# current date and time
now = datetime.now()
timestamp = datetime.timestamp(now)

import tools

lines = []
persons = 0
scan_path = '../images/serie_2'
csv_path = './persons/results.csv'
known_files = []
known_path = './persons'
npz_file = './persons/known_encodings.npz'
tolerance = 0.50
tolerance_min = 0.0
max_percent = 100
min_percent = 60
upsample = 1

save_results = True

exclude_from_header = ['face_location']
data_row = {
    'timestamp': timestamp,
    'file_identifier': '', 
    'known_file': False, 
    'hash': '',
    'top': -1, 
    'bottom': -1, 
    'left': -1, 
    'right': -1, 
    'faces': 0,
    'face_location': (), 
    'known_face': False, 
    'known_encodings': 0,
    'known_persons': 0, 
    'match': False, 
    'number_of_match': 0, 
    'new_person': False, 
    'name': '', 
    'distance': -1, 
    'match_percent': 0,
    'match_hash': None, 
    'save': False,
}

level = logging.INFO

logging.basicConfig()
logging.root.setLevel(level)

if not os.path.isdir(known_path):
    os.mkdir(known_path)


# determine polynomial equation to transform distance
poly = tools.get_equation([(tolerance_min, max_percent), (tolerance, min_percent)])
logging.info("Match distance >= {} ({:.2f}%), <= {} ({:.2f}%)".format(tolerance, min_percent, tolerance_min, max_percent))

if level >= logging.DEBUG:
    logging.debug(poly)
    step = 0.05
    for value in np.arange((tolerance_min-step), (tolerance+step), step):
        logging.debug("Distance {:.2f} = {:.2f}%".format(value, poly(value)))

# finding images to scan
img_files = tools.scan_folder(scan_path, ['.jpeg','.jpg','.png'])
logging.info("{} file(s) found".format(len(img_files)))

# load known encodings
known_names, known_hash, known_encodings = tools.load_known_encodings(npz_file, known_path)
known_persons = list(set(known_names))
logging.info("{} known encoding(s)".format(len(known_encodings)))
logging.info("Person(s): {}".format(len(known_persons)))

for filename in img_files:
    logging.info(">> {}".format(filename))
    picture = face.load_image_file(filename)
    picture = tools.resize(picture)

    # search for face locations
    face_locations = face.face_locations(picture,number_of_times_to_upsample=upsample)
    
    pil_image = Image.fromarray(picture)
    md5 = int(hashlib.md5(pil_image.tobytes()).hexdigest(), base=16)
    logging.info("\tCalculating md5 hash: {}".format(md5))
    
    known_file = True if md5 in known_files else False
    
    new_row = data_row.copy()
    new_row.update({
        'file_identifier': md5, 
        'known_encodings': len(known_encodings), 
        'known_persons': len(known_persons) + persons, 
        'known_file': known_file,
    })    
    
    if known_file:
        logging.info("\tKnown file, skipping.")
        tmp = new_row.copy()
        tmp['known_face'] = True
        lines.append(tmp.copy())
        continue
    else:
        known_files.append(md5)
    
    if not face_locations:
        tmp = new_row.copy()
        tmp.update({'faces': 0})
        lines.append(tmp.copy())
        del tmp
        
        logging.info("\tNo face locations found, skipping.")
        continue
    
    logging.info("\tFace(s) found: {}".format(len(face_locations)))
    
    new_row['faces'] = len(face_locations)

    face_encodings = face.face_encodings(picture, face_locations)
    for pos, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = pos
        match, save, new_person = False, False, False
        
        current_hash = tools.hash_array(face_encoding)
        
        new_row.update({
            'hash': current_hash,
            'face_location': pos, 
            'known_encodings': len(known_encodings), 
            'known_persons': len(known_persons) + persons, 
            'match': match, 
            'number_of_match': 0, 
            'save': save,
            'new_person': new_person,
            'name': '',
            'distance': -1,
        })
        
        if current_hash in known_hash:
        
        # same array, same face so skip
#         if any([np.array_equal(face_encoding, known_encoding) for known_encoding in known_encodings]):
            logging.info("\tKnown enconding, skipping.")
            # TODO: fuck off !!!
            lines.append(new_row.copy())
            continue
        
        if known_encodings:
            #code snippets from face_recognition, adaption of compare_faces but with distance returned
            distances = face.face_distance(known_encodings, face_encoding)
            results = list(distances <= tolerance)
            match = any(results)
            
            if match:
                indexes = [i for i in range(len(distances))]
                r = [index for match,index in zip(results,indexes) if match]
                matches = [(known_hash[i], known_names[i] ,poly(distances[i]), distances[i]) for i in r]
                matches = sorted(matches, key=lambda x: x[3])
                
                # store for analyzing
                if len(matches) > 1:
                    for match_hash, name, match_percent, distance in matches[1:]:
                        new_row.update({
                            'match': match, 
                            'number_of_match': len(matches), 
                            'name': name, 
                            'distance': distance, 
                            'match_percent': match_percent, 
                            'match_hash': match_hash,
                        })
                        lines.append(new_row.copy())
                
                match_hash, name, match_percent, distance = matches[0]
                logging.info("\tMatch with {}".format(name)) 
                
                save = True
                new_row.update({
                    'match': match, 
                    'number_of_match': len(matches), 
                    'name': name, 
                    'distance': distance, 
                    'match_percent': match_percent, 
                    'match_hash': match_hash,
                    'save': True,
                })
                lines.append(new_row.copy())
            else:
                logging.info("\tNo match found.")

        if not known_encodings or not match:            
#             name = tools.new_person(known_path)
            ind = max([int(n.split('_')[1]) for n in known_names])+1 if len(known_names) else 1
            name = "person_{}".format(ind)
            save = True
            new_person = True
            
            logging.info("\tNew person found, create {}".format(name))
            persons += 1
            
            new_row.update({
                'match': match, 
                'match_hash': None, 
                'number_of_match': 0, 
                'new_person': new_person, 
                'name': name, 
                'distance': -1, 
                'match_percent': -1, 
                'save': save
            })
            lines.append(new_row.copy())
                            
        if save:
#             path = os.path.join(known_path, name)
            path = known_path
            pos = (left, top, right, bottom)
            id = tools.generate_id()
            options = {'padding': 10, 'extension':'png', 'crop':'circle'}
            tools.save_array(os.path.join(path, 'encodings'), current_hash, face_encoding)
            tools.save_thumbnail(os.path.join(path, 'faces'), current_hash, pil_image, pos)
            tools.save_thumbnail(os.path.join(path, 'thumbnails'), current_hash, pil_image, pos, **options)

            known_names.append(name)
            known_encodings.append(face_encoding)
            known_hash.append(current_hash)

    del pil_image
    tools.compress_and_save(npz_file, known_names, known_encodings)

print("={}+{}".format(len(known_persons), persons))

if not save_results:
    sys.exit(0)


create_file = True if not os.path.isfile(csv_path) else False    

with open(csv_path, 'a') as csvfile:
    fieldnames = [key for key in data_row.keys() if key not in exclude_from_header]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if create_file:
        writer.writeheader()
    
    for data in lines:
        pos = data['face_location']
        (top, right, bottom, left) = pos if pos else (-1, -1, -1, -1)
        data.update({
            'top': top, 
            'bottom': bottom, 
            'left': left, 
            'right': right, 
        })
        del data['face_location']
        writer.writerow(data)