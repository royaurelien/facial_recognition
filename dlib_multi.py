import argparse
import os
import sys
import numpy as np
from PIL import Image
import logging
import shutil
import face_recognition as face
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import dlib
from tqdm import tqdm

# current date and time
now = datetime.now()
timestamp = datetime.timestamp(now)

import analyze
import tools

IMAGES_TYPES = ['.jpeg','.jpg','.png']
DEFAULT_DIR = 'results'
NPZ_FILE = 'encodings.npz'
CSV_FILE = 'results.csv'
FACES_DIR = 'faces'
THUMBNAILS_DIR = 'thumbnails'
ENCODINGS_DIR = 'encodings'

DEFAULT_TOLERANCE = (0.10,0.50)
DEFAULT_MIN_MAX = (60,100)

DEFAULT_UPSAMPLE = 1
DEFAULT_SAVE = True
DEFAULT_EXPORT = True
DEFAULT_GRAPH = False
DEFAULT_LEVEL = logging.WARNING

calc_method = 'linear' # poly, percent

data = []
known_files = []
persons = 0

tolerance = DEFAULT_TOLERANCE[1]
tolerance_min = DEFAULT_TOLERANCE[0]
max_percent = DEFAULT_MIN_MAX[1]
min_percent = DEFAULT_MIN_MAX[0]


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

def percent_equation(value):
    max_value = DEFAULT_TOLERANCE[1]
    new_value = max_value - value
    new_value = new_value * 100 / max_value
    return new_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', dest='dest_dir', default=DEFAULT_DIR, help='Destination folder')
    parser.add_argument("src_dir", help='Source path')
    parser.add_argument('--level', action='store', dest='level', default=DEFAULT_LEVEL, help='Debug level')
    parser.add_argument('--save', action='store_true', dest='save_result', default=DEFAULT_SAVE, help='Save result')
    parser.add_argument('--export', action='store_true', dest='export_result', default=DEFAULT_EXPORT, help='Export result to CSV')
    parser.add_argument('--graph', action='store_true', dest='graph_result', default=DEFAULT_GRAPH, help='Show network graph')
    parser.add_argument('--erase', action='store_true', dest='erase', help='Erase all previous results before processing')

    args = parser.parse_args()

    logging.basicConfig()
    logging.root.setLevel(args.level)

    if not os.path.exists(args.src_dir):
        sys.exit(1)

    if args.erase:
        try:
            shutil.rmtree(args.dest_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            sys.exit(1)

    if not os.path.exists(args.dest_dir):
        os.mkdir(args.dest_dir)

    # determine polynomial equation to transform distance
    if calc_method == 'linear':
        distance_to_percent = tools.linear_equation([(tolerance_min, max_percent), (tolerance, min_percent)])
    elif calc_method == 'poly':
        distance_to_percent = tools.polynomial_equation([(tolerance_min, max_percent), (tolerance, min_percent)])
    elif calc_method == 'convert':
        distance_to_percent = percent_equation
    else:
        raise NotImplementedError

    logging.info("Match distance >= {} ({:.2f}%), <= {} ({:.2f}%)".format(tolerance, min_percent, tolerance_min, max_percent))

    if args.level >= logging.DEBUG:
        logging.debug(distance_to_percent)
        step = 0.05
        for value in np.arange((tolerance_min-step), (tolerance+step), step):
            logging.debug("Distance {:.2f} = {:.2f}%".format(value, distance_to_percent(value)))

    ######## START ############################################################

    start = time.time()

    detector = dlib.get_frontal_face_detector()

    # finding images to scan
    files = tools.scan_folder(args.src_dir, IMAGES_TYPES)
    logging.info("{} file(s) found".format(len(files)))

    # load known encodings, names and hash
    known_names, known_hash, known_encodings = tools.load_encodings(args.dest_dir, NPZ_FILE)
    known_persons = list(set(known_names))
    logging.info("{} known encoding(s)".format(len(known_encodings)))
    logging.info("Person(s): {}".format(len(known_persons)))

    images, list_of_hash, list_of_locations = [], [], []

    max_workers = 4
    threads = []
    options = dict(upsample=DEFAULT_UPSAMPLE)

    n_list = np.array_split(files, max_workers)
    # pbar = tqdm(total=100)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        parts = np.array_split(files, max_workers)
        futures = {executor.submit(tools.multi_process_file, part, detector, options):part for part in parts}

        for future in as_completed(futures):
            # pbar.update((100/max_workers))
            part = futures[future]
            try:
                data = future.result()
                images += data[0]
                list_of_hash += data[1]
                list_of_locations += data[2]
            except Exception as exc:
                print('%r generated an exception: %s' % (part, exc))
                sys.exit(1)
            # else:
            #     print('%r page is %d bytes' % (part, len(data)))

    # pbar.close()
    stop = time.time() - start
    tools.print_timer(stop)

    data = []


    pbar = tqdm(total=len(images))
    update = 100 / len(images)

    for image_array, image_hash, face_locations in zip(images, list_of_hash, list_of_locations):



        new_row = data_row.copy()
        new_row.update({
            'file_identifier': image_hash,
            'known_encodings': len(known_encodings),
            'known_persons': len(known_persons) + persons,
            'known_file': False,
        })

        pbar.update(update)

        if not face_locations:
            tmp = new_row.copy()
            tmp.update({'faces': 0})
            data.append(tmp.copy())
            del tmp

            logging.info("\tNo face locations found, skipping.")
            continue

        logging.info("\tFace(s) found: {}".format(len(face_locations)))

        new_row['faces'] = len(face_locations)


        face_encodings = face.face_encodings(image_array, face_locations)
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

            # NEXT: same array, same face so skip
            # if current_hash in known_hash:

            #     logging.info("\tKnown enconding, skipping.")
            #     # TODO: fuck off !!!
            #     data.append(new_row.copy())
            #     continue

            # first_anayze = False if current_hash in known_hash else True


            if known_encodings:
                #code snippets from face_recognition, adaption of compare_faces but with distance returned
                distances = face.face_distance(known_encodings, face_encoding)
                results = list(distances <= tolerance)
                match = any(results)

                if match:
                    indexes = [i for i in range(len(distances))]
                    r = [index for match,index in zip(results,indexes) if match]
                    matches = [(known_hash[i] ,str(known_hash[i])[-3:], distance_to_percent(distances[i]), distances[i]) for i in r]
                    matches = sorted(matches, key=lambda x: x[3])

                    # store for analyzing
                    if len(matches) > 1:
                        for match_hash, name, match_percent, distance in matches[1:]:
                            if distance == 0.0:
                                logging.info("\tDuplicate entry, skip.")
                                continue

                            new_row.update({
                                'match': match,
                                'number_of_match': len(matches),
                                'name': name,
                                'distance': distance,
                                'match_percent': match_percent,
                                'match_hash': match_hash,
                            })
                            data.append(new_row.copy())

                    match_hash, name, match_percent, distance = matches[0]
                    logging.info("\tMatch with {} at {:.2f}% ({:.4f})".format(name, match_percent, distance))

                    if distance == 0.0:
                        logging.info("\tDuplicate entry, skip.")
                        continue

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
                    data.append(new_row.copy())
                else:
                    logging.info("\tNo match found.")

            if not known_encodings or not match:
    #             name = tools.new_person(known_path)
                # ind = max([int(n.split('_')[1]) for n in known_names])+1 if len(known_names) else 1
                # name = "person_{}".format(ind)
                name = str(current_hash)[-3:]
                save = True
                new_person = True

                logging.info("\tNew person {}.".format(name))
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
                data.append(new_row.copy())

            if save:
                if args.save_result:
                    pos = (left, top, right, bottom)
                    id = tools.generate_id()
                    options = {'padding': 10, 'extension':'png', 'crop':'circle'}
                    pil_image = Image.fromarray(image_array)
                    tools.save_array(os.path.join(args.dest_dir, ENCODINGS_DIR), current_hash, face_encoding)
                    tools.save_thumbnail(os.path.join(args.dest_dir, FACES_DIR), current_hash, pil_image, pos)
                    tools.save_thumbnail(os.path.join(args.dest_dir, THUMBNAILS_DIR), current_hash, pil_image, pos, **options)
                    del pil_image

                known_names.append(name)
                known_encodings.append(face_encoding)
                known_hash.append(current_hash)
pbar.close()

if args.save_result:
    known_hash = [str(h) for h in known_hash]
    tools.compress_and_save(os.path.join(args.dest_dir, NPZ_FILE), known_hash, known_encodings)

if args.export_result or args.graph_result:
    # prepare data
    for line in data:
        pos = line['face_location']
        (top, right, bottom, left) = pos if pos else (-1, -1, -1, -1)
        line.update({
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
        })
        del line['face_location']

if args.export_result:
    csv_path = os.path.join(args.dest_dir, CSV_FILE)
    fieldnames = [key for key in data_row.keys() if key not in exclude_from_header]

    tools.save_csv(csv_path, fieldnames, data)

if args.graph_result:
    graph = analyze.network(data, img_path=os.path.join(args.dest_dir, THUMBNAILS_DIR))
    graph.show('mygraph.html')

stop = time.time() - start
tools.print_timer(stop)