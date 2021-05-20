import os
import re
import warnings
import scipy.misc
import sys
import numpy as np
from PIL import Image, ImageDraw

import face_recognition

from tools import load_known_encodings, resize, scan_folder




scan_path = '../images/unknown'
tolerance = 0.55
number_of_times_to_upsample = 0

# finding images to scan
img_files = scan_folder(scan_path, ['.jpg'])
print("{} file(s) found".format(len(img_files)))

known_names, known_encodings = load_known_encodings('./')
print("{} known encoding(s)".format(len(known_encodings)))


for filename in img_files:    
    print(">> {}".format(filename))
    
    picture = face_recognition.load_image_file(filename)
    picture = resize(picture)
            
    face_locations = face_recognition.face_locations(picture,number_of_times_to_upsample=number_of_times_to_upsample)
    
    if not face_locations:
        continue
    
    print("\t face(s) found: {}".format(len(face_locations)))
    
    face_encodings = face_recognition.face_encodings(picture, face_locations)
    
    for single_encoding in range(len(face_encodings)):
        #code snippets from face_recognition, adaption of compare_faces but with distance returned
        distances = face_recognition.face_distance(known_encodings, face_encodings[single_encoding])
        result = list(distances <= tolerance)

        if any(result):
            indexes = [i for i in range(len(distances))]
            r = [index for match,index in zip(result,indexes) if match]
            matches = [(known_names[i],distances[i]) for i in r]
            matches = sorted(matches, key=lambda x: x[1])
            print(matches)
            
#             print("\tMatch found with {}".format(name))
        else:
            print("\tNo match found.")

