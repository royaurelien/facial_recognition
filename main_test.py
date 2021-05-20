import os
import re
import warnings
import scipy.misc
import sys
import numpy as np
from PIL import Image, ImageDraw
import uuid

import face_recognition as fcr


from tools import load_known_encodings, resize, scan_folder


scan_path = '../images/selection'
known_path = './known'
save = True
number_of_times_to_upsample = 0


# finding images to scan
img_files = scan_folder(scan_path, ['.jpg'])
print("{} file(s) found".format(len(img_files)))

# load known encodings
known_names, known_encodings = load_known_encodings('./')
print("{} known encoding(s)".format(len(known_encodings)))

for filename in img_files:    
    print(">> {}".format(filename))
    picture = fcr.load_image_file(filename)
    picture = resize(picture)
            
    face_locations = fcr.face_locations(picture,number_of_times_to_upsample=number_of_times_to_upsample)
    
    if not face_locations:
        continue
    
    print("\tFace(s) found: {}".format(len(face_locations)))
    
    pil_image = Image.fromarray(picture)
        
    face_encodings = fcr.face_encodings(picture, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        if any([np.array_equal(face_encoding, known_encoding) for known_encoding in known_encodings]):
            print("\tKnown enconding, skipping.")
            continue
        
        if save:
            id = uuid.uuid4()
            filepath = os.path.join(known_path, "{}.npy".format(id))
            encoded_file = np.save(filepath, face_encoding)
            known_names.append(filepath)
            known_encodings.append(face_encoding)

            im_crop = pil_image.crop((left, top, right, bottom))
            im_crop.save(os.path.join(known_path, "{}.jpg".format(id)))
            
            print("\tNew enconding found, saving.")
    
    del pil_image
        
#     draw = ImageDraw.Draw(pil_image)    
#     for (top, right, bottom, left) in face_locations:
#         draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
#     del draw
#    
#     new_filename = filename.replace('.jpg', '_FACERECO.jpg')
#     pil_image.save(new_filename)