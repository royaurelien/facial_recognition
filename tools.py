import os
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import logging
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import hashlib

_logger = logging.getLogger(__name__)

def hash_array(numpy_array):
    return int(hashlib.md5(bytes(numpy_array)).hexdigest(), base=16)

def scan_folder(path, extension=[]):
    files = []
    for root, subdirs, current_files in os.walk(path):
        files += [os.path.join(root,name) for name in current_files if name.endswith(tuple(extension))]
    
    return files

def load_known_encodings(path):
    face_encodings, names, hash_list = [], [], []
    numpy_files = scan_folder(path, ['.npy'])
    
    for f in numpy_files:
        array = np.load(f)
        if array is not None:
            face_encodings.append(array)
            names.append(f)
            hash_list.append(hash_array(array))
            
    return names, hash_list, face_encodings


def load_from_subdirs(path):
    extension = ['.npy']
    face_encodings, names, hash_list = [], [], []

    for root, subdirs, current_files in os.walk(path):
        for filename in current_files:
            if not filename.endswith(tuple(extension)):
                continue
                
            array = np.load(os.path.join(root, filename))
            name = root.replace(path, '').replace('.npy','').replace('/','')
            if array is not None:
                face_encodings.append(array)
                names.append(name)
                hash_list.append(hash_array(array))
            
    return names, hash_list, face_encodings

def resize(picture, max_width=1600):
    #Scale down image if large, performance suggestion snippet from face_recognition
    if picture.shape[1] > max_width:
        scale_factor = float(max_width) / picture.shape[1]
        im = Image.fromarray(picture)
        size = tuple((np.array(im.size) * scale_factor).astype(int))
        picture = np.array(im.resize(size))        
        
#         _logger.info("\tResizing.")
        _logger.info("\tResizing to {size[0]}x{size[1]} pixels with {o:.2f}x".format(size=size, o=scale_factor))    
        
    return picture

def new_person(path, prefix='person'):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    dirs = os.listdir(path)
    if dirs:
        index = sorted([int(elem.split('_')[1]) for elem in dirs])[-1]
        index += 1
    else:
        index = 1
        
    name = "_".join([prefix,str(index)])
    os.mkdir(os.path.join(path,name))
    
    return name

def generate_id(**kwargs):
    return  uuid.uuid4()

def save_array(path, filename, array):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    filepath = os.path.join(path, "{}.npy".format(filename))
    encoded_file = np.save(filepath, array)

def mask_circle_transparent(pil_img, blur_radius, offset=0):
    offset = blur_radius * 2 + offset
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    result = pil_img.copy()
    result.putalpha(mask)
    result = result.resize((64,64), Image.LANCZOS)

    return result    
    
def save_thumbnail(path, filename, image, pos=(), extension='jpg', padding=0, crop='square'):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    filepath = os.path.join(path, "{}.{}".format(filename, extension))
    
    if padding:
#         pos = (left, top, right, bottom)
        pos = list(pos)
        pos[0] -= padding 
        pos[1] -= padding
        pos[2] += padding
        pos[3] += padding
        pos = tuple(pos)
        
    new_image = image.crop(pos)
    
    if crop == 'circle':
        new_image = mask_circle_transparent(new_image, 0)
    
    new_image.save(filepath, quality=90)
    
def compress_and_save(path, list_of_names, list_of_array):
    data = {name:array for array, name in zip(list_of_array, list_of_names)}
    np.savez_compressed(path, **data)
    
def load_from_npz(path):
    if not os.path.isfile(path):
        return [], [], []
    
    tmp = np.load(path)
    hash_list = [hash_array(array) for array in tmp.values()]
    return list(tmp.keys()), hash_list, list(tmp.values())

def load_known_encodings(npz_file, npy_folder=None):
    names, hash_list, encodings = load_from_npz(npz_file)
    
    if npy_folder and not names or not encodings:
        _logger.debug("<<Switch back to npy files>>")
        names, hash_list, encodings = load_from_subdirs(npy_folder)
    
    return names, hash_list, encodings



def get_equation(series):
    x = [serie[0] for serie in series]
    y = [serie[1] for serie in series]
    min_value, max_value = min(x),max(x)
    
    return Polynomial.fit(x, y, 1, window=[min_value,max_value])
    
#     x = [0.1, 0.55]
#     y = [90, 50]
