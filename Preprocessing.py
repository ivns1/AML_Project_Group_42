import numpy as np
import os

root = os.path.dirname(__file__)
cn_path = os.path.join(root, 'aml-2025-feathers-in-focus', 'class_names.npy')
attr_path = os.path.join(root, 'aml-2025-feathers-in-focus', 'attributes.npy')

# Load both files
class_names = np.load(cn_path, allow_pickle=True)
attributes = np.load(attr_path, allow_pickle=True)

print('class_names raw dtype:', getattr(class_names, 'dtype', None))


#print(class_names)
