import numpy as np
import os
import csv

root = os.path.dirname(__file__)
cn_path = os.path.join(root, 'aml-2025-feathers-in-focus', 'class_names.npy')
attr_path = os.path.join(root, 'aml-2025-feathers-in-focus', 'attributes.npy')

# Load both files
class_names = np.load(cn_path, allow_pickle=True)
attributes = np.load(attr_path, allow_pickle=True)

# Extract the stored object
class_names = class_names.item()  

rows = []
for item in class_names:
    item_str = str(item)
    parts = item_str.split('.', 1)
    rows.append((parts[0], parts[1]))

rows.sort(key=lambda x: int(x[0]))

with open('class_names.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'Class_name'])
    writer.writerows(rows)

print("Class names CSV saved as 'class_names.csv'")