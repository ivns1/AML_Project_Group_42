import numpy as np
import os
import csv

root = os.path.dirname(__file__)
cn_path = os.path.join(root, 'aml-2025-feathers-in-focus', 'class_names.npy')
attr_path = os.path.join(root, 'aml-2025-feathers-in-focus', 'attributes.npy')
attr_names_path = os.path.join(root, 'aml-2025-feathers-in-focus', 'attributes.txt')

class_names = np.load(cn_path, allow_pickle=True)
attributes = np.load(attr_path)

# Class names preprocessing
class_names = class_names.item()  

rows = []
for item in class_names:
    item_str = str(item)
    parts = item_str.split('.', 1)
    rows.append((parts[0], parts[1]))

rows.sort(key=lambda x: int(x[0]))

with open('class_names.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Label', 'Class_name'])
    writer.writerows(rows)

print("Class names CSV saved as 'class_names.csv'")


# Attribute names preprocessing
rows = []
with open(attr_names_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        id_part, rest = line.split(" ", 1) 
        group, name = rest.split("::", 1) 

        rows.append((int(id_part), group, name))


rows.sort(key=lambda x: x[0])

with open('attribute_names.csv', "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "attribute_group", "attribute"])
    writer.writerows(rows)

print("Attribute names CSV saved as 'attribute_names.csv'")



# Attributes for each class

with open('attributes.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Class'] + [f'{i}' for i in range(attributes.shape[1])])
    
    for idx in range(attributes.shape[0]):
        writer.writerow([idx] + attributes[idx].tolist())
        
print("Attributes CSV saved as 'attribute_scores.csv'")