import os
import shutil
import pandas as pd

# Charger la table de passage avec les classes et les noms de fichier
table = pd.read_csv('../data/table_passage.csv')

# Créer les dossiers pour chaque classe
spiral_dir = 'imagedata/train/spiral'
elliptical_dir = 'imagedata/train/elliptical'
uncertain_dir = 'imagedata/train/uncertain'
extension = ".jpg"
os.makedirs(spiral_dir, exist_ok=True)
os.makedirs(elliptical_dir, exist_ok=True)
os.makedirs(uncertain_dir, exist_ok=True)

# Parcourir chaque ligne de la table et copier les fichiers dans les dossiers correspondants
spiral_count = 0
elliptic_count = 0
uncertain_count = 0
compteur = {0:0, 1:0, 2:0}
file_not_found = []
max_images_per_folder = 17_000  # 10 000 for training, 2000 for validation, 5000 for testing
for _, row in table.iterrows():
    class_label = row['classe']
    filename = str(row['asset_id'])
    print(class_label)
    # Sélectionner le dossier de destination en fonction de la classe
    if compteur[class_label] == max_images_per_folder:
        continue
    if class_label == 0:
        dest_dir = spiral_dir
    
    elif class_label == 1:
        dest_dir = elliptical_dir
    else:
        dest_dir = uncertain_dir
    
    # Copier le fichier dans le dossier de destination
    src_file = os.path.join('/Users/lucas/Documents/academic-physics/galaxy-classification/', 'images', filename + extension)
    dest_file = os.path.join(dest_dir, filename + extension)
    if os.path.isfile(src_file) and not os.path.isfile(dest_file):  # make sure the source file exists, and that it's not already in the dest folder. Seems there are doubles.
        compteur[class_label] += 1
    try:
        shutil.copyfile(src_file, dest_file)
        print("Moved :", src_file, "to", dest_file)
    except FileNotFoundError:
        print(f"File {src_file} could not be found")
        file_not_found.append(src_file)

print(f"[*] {compteur[0]} spiral galaxies have been moved to {spiral_dir}")
print(f"[*] {compteur[1]} spiral galaxies have been moved to {elliptical_dir}")
print(f"[*] {compteur[2]} spiral galaxies have been moved to {uncertain_dir}")

# print("These following images haven't been found", file_not_found)
print(f"{len(file_not_found)} could not be found out of {len(table)}")