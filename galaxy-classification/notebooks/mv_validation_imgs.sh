#!/bin/bash

# Récupération des arguments
dir_from=$1
dir_to=$2
class=$3
n_samples=$4

# Vérification des arguments
if [ $# -ne 4 ]; then
  echo "Usage: $0 dir_from dir_to class n_samples"
  exit 1
fi

if [ ! -d "$dir_from" ]; then
  echo "Error: $dir_from is not a directory"
  exit 1
fi

if [ ! -d "$dir_to" ]; then
  echo "Error: $dir_to is not a directory"
  exit 1
fi

if [[ ! "$class" =~ ^(spiral|elliptical|uncertain)$ ]]; then
  echo "Error: class must be 'spiral', 'elliptical', or 'uncertain'"
  exit 1
fi

if [ $n_samples -le 0 ]; then
  echo "Error: n_samples must be positive"
  exit 1
fi

# Sélection aléatoire de $n_samples fichiers
files=($(ls -1 "$dir_from/$class/" | sort -R | head -n "$n_samples"))

# Copie des fichiers dans le dossier de destination
for file in "${files[@]}"; do
  mv "$dir_from/$class/$file" "$dir_to/$class"
done

