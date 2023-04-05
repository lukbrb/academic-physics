#!/bin/bash

echo "[*] Creating validation directories..."
mkdir imagedata/validation
mkdir imagedata/validation/spiral
mkdir imagedata/validation/elliptical

echo "[+] Validation directories created"

echo "[*] Creating test directories..."
mkdir imagedata/test
mkdir imagedata/test/spiral
mkdir imagedata/test/elliptical


echo "[+] Test directories created"

echo "[*] Moving 2000 images to validation/spiral folders..."
./mv_validation_imgs.sh imagedata/train/ imagedata/validation spiral 2000
echo "[+] Done!"

echo "[*] Moving 2000 images to validation/elliptical folders..."
./mv_validation_imgs.sh imagedata/train/ imagedata/validation elliptical 2000
echo "[+] Done!"

echo "[*] Moving 5000 images to test/spiral folders..."
./mv_validation_imgs.sh imagedata/train/ imagedata/test spiral 5000
echo "[+] Done!"

echo "[*] Moving 5000 images to test/elliptical folders..."
./mv_validation_imgs.sh imagedata/train/ imagedata/test elliptical 5000
echo "[+] Done!"

echo "[+] The setup of images has finished successfully"