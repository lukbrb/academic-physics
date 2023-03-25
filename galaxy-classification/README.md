# Galaxy Classification

This project aims to give a tour of the Scikit-learn library capabilities, and to give some concrete examples on how to use classifiers for real world problems.
The problem we tackle here is the classification of galaxies. The data used comes from the Galaxy Zoo project.

*Note: This project is based on the excellent sklearn tutorial from Jake VanderPlas. The original GitHub repository can be found [here](https://github.com/jakevdp/sklearn_tutorial)*

## Dataset presentation

### Galaxy Zoo1 Catalog Data

This dataset contains information on galaxy classifications made by participants of the Galaxy Zoo project.
The file can be found at the following URL: <https://galaxy-zoo-1.s3.amazonaws.com/GalaxyZoo1_DR_table2.csv.gz>

This table gives classifications of galaxies which have spectra included in SDSS Data Release 7.
The fraction of the vote in each of the six categories is given, along with debiased votes in elliptical and spiral categories and
flags identifying systems as classified as spiral, elliptical or uncertain.

#### Column Descriptions

1. OBJID: Unique identifier of the galaxy in the SDSS catalog.
2. RA: Right Ascension (in degrees) of the galaxy.
3. DEC: Declination (in degrees) of the galaxy.
4. NVOTE: Number of votes obtained for this galaxy.
5. P_EL: Probability that the galaxy is elliptical.
6. P_CW: Probability that the galaxy is a clockwise spiraled galaxy.
7. P_ACW: Probability that the galaxy is a anticlockwise spiraled galaxy.
8. P_EDGE: Probability that the galaxy is a galaxy with a blurred edge.
9. P_DK: Probability that the galaxy is a galaxy with a dominant nucleus.
10. P_MG: Probability that the galaxy is a galaxy with multiple nuclei.
11. P_CS: Probability that the galaxy is a galaxy with strange features.
12. P_EL_DEBIASED: Probability of the elliptical classification, corrected for bias effect.
13. P_CS_DEBIASED: Probability of the classification with strange features, corrected for bias effect.
14. SPIRAL: Number of votes for the "spiral" classification.
15. ELLIPTICAL: Number of votes for the "elliptical" classification.
16. UNCERTAIN: Number of votes for the "uncertain" classification.

Each of these columns is used to describe the classification of a given galaxy by Galaxy Zoo participants.
The P_* columns contain classification probabilities for each type of galaxy, while the SPIRAL, ELLIPTICAL,
and UNCERTAIN columns provide the number of votes for each classification. The P_*_DEBIASED columns are corrected for bias effect.

## Installation Notes

**Download the project:**

``` shell
$ cd /<desired-diretory>
```

Then :

``` shell
$ git clone https://github.com/lukbrb/academic-physics/tree/master

$ mv academic-physics/galaxy-classification/ .

$ cd galaxy-classification
```

This will download the whole `academic-projects` directory, then move the `galaxy-classification` project in your current directory. Finally,
it will remove the whole `academic-physics` folder and subfolders.

**Requirements:**  

``` shell
$ conda create -n glx_clf python=3.10.8 --file requirements.txt
```

``` shell
$ activate glx_clf
```

``` shell
$ jupyter notebook --notebook-dir='<tutorial folder>'
```
