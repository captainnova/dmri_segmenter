# dmri\_segmenter
[![pypi badge](https://img.shields.io/pypi/v/dmri_segmenter.svg)](https://pypi.python.org/pypi/dmri_segmenter)
[![travis-ci badge](https://img.shields.io/travis/captainnova/dmri_segmenter.svg)](https://travis-ci.org/captainnova/dmri_segmenter)
[![Documentation Status](https://readthedocs.org/projects/dmri-segmenter/badge/?version=latest)](https://dmri-segmenter.readthedocs.io/en/latest/?badge=latest)

- Eventual readthedocs.io documentation: (https://dmri-segmenter.readthedocs.io)

## About
This package includes both a program, skullstrip_dmri, already trained to strip
(adult human) skulls from in vivo diffusion MR images, and software for
training it to recognize tissue classes for other types of diffusion MRI (dMRI)
subjects. Skull stripping is typically needed, or at least wanted, early in a
dMRI processing pipeline to prevent divisions by zero, avoid computations on
irrelevant voxels, and aid registration between images.

To avoid a chicken-and-egg problem, skullstrip_dmri typically operates without
needing bias correction or a T1-weighted image, and mainly relies on the
diffusion properties of voxels to classify them. It uses a random forest
classifier from machine learning, and (so far!) is fairly tolerant of changes
in scan protocol such as b value, voxel size, and scanner manufacturer.

dmri_segment can do basic tissue classification (brain, CSF, air/extracranial
tissue and "other" (tentorium, etc.)), but the main purpose of this package is
to produce a mask for separating the brain, CSF, and other classes, e.g. the
total intracranial volume (TIV), from extracranial voxels. The difference
between brain and "other" is particularly fuzzy - "other" is just anything
determined to be in the TIV which is not obviously brain or CSF. Since dMRI
tends to suffer from large distorted voxels we expect that most users will use
a 3D acquisition (e.g. T1w and/or FLAIR) for a more precise measure of the
brain volume.

## Note
You probably do NOT need to train your own classifier, or worry about most of
skullstrip\_dmri's options, unless you are stripping unusual brains (e.g. phantoms).

## License
Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).

## Dependencies
All but the first two can be installed with pip(env).
- POSIX? (not yet tested with Windows)
- python 2.7, or python 3.6+
- dipy
  - Dependencies:
    - nibabel
    - numpy
    - (If installing via pip): 
      - sudo apt install python-dev (Debian and its derivatives)
      - sudo yum install python-devel (Red Hat and its derivatives)
- future (for python 2/3 compatibility)
- scikit-image
- scikit-learn
- if using scikit-learn > 0.23.2: onnxruntime
- optional (needed for saving trained classifiers in onnx format, not for day-to-day
  use): skl2onnx

## Versions
- 1.2.0 released 2019-09-29 (adds python 3 compatibility)
- 1.1.0 released 2019-07-28 (1st release on github, after > 1 year of in-house use.)

## Skull Stripping
```
Usage:
  skullstrip_dmri [-i=FL -t=BRFN --verbose=VE] [-c=CR -d=D -m=MR -n=NM -s=SVC -w] DATA BVALFN OUTFN
  skullstrip_dmri [--verbose=VE] -p DATA BVALFN OUTFN
  skullstrip_dmri (-h|--help|--version)

Arguments:
  DATA:    Filename of a 4D dMRI nii, ideally but not necessarily eddy corrected
  BVALFN:  Name of an ASCII file holding the diffusion strengths in a single
           space separated row.
  OUTFN:   Filename for the TIV (or equivalent) output nii.

Common Options:
  -h --help               Show this message and exit.
  --version               Show version and exit.
  -i FL --isFL=FL         Save some time but specifying whether it is (1) or
                          is not (0) a FLAIR diffusion scan. It defaults to
                          trying to find that from the InversionTime (if
                          available) and/or the CSF to tissue brightness ratio.
  -p --phantom            Use for phantoms.
  -t BRFN --brfn=BRFN     If given, also write a tighter brain mask to BRFN.
  --verbose=VE            Be chatty.
                          [default: 1] (True)

Options intended for animal dMRI:
  -c CR --cr=CR           Closing radius relative to the maximum voxel size
                          with which to close the mask before filling holes.
                          [default: 3.7]
  -d D --dil=D            Controls dilation.
                          **N.B.: it only affects FLAIR DTI!**
                          If a positive number, it will be used as the radius,
                          relative to the maximum voxel size, to dilate with.
                          If a nonnegative number, no dilation will be done.
                          If y or t (case insensitive), mr * nmed will be
                          used.
                          [default: 0.5]
  -m MR --mr=MR           Radius of the median filter relative to the largest
                          voxel size.
                          [default: 1]
  -n NM --nmed=NM         Number of times to run the median filter
                          [default: 2]
  -s SVC --svc=SVC        Pickle or joblib dump file holding the classifier
                          parameters. (Not used for FLAIR.)
                          [default: RFC_classifier.pickle]
  -w --whiskers           If given, do NOT try harder to trim whiskers.
```

## Segmenting
dmri\_segmenter is primarily intended for skull stripping, but it works by
classifying voxels as air, scalp/face, eyeball, CSF, brain tissue, or
intracranial "other" (e.g. tentorium). Thus, you can get it to segment using
either dmri\_segment or skullstrip\_dmri -b. Just be aware that the sum of CSF,
tissue, and other tends to be more accurate than the individual
components. Both dmri\_segment and skullstrip\_dmri, support working from raw
data (otherwise there would be a chicken and egg problem), so they avoid
certain kinds of calculations that a segmenter designed to work with processed
data might use.

## Do I really need all those classifiers?
Probably not. The ones included here so far are the same data in different
formats, as needed by different python environments.

| Python environment                         | Matching Classifier      |
|--------------------------------------------|--------------------------|
| 2                                          | RFC\_ADNI6\_py27.pickle  |
| 3, with onnxruntime                        | RFC\_ADNI6\_onnx         |
| 3, with sklearn < 0.24 and no onnxruntime  | RFC\_ADNI6\_sk0p23.pickle |
| 3, with sklearn >= 0.24 and no onnxruntime | RFC\_ADNI6\_sk0p24.pickle |

skullstrip\_dmri will attempt to load the correct one by default.

## Training your own Classifier

dmri\_segmenter includes a classifier which was already trained with scans from
older adults, but you might want to train a classifier with your own data.
Give the stock classifier a try first, though - although dmri\_segmenter uses some
morphological information that can specialize it to the typical anatomy of the
training set, it mostly relies on the mean diffusivity and T2 properties of
tissue, which do not change as much from person to person.

While preparing the CDMRI paper[1] we were concerned that a classifier trained
with data from one scanner or person might not be applicable to scans from
other people or scanners, so we compared classifiers trained with a wide
variety of combinations of scanners and subjects.  Fortunately, the result was
that what mostly matters is that the training scans are relatively free of
artifacts and have good spatial resolution.  Another way of thinking about it
is to consider every voxel as a sample, so a single scan provides hundreds of
thousands of samples with a wide variety of conditions.  That might be a bit
optimistic, but you will find that you want to keep your training and test sets
small because of the manual segmentation step.

### Making Feature Vectors
When training things are a bit more fragmented than in skull stripping.  The
first step, making feature vector images, is easily parallelizable with a grid
engine.
```sh
make_fvecs GE/0/dtb_eddy.nii
make_fvecs GE/1/dtb_eddy.nii
make_fvecs Siemens/0/dtb_eddy.nii
make_fvecs Siemens/1/dtb_eddy.nii
make_fvecs Philips/0/dtb_eddy.nii
make_fvecs Philips/1/dtb_eddy.nii
```

### Making Manual Segmentations

I start with a trial segmentation from the stock classifier and edit the
results with fsleyes.

```sh
dmri_segment -o GE/0/dmri_segment.nii GE/0/dtb_eddy_fvecs.nii ${path_to_stock_classifier}/RFC_classifier.pickle
#... (parallelizable)

cd GE/0; fsleyes dtb_eddy_fvecs.nii dmri_segment.nii & # Save to dmri_segment_edited.nii
#... (Sadly only parallelizable if using multiple human segmenters)
```

### Running the Trainer
```sh
# List the training directories into a file.
echo [GPS]*/* | sed 's/ /\n/g' > all.srclist

time /mnt/data/m101013/src/dmri_segmenter/train_from_multiple training.srclist all dtb_eddy_fvecs.nii &
real	1m0.472s
user	1m50.555s
sys	0m6.735s
```

## Acknowledgements
The images used to train the default version of the classifier
(RFC_ADNI6.pickle) came from [the Alzheimer's Disease Neuroimaging Initiative
(ADNI)](http://adni.loni.usc.edu/).

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.


## References
Reid, R. I. et al. Diffusion Specific Segmentation: Skull Stripping with Diffusion MRI Data Alone. in Computational Diffusion MRI (eds. Kaden, E., Grussu, F., Ning, L., Tax, C. M. W. & Veraart, J.) 6780 (Springer, Cham, 2018). doi:10.1007/978-3-319-73839-0_5
