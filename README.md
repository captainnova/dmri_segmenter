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

## License
Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).

## Dependencies
python 2.6+, not yet tested with python 3.
dipy
nibabel
numpy

## Versions
1.1.0 released 2019-07-28.

## Skull Stripping
Don't worry - you probably won't need the options!
```
Usage:
  skullstrip_dmri [-c=CR -d=D -i=FL -m=MR -n=NM -t=BRFN --verbose=VE -s=SVC -w] <ecfn> <bvals> <tivfn>
  skullstrip_dmri [--verbose=VE] -p <ecfn> <bvals> <tivfn>
  skullstrip_dmri (-h|--help|--version)

Arguments:
  ecfn:    Filename of an eddy corrected 4D dMRI nii.
  bvals:   Name of an ASCII file holding the diffusion strengths in a single
           space separated row.
  tivfn:   Filename for a TIV-style type output nii.

Options:
  -h --help               Show this message and exit.
  --version               Show version and exit.
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
  -i FL --isFL=FL         Specify whether it is (1) or is not (0) a FLAIR
                          diffusion scan.  It defaults to trying to find that
                          from the InversionTime (if available) and the CSF
                          to tissue brightness ratio.
  -m MR --mr=MR           Radius of the median filter relative to the largest
                          voxel size.
                          [default: 1]
  -n NM --nmed=NM         Number of times to run the median filter
                          [default: 2]
  -p --phantom            Use for phantoms.
  -s SVC --svc=SVC        Pickle or joblib dump file holding the classifier
                          parameters. (Not used for FLAIR.)
                          [default: RFC_classifier.pickle]
  -t BRFN --brfn=BRFN     If given, also write a tighter brain mask to BRFN.
  --verbose=VE            Be chatty.
                          [default: 1] (True)
  -w --whiskers           If given, do NOT make an extra effort to trim whiskers.
```

## Segmenting

## Training the Classifier

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

# List the training directories into a file.
```sh
echo [GPS]*/* | sed 's/ /\n/g' > all.srclist

time /mnt/data/m101013/src/dmri_segmenter/train_from_multiple training.srclist all dtb_eddy_fvecs.nii &
real	1m0.472s
user	1m50.555s
sys	0m6.735s
```

## References
