# dmri\_segmenter
[![pypi badge](https://img.shields.io/pypi/v/dmri_segmenter.svg)](https://pypi.python.org/pypi/dmri_segmenter)
[![travis-ci badge](https://img.shields.io/travis/captainnova/dmri_segmenter.svg)](https://travis-ci.org/captainnova/dmri_segmenter)
[![Documentation Status](https://readthedocs.org/projects/dmri-segmenter/badge/?version=latest)](https://dmri-segmenter.readthedocs.io/en/latest/?badge=latest)

- Eventual readthedocs.io documentation: (https://dmri-segmenter.readthedocs.io)

## About
This package includes both a program, skullstrip_dmri, already trained to strip
skulls from in vivo adult human diffusion MR images, and software for
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
- python 3.6+
- dipy
  - Dependencies:
    - nibabel
    - numpy
    - (If installing via pip): 
      - sudo apt install python-dev (Debian and its derivatives)
      - sudo yum install python-devel (Red Hat and its derivatives)
- future
- scikit-image
- scikit-learn
- if using scikit-learn > 0.23.2 (very likely): onnxruntime
- optional (needed for saving trained classifiers in onnx format, not for day-to-day
  use): skl2onnx

## Versions
- 2.3.0 released 2024-06-27 Improved support for problem data with badly
  placed FOVs, inoptimally corrected susceptibility distortion, and CSF that is
  much brighter than brain tissue at b = 0. (i.e. a ratio ~ 4 instead of ~2).
  - The CSF/brain brightness ratio is now empirically estimated using a 1st
    pass at the tissue class probabilities, so for better or worse this is
    making every scan's runtime longer for the benefit of a few. Presumably
    this is worth it since otherwise dealing with those few would be very
    time consuming.
  - It now also uses internal bias field correction to ameliorate "signal
    shading".  Don't get too excited - it doesn't make much difference to the
    segmentation, the field it is correcting is a combination of the coil
    sensitivity map and signal dilution by susceptibility distortion; it is not
    the classic sensitivity map.
- 2.0.0 released 2024-01-03 (onnx support + improvements from using
  `morphological_geodesic_active_contour`)
  - If you had trouble before with either loading the classifier, or
    challenging images with iffy receiver coils and/or nasty EPI distortion,
    2+ should help a lot - please give it a try!
- 1.3.0 released 2021-06-30 (restores compatibility with recent scikit-learn
  versions, e.g. 0.24.+)
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

time dmri_segmenter/train_from_multiple training.srclist all dtb_eddy_fvecs.nii &
real	1m0.472s
user	1m50.555s
sys	0m6.735s
```

### Comparison to Other Skull Strippers for dMRI
For a detailed but dated comparison of old skullstrip\_dmri (v. 1.0.0) to
competing older skull strippers, see the 2018 reference below. In the meantime,
an interesting deep learning based skull stripper has arrived on the scene:
[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/), which is
included in FreeSurfer 7+.

Wait, FreeSurfer, you say - does that mean it's for T1-weighted images? Well,
sort of. They trained with T1w inputs, but only after applying all sorts of
corruptions to the contrast to prevent the network from getting attached to any
particular type of contrast. It works quite well with dMRI, and has a similar
runtime to dmri\_segmenter. (Surprisingly, mri\_synthstrip's GPU option seems
to make it slower.) The results are different in the details,
though. dmri\_segmenter deliberately avoids using a neural network, and does
not have a strong prior for the overall shape and size of a skull. (It does
have expectations for the diffusion properties of tissue and CSF.) SynthStrip
is not picky about the contrast, but does have a strong prior of what a skull
should look like, and as far as I can tell, EPI distortion was _not_ included
in the list of perturbations for the training data. Thus SynthStrip tends to
miss parts that have been stretched out by EPI distortion, but on the other
hand can get places like under the recti where the signal has completely
dropped out. Since there is no signal there anyway, and I am biased, I prefer
dmri\_segmenter. But if you're lucky enough to not have to deal with severe EPI
distortion then SynthStrip offers the convenience of one skull stripper for all
scan types.

I also noticed that mri\_synthstrip's --no-csf option includes the CSF in the
ventricles, which is most of the CSF in older people! I don't think that sort
of segmentation is the main point of either mri\_synthstrip or dmri\_segmenter,
though.

Since dmri\_segmenter supports using a "T1"-based TIV mask as a prior (that it
blurs in the y direction to account for EPI distortion), you can use the output
of SynthStrip (or any other stripper that makes a mask .nii) as a suggestion
for dmri\_segmenter. Unsurprisingly, the result tends to be somewhere between
the suggestion and what dmri\_segmenter would produce by itself.

## Acknowledgements
The images used to train the default version of the classifier
(RFC_ADNI6.pickle) came from [the Alzheimer's Disease Neuroimaging Initiative
(ADNI)](http://adni.loni.usc.edu/).

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.


## References
Reid, R. I. et al. Diffusion Specific Segmentation: Skull Stripping with Diffusion MRI Data Alone. in Computational Diffusion MRI (eds. Kaden, E., Grussu, F., Ning, L., Tax, C. M. W. & Veraart, J.) 6780 (Springer, Cham, 2018). doi:10.1007/978-3-319-73839-0_5
