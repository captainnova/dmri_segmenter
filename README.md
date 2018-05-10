## About

## License

## Dependencies

## Versions

## Skull Stripping

## Segmenting

## Training the Classifier

dmri_segmenter includes a classifier which was already trained with scans from
older adults, but you might want to train a classifier with your own data.
Give the stock classifier a try first, though - although dmri_segmenter uses some
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

make_fvecs GE/0/dtb_eddy.nii
make_fvecs GE/1/dtb_eddy.nii
make_fvecs Siemens/0/dtb_eddy.nii
make_fvecs Siemens/1/dtb_eddy.nii
make_fvecs Philips/0/dtb_eddy.nii
make_fvecs Philips/1/dtb_eddy.nii

### Making Manual Segmentations

I start with a trial segmentation from the stock classifier and edit the
results with fsleyes.

dmri_segment -o GE/0/dmri_segment.nii GE/0/dtb_eddy_fvecs.nii ${path_to_stock_classifier}/RFC_classifier.pickle
... (parallelizable)

cd GE/0; fsleyes dtb_eddy_fvecs.nii dmri_segment.nii & # Save to dmri_segment_edited.nii
... (Sadly only parallelizable if using multiple human segmenters)

### Running the Trainer

# List the training directories into a file.
echo [GPS]*/* | sed 's/ /\n/g' > all.srclist

time /mnt/data/m101013/src/dmri_segmenter/train_from_multiple training.srclist all dtb_eddy_fvecs.nii &
real	1m0.472s
user	1m50.555s
sys	0m6.735s


## References
