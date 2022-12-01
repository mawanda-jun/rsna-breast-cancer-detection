# DATA INSIGHTS
## Composition
- There are several patients. 
- For each patient, there are: 
    - At least 4 mammographies: 2 for L, 2 for R.
    = There are at maximum 8 mammographies for R, 7 for L.
- We are asked to find the cancer probability of a patient, for each laterality. Therefore, for each patient, there will be two predictions. The key is `{patient_id}_{laterality}`
    - -> might be possible to use siamese networks? Pass each image to the network, average the representations, then apply for the binary classification.
- Given a patient and her laterality, even if there are more than one image, each image will have the same label.
- High dataset imbalance!
    - Use of weighted loss?
    - Dataset reduction/lots of augmentations.

## What to test?
### Metrics
- Competitions measures the F1. Other metrics might be interesting.
### Losses
- Dice loss seems to be the most promising. Might use also other.

