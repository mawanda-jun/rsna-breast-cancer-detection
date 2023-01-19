# DATA INSIGHTS
## Composition
- There are several patients.
- Mammographies comes with two main information: presence of cancer (0 for no, 1 for yes) and laterality (L for left, R for right).
- For each patient, there are: 
    - At least 4 mammographies: 2 for L, 2 for R.
    - There are at maximum 8 mammographies for R, 7 for L.
    - On average, there are ~2.3 for each laterality, both among the positives and the negatives.
- We are asked to find the cancer probability of a patient, for each laterality. Therefore, for each patient, there will be two predictions. The key is `{patient_id}_{laterality}`.
    - -> might be possible to use siamese networks? Pass each image to the network, average the representations, then apply for the binary classification. We should be able to use off the mammographies!
- Given a patient and a laterality, even if there are more than one image for laterality, each image will have the same label.
- High dataset imbalance!
    - 492 positive patients w/ laterality, 23334 negatives;
    - negative samples has the patients with higher number of mammograhies, but mean is quite similar to the positive ones.
    - Division for 20% validation: 98 positive patients_ids, 4667 negative ids.
    - -> Use of weighted loss?
    - -> Dataset reduction/lots of augmentations.


## What to test?
### Metrics
- Competitions measures the F1 score. Other metrics might be interesting.
### Losses
- Dice loss seems to be the most promising. Might use also other.

### TODO
1. Find LR for v2s models, batch size 60
2. Val loss is too high! We must reduce it. It might be due to the fact that the positives are too easy during training, and overfitting (for the positives) occurs:
    - less positives? But we like balanced trainings!
    - weighted loss, where errors on positives counts more?

