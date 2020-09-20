## Implementation of Personalized Pairwise Novelty Weighting (PPNW)


This is an implementation of PPNW for the manuscript submitted to **"Knowledge and Information Systems 2020 (Springer)"**:
```
Kachun Lo, Tsukasa Ishigaki, "PPNW: Personalized Pairwise Novelty Loss Weighting for Novel Recommendation"
```

In this repository, all the following parts are included to **support reproductivity** of the manuscript.

  - **Two datasets** used in the paper.
  - **Accuracy-Focused Base Models** for comparison.
  - The proposed **PPNW Framework**.
  - **Quick Start** instruction for Pre-training & Training.

This repository uses the accuracy-focused Base Model (**CMN**): https://arxiv.org/pdf/1804.10862.pdf

----
 
### Recommended Environment

```shell
Python 3.5
TensorFlow 1.15.0
dm-sonnet
networkx
```

### Dataset

Since implicit feedback data are considered in our work, all **data values are binarized**. 

For all dataset, 80% of a user’s historical items would be randomly sampled as the training set and the rest items are collected as the test set.

Please **download** the preprocessed datasets before runnning the code.

- Citeulike-a (19M) :

  ```
  https://drive.google.com/open?id=1mW5UD8Ds29fN0lH9JcvBuf-yAg_ZYdWl
  ```

- MovieLens-1M (76M):

  ```
  https://drive.google.com/open?id=1rwGV60iK_Cqtx82J3DoV0IMMA8seaMBq
  ```

----

### Quick Start

To help get started smoothly, we provide default settings of PPNW for pretraining and training.


#### Pretraining on GMF (CMN required)

```shell
sh pretrain.sh
```
- Use/Not Use "Novelty Weighting" by setting `--use_unpop_weight` in `pretrain.sh`.

#### Training Base Model with PPNW

```shell
sh train.sh
```

- Use/Not Use "Novelty Weighting" by setting `--use_unpop_weight` in `train.sh`.
- Recommended Settings for different datasets can be found in comment of `train.sh`.

