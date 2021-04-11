# Implementation of Personalized Pairwise Novelty Weighting (PPNW)



This is an implementation of **[our paper PPNW](https://link.springer.com/article/10.1007/s10115-021-01546-8)** for **"Knowledge and Information Systems (Springer)"**:
```
Kachun Lo, Tsukasa Ishigaki, "PPNW: Personalized Pairwise Novelty Loss Weighting for Novel Recommendation"
```

<p align="center">
  <img align="center" src="https://github.com/ArgentLo/PPNW-KAIS/blob/master/PPNW_structure.png" width="440" height="431"/>
</p>


In this repository, all the following parts are included to **support reproductivity** of the manuscript.

  - **Two datasets** used in the paper.
  - **Accuracy-Focused Base Models** for comparison.
  - The proposed **PPNW Framework**.
  - **Quick Start** instruction for Pre-training & Training.

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


#### Pretraining on General Matrix Factorization (GMF)

```shell
sh pretrain.sh
```
- Use/Not Use "Novelty Weighting" by setting `--use_unpop_weight` in `pretrain.sh`.

#### Training Base Model (CMN) with PPNW

```shell
sh train.sh
```

- Use/Not Use "Novelty Weighting" by setting `--use_unpop_weight` in `train.sh`.
- Recommended Settings for different datasets can be found in comment of `train.sh`.

----

### License
 
The MIT License (MIT)

Copyright (c) 2020 Argent Lo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



