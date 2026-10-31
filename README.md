# Binary Classification of Sea Ice Image Patches with a CNN

This project trains a small convolutional neural network (CNN) to separate two classes of 128 x 128 single-channel satellite image patches, stored as `NI_6s.npy` and `OW_6s.npy`. All code is in [`seaice_iceberg_Classification.py`](seaice_iceberg_Classification.py).

---

## Dataset

* **Files:** `NI_6s.npy` and `OW_6s.npy` (IEEE Dataport). The data files are not included in this repository.
* **Labels used in the code:** every patch in `NI_6s.npy` is labelled 0 and every patch in `OW_6s.npy` is labelled 1. The script originally called these "sea ice" and "iceberg".
* **To check:** in sea ice remote sensing, NI and OW often mean *new ice* and *open water*. The class names in this project should follow the dataset's own documentation. Source link and citation: *to be added*.

---

## Model

* Three convolution blocks (32, 64 and 128 filters, 3 x 3, ReLU) each followed by 2 x 2 max pooling
* Dense (256, ReLU), Dropout (0.5), Dense (128, ReLU), Dropout (0.5), Dense (1, sigmoid)
* Adam optimiser, binary cross-entropy loss, batch size 32, up to 50 epochs
* Early stopping on validation loss (patience 10) and best-model checkpoint
* Random 80/20 train and validation split

The script prints validation accuracy, a confusion matrix and a classification report, and plots accuracy and loss curves.

---

## Results

Figures from an earlier run:

![1](https://github.com/user-attachments/assets/07248ede-470f-40b8-8bd1-2dcdbcacdcee)
*Training and validation accuracy.*

![2](https://github.com/user-attachments/assets/a98d8a48-4206-4aa5-b469-efbe0a7c3417)
*Training and validation loss.*

The classification report and confusion matrix previously shown here have been removed. The earlier script could not produce them correctly: it had a syntax error, and the validation generator shuffled the images, so predictions were compared with labels in a different order. Run the fixed script to get valid numbers.

---

## Fixes in the Current Script

* Fixed a syntax error in the plot legend that stopped the script from running
* Validation predictions now use the images in their original order, so the confusion matrix and report are correct
* Added the missing `ListedColormap` import
* Removed a "simulated classified image" that was generated from random numbers and did not come from the model

---

## Limitations

* Each class comes from a separate file, so the model may learn differences between the two files (for example, acquisition date or processing) rather than the surface type itself
* No separate test set; the validation set is also used for early stopping
* No data augmentation or class balancing

---

## How to Run

```bash
pip install -r requirements.txt
# place NI_6s.npy and OW_6s.npy in the project folder
python seaice_iceberg_Classification.py
```

---

## Tools

Python, NumPy, TensorFlow/Keras, scikit-learn, Matplotlib, Seaborn
