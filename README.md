# Deep_Image_Learning

For scores [99.72, 98.93]:

#### Data preprocessing:

Only using original MNIST dataset,

Normalise, augment, optimise, call-backs. Train for 100e.

Train on augmented examples:

rotation_range=10,

width_shift_range=0.1,

height_shift_range=0.1,

shear_range=0.2,

zoom_range=0.1,

fill_mode='nearest',

preprocessing_function=lambda x: x + tf.random.normal(tf.shape(x), stddev=0.05)  # Add Gaussian noise

Optimizer, Loss function:

Adam with lr = 0.001

CategoricalCrossentropy with label smoothing = 0.1


#### Callbacks:

ReduceLROnPlateau with learning_rate_patience = 20, learning_rate_decay = 0.2

ModelCheckpoint to save best checkpoint only

EarlyStopping

Trained for ~100 epochs.

Model:
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
===================================================

(InputLayer)    [(None, 28, 28, 1)]       0

(Conv2D)       (None, 26, 26, 32)        128

(BatchNorm)  (None, 26, 26, 32)       128

(Conv2D)       (None, 24, 24, 32)        9216

(BatchNorm  (None, 24, 24, 32)       128

(Conv2D)       (None, 12, 12, 32)        25632

(Dropout)   (None, 12, 12, 32)        0

(Conv2D)       (None, 10, 10, 64)        18432

(BatchNorm)  (None, 10, 10, 64)       256

(Conv2D)       (None, 8, 8, 64)          36864

(BatchNorm)  (None, 8, 8, 64)         256

(Conv2D)       (None, 4, 4, 64)          102464

(Dropout)   (None, 4, 4, 64)          0

(Conv2D)       (None, 2, 2, 128)         204928

(Flatten)   (None, 512)               0

(Dense)        (None, 10)                5130

Parameter count: 403562
