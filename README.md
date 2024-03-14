# Deep_Image_Learning

For scores [99.72, 98.93]:

### Data preprocessing:

Only using original MNIST dataset,

Normalise, augment, optimise, call-backs. Train for 100e.

Train on augmented examples:

- rotation_range=10,

- width_shift_range=0.1,

- height_shift_range=0.1,

- shear_range=0.2,

- zoom_range=0.1,

- fill_mode='nearest',

- preprocessing_function=lambda x: x + tf.random.normal(tf.shape(x), stddev=0.05)  # Add Gaussian noise

Optimizer, Loss function:

- Adam with lr = 0.001

- CategoricalCrossentropy with label smoothing = 0.1


### Callbacks:

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

2. Alternative Method:

For score: [99.65, 98.81]:
2.2m parameters

Model architecture:
- convolutional layers:
     * 2 sets of Conv2D layers + Batch Normalization and ReLU activation.
     * MaxPooling2D layers after each set
     * Dropout layers, rate = 0.25.
- fully Connected Layers:
    * Flatten layer
    * 2 Dense layers with ReLU activation + Batch Normalization and dropout
    * Output layer + softmax activation

Data augmentation:
- featurewise_center=False, 
- samplewise_center=False, 
- featurewise_std_normalization=False, 
- samplewise_std_normalization=False, 
- zca_whitening=False, 
- rotation_range=10, 
- zoom_range = 0.1, 
- width_shift_range=0.1, 
- height_shift_range=0.1, 
- horizontal_flip=False,
- vertical_flip=False

Optimizer: Adam, lr = 0.001
- Loss function: categorical crossentropy

Callbacks: learning rate scheduler to adjust the lr (decrease by a factor of 0.9 after each epoch)

batch size = 64
epochs = 50


![image](https://github.com/mrmoxon/Deep-Image-Learning/assets/110777587/aa950ce4-9f4b-4dff-82cb-f65bd8c1869e)


