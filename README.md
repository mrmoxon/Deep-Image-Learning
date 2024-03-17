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

<br>
<br>
<br>

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

<br>
<br>
<br>

3. Alternative Method:

For scores [99.79, 98.79]:

500,000 parameters

Dataset:

Using the provided MNIST dataset.

### Data Augmentation:

- rotation_range=10,          # Random rotation between 0 and 10 degrees

- width_shift_range=0.1,      # Randomly shift images horizontally (fraction of total width)

- height_shift_range=0.1,     # Randomly shift images vertically (fraction of total height)

- zoom_range=0.1,   # Randomly zoom in/out on images

### Optimizer, Loss function:

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )

loss = 'categorical_crossentropy'

### Callbacks:

1.    Custom learning Function:

def scheduler(epoch, lr):

    if epoch < 10:

        return lr

    else:

        return lr * tf.math.exp(-0.1)

2.    Model Checkpoint:

Save the best model based on value accuracy.

Batch Size = 256 (To better utilise GPU)

Epochs = 50

Per Epoch time = 20 seconds

Total training time = ~16 minutes

<br>
<br>
<br>

4. Alternative Method

For scores [99.63, 98.79]:

Dataset:

Using the provided MNIST dataset.

Data Augmentation:

rotation_range=8 

width_shift_range=0.08

height_shift_range=0.08 

shear_range=0.3

zoom_range=0.8

Optimizer, Loss function:

optimizer = Adam

loss = categorical_crossentropy

Callbacks:

1. Learning Rate Schedulers

ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001) 

3. Early Stopping

EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

Training:

batch_size = 170

steps_per_epoch=x_train.shape[0]//batch_size

epochs=30

Model:
___________________________________________________________________________________
Layer (type)                                        Output Shape               Param # 
=================================================================
conv2d (Conv2D)                               (None, 28, 28, 32)            832 

conv2d_1 (Conv2D)                           (None, 28, 28, 32)        25632 

max_pooling2d (MaxPooling2D)     (None, 14, 14, 32)                0 

dropout (Dropout)                             (None, 14, 14, 32)                0 

conv2d_2 (Conv2D)                            (None, 14, 14, 64)       18496 

conv2d_3 (Conv2D)                            (None, 14, 14, 64)       36928 

max_pooling2d_1(MaxPooling 2D) (None, 7, 7, 64)                    0 

dropout_1 (Dropout)                         (None, 7, 7, 64)                   0 

flatten (Flatten)                                   (None, 3136)                       0 

...

Non-trainable params: 0


Trained using = Nvidia P100

[image](https://github.com/mrmoxon/Deep-Image-Learning/assets/110777587/c25b1313-5018-4b78-aca7-f54dabe6a32d)

