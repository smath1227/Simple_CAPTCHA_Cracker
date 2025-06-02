# Import statements
import os
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path

directory = Path("samples")

# Sort all images with a .png extension --> retrieve only the "label" (file name) from path
dir_img = sorted(list(map(str, list(directory.glob("*.png")))))
img_labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in dir_img]

# Identify all unique characters that exist within labels
char_img = set(char for labels in img_labels for char in labels)
char_img = sorted(list(char_img))

print("Number of dir_img found: ", len(dir_img))
print("Number of img_labels found: ", len(img_labels))
print("Number of unique char_img: ", len(char_img))
print("Characters present: ", char_img)

# How many examples we will process in a single pass
batch_size = 16

# Image specifications
img_width = 200
img_height = 50

# Reduce feature maps by a factor of 4 compared to original
downsample_factor = 4

max_length = max([len(label) for label in img_labels])

# Map chars --> nums and vice versa
char_to_num = layers.StringLookup(vocabulary = list(char_img), mask_token = None)
num_to_char = layers.StringLookup(vocabulary = char_to_num.get_vocabulary(), mask_token = None, invert = True)

def data_split(dir_img, img_labels, train_size = 0.9, shuffle = True):

    # Determine total number of examples and create indices. Then, shuffle the indices
    size = (len(dir_img))
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)

    # Dedicate 90% of the total examples to training data
    train_samples = int(size * train_size)

    # Split the sample data into training and validation sets
    # (first 90% of shuffled indices = training, rest = validation)
    x_train, y_train = dir_img[indices[:train_samples]], img_labels[indices[:train_samples]]
    x_valid, y_valid = dir_img[indices[train_samples:]], img_labels[indices[train_samples:]]
    return x_train, y_train, x_valid, y_valid

# Create training sets
x_train, y_train, x_valid, y_valid = data_split(np.array(dir_img), np.array(img_labels))

def encode(img_path, label):
    # Load image from memory
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels = 1)

    # Grayscale the image
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize + transpose
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm = [1, 0, 2])

    # Break the label into a list of single characters
    label = char_to_num(tf.strings.unicode_split(label, "UTF-8"))

    return {"image" : img, "label" : label}

# Create full training and validation sets
train_data = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(encode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE))
valid_data = (tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).map(encode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE))

# Define custom loss function
class LayerCTC(layers.Layer):
	def __init__(self, name=None):
		super().__init__(name=name)
		self.loss_fn = keras.backend.ctc_batch_cost

	def call(self, y_true, y_pred):
		# Compute the training-time loss value
		batch_len = tf.cast(tf.shape(y_true)[0],
							dtype="int64")
		input_length = tf.cast(tf.shape(y_pred)[1],
							dtype="int64")
		label_length = tf.cast(tf.shape(y_true)[1],
							dtype="int64")

		input_length = input_length * \
			tf.ones(shape=(batch_len, 1), dtype="int64")
		label_length = label_length * \
			tf.ones(shape=(batch_len, 1), dtype="int64")

		loss = self.loss_fn(y_true, y_pred,
							input_length, label_length)
		self.add_loss(loss)

		# Return Computed predictions
		return y_pred


def build_model():
    # Define the inputs to the model
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    img_labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First convolutional block
    x = layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2,2), name="pool1")(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2,2), name="pool2")(x)

    # Reshape input prior to processing in RNN
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout")(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, name="lstm1"))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25, name="lstm2"))(x)

    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    output = LayerCTC() (img_labels, x)

    model = keras.models.Model(inputs=[input_img, img_labels], outputs=output, name="model")
    opt = keras.optimizers.Adam()

    model.compile(optimizer=opt)

    return model

model = build_model()
model.summary()

epochs = 100
early_stopping_patience = 10

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=early_stopping_patience,
    restore_best_weights=True
)

# Training the model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    callbacks=[early_stopping],
)