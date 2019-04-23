import tensorflow
import math
import numpy
import os
import sys
import matplotlib.pyplot as pyplot

import lib.logger.logger as logger

import Loader
import ImageConverter

# tensorflow.compat.v1.disable_v2_behavior()

BATCH_SIZE = 24

def train_input_fn(features, labels, batch_size):
    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

def change_range(image, label):
    return 2*image-1, label

def my_model_fn(features, labels, mode, params):
    logger.log("cat", "Cat")

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], "None", img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])
  
    pyplot.imshow(img)

    predicted_label = "PREDIT"
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    pyplot.xlabel("{} {:2.0f}% ({})".format(true_label,
                                100*numpy.max(predictions_array),
                                predicted_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])
    thisplot = pyplot.bar(range(len(predictions_array)), predictions_array, color="#777777")
    pyplot.ylim([0, 1]) 
    predicted_label = numpy.argmax(predictions_array)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def get_activations(layer, stimuli):
    units = sess.run(layer, feed_dict={'x':numpy.reshape(stimuli,[1, 7, 84], order='F'), 'rate':0.0})
    plotNNFilter(units)

def create_dataset(data, labels, batch_size):
    def gen():
        for image, label in zip(data, labels):
            yield image, label
    ds = tensorflow.data.Dataset.from_generator(gen, (tensorflow.float32, tensorflow.int32), ((28, 28), ()))

    return ds.repeat().batch(batch_size)

################################################# 

logger.log("TENSORFLOW_VERSION", str(tensorflow.__version__))

trainingSet = ImageConverter.ImageConverter(os.path.dirname(__file__) + "\images_training", 28, 28)
trainingImages, trainingLabels = trainingSet.process()

testSet = ImageConverter.ImageConverter(os.path.dirname(__file__) + "\images_test", 28, 28)
testImages, testLabels = testSet.process()

# train_set = create_dataset(trainingImages, trainingLabels, 10)
# test_set = create_dataset(testImages, testLabels, 20)

# print(train_set)

model = tensorflow.keras.Sequential(
    [
        tensorflow.keras.layers.Flatten(input_shape=(28, 28)),  #Probably only possible when the image is single-channel
        tensorflow.keras.layers.Dense(64, activation=tensorflow.nn.softmax),
        tensorflow.keras.layers.Dense(24, activation=tensorflow.nn.softmax),
        #tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
        tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              #loss=tensorflow.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

checkpoint_path = "data/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) 

train = False
if(train == True):
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)

    model.fit(trainingImages, trainingLabels, epochs=5, validation_data = (testImages, testLabels))
    model.save_weights("data/weights")
else:
    model.load_weights("data/weights")

logger.log("MODEL SUMMARY", str(model.summary()))

prediction = model.predict(testImages, steps=32)

weights = model.get_weights()

FIG_COLS = 9

for i in [0, 1000, 2000, 3000, 4000, 6000, 8000]:
    pyplot.figure(figsize=(14,3))
    pyplot.subplot(1,FIG_COLS,1)
    print(prediction[i])
    plot_image(i, prediction, trainingLabels, testImages)

    activation = testImages[i]

    for layer in model.layers:
        print(layer)

    for j in range(0, len(model.layers)-1):
        logger.log(str(j), "GO")
        pyplot.subplot(1,FIG_COLS,2+j)
        layer = model.get_layer(index=j)
        try:
            activation = layer.apply(activation)
            pyplot.imshow(activation)
        except Exception as e:
            logger.log(str(j) + ":APPLYFAIL", str(e))
            pass

    pyplot.subplot(1,FIG_COLS,len(model.layers)+2)
    plot_value_array(i, prediction, testLabels)
    pyplot.show()

# model.fit(train_images, train_labels, epochs=args.epochs, validation_data = (test_images, test_labels), callbacks=[cp_callback])


#sess = tensorflow.Session()
#init = tensorflow.global_variables_initializer()
#sess.run(init)

#(train_x, train_y), (test_x, test_y)

#my_feature_columns = []
#for key in train_x.keys():
#    my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))
