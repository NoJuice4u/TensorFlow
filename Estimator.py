import tensorflow
import numpy
import os
import sys
import matplotlib.pyplot as pyplot

import lib.logger.logger as logger

import Loader

BATCH_SIZE = 24
tensorflow.enable_eager_execution()

def train_input_fn(features, labels, batch_size):
    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

def change_range(image, label):
    return 2*image-1, label

def my_model_fn(features, labels, mode, params):
    logger.log("cat", "Cat")

def plot_image(i, predictions_array, true_label, img, labels):
    predictions_array, true_label, img = predictions_array[i], labels[true_label[i]], img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])
  
    pyplot.imshow(img, cmap=pyplot.cm.binary)

    predicted_label = labels[numpy.argmax(predictions_array)]
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

#################################################

logger.log("TENSORFLOW_VERSION", str(tensorflow.__version__))
trainingLoader = Loader.Loader(192, 192, BATCH_SIZE)
training_image, training_label = trainingLoader.getDataSet(os.path.dirname(__file__) + "\images_training")

testLoader = Loader.Loader(192, 192, BATCH_SIZE)
test_image, test_label = testLoader.getDataSet(os.path.dirname(__file__) + "\images_test")

labelNames = testLoader.getLabelIndexArray()

model = tensorflow.keras.Sequential(
    [
        tensorflow.keras.layers.Flatten(input_shape=(192, 192, 3)),
        tensorflow.keras.layers.Dense(192, activation=tensorflow.nn.relu),
        #tensorflow.keras.layers.GlobalAveragePooling2D(),
        tensorflow.keras.layers.Dense(len(labelNames), activation=tensorflow.nn.softmax)
    ])

model.compile(optimizer=tensorflow.train.AdamOptimizer(), 
              loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

logger.log("MODEL SUMMARY", str(model.summary()))

steps_per_epoch = int(tensorflow.ceil(len(trainingLoader.getFiles())/trainingLoader.getBatchSize()).numpy())
logger.log("#STEPS PER EPOCH#", str(steps_per_epoch))
model.fit(trainingLoader.getDS(), epochs=5, steps_per_epoch=5, validation_data=testLoader.getDS(), validation_steps=10)

#prediction = model.predict(testLoader.getDS(), steps=32)
prediction = model.predict(test_image, steps=8)

for i in range(0, 20):
    pyplot.figure(figsize=(6,3))
    pyplot.subplot(1,2,1)
    print(prediction[i])
    plot_image(i, prediction, test_label, test_image, labelNames)
    pyplot.subplot(1,2,2)
    plot_value_array(i, prediction, test_label)
    pyplot.show()

# model.fit(train_images, train_labels, epochs=args.epochs, validation_data = (test_images, test_labels), callbacks=[cp_callback])


#sess = tensorflow.Session()
#init = tensorflow.global_variables_initializer()
#sess.run(init)

#(train_x, train_y), (test_x, test_y)

#my_feature_columns = []
#for key in train_x.keys():
#    my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))
