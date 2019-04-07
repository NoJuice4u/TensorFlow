from __future__ import absolute_import, division, print_function

#Environment Imports
import os
import sys
import argparse
import tensorflow
import numpy
import matplotlib.pyplot as pyplot

#Project Imports
import lib.logger.logger as logger
#import Estimator

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def main(args):
    logger.log("TENSORFLOW_VERSION", str(tensorflow.__version__))
    fashion_mnist = tensorflow.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    logger.log('## IMAGE_BATCH ##', str(train_images))
    logger.log('## LABEL_BATCH ##', str(train_labels))

    checkpoint_path = "data/checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if(args.train == True):
        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)

        model = create_model()
        model.fit(train_images, train_labels, epochs=args.epochs, validation_data = (test_images, test_labels), callbacks=[cp_callback])
        model.save_weights(args.data)
    else:
        model = create_model()
        model.load_weights(args.data)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    logger.log("Accuracy", "{:5.2f}%".format(100*test_acc))

    flattenLayer = model.get_layer(index=1)
    logger.log('## LABEL_BATCH ##', str(flattenLayer.weights[0]))

    logger.log("TEST ACCURACY", str(test_acc))

    predictions = model.predict(test_images)

    if(args.savemodel is not None):
        save_model(model, args.savemodel)

    i = 48
    pyplot.figure(figsize=(6,3))
    pyplot.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images)
    pyplot.subplot(1,2,2)
    plot_value_array(i, predictions, test_labels)

    pyplot.show()

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])
  
    pyplot.imshow(img, cmap=pyplot.cm.binary)

    predicted_label = numpy.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    pyplot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*numpy.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])
    thisplot = pyplot.bar(range(10), predictions_array, color="#777777")
    pyplot.ylim([0, 1]) 
    predicted_label = numpy.argmax(predictions_array)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Returns a short sequential model
def create_model():
    model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
            tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu),
            tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax)
        ])
  
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(),
                loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
    return model

def save_model(model, file):
    model.save(file)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', type=bool, default=True,        help='Perform training')
    parser.add_argument('--data', type=str, default="data/weights",     help='Data file to Load/Save')
    parser.add_argument('--epochs', type=int, default=3,           help='Number of training epochs')
    parser.add_argument('--savemodel', type=str, default="model.h5",      help='Number of training epochs')

    args = parser.parse_args()

    try:
        main(args)
    except SystemExit as e:
        print(e) 