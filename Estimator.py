import tensorflow
import numpy
import os
import sys

from logger import logger

import Loader

tensorflow.enable_eager_execution()

def train_input_fn(features, labels, batch_size):
    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

def my_model_fn(features, labels, mode, params):
    logger.log("cat", "Cat")

#################################################

logger.log("TENSORFLOW_VERSION", str(tensorflow.__version__))
trainingLoader = Loader.Loader(192, 192, 32)
training_image, training_label = trainingLoader.getDataSet(os.path.dirname(__file__) + "\images_training")

testLoader = Loader.Loader(192, 192, 32)
test_image, test_label = testLoader.getDataSet(os.path.dirname(__file__) + "\images_test")

logger.log("IMAGEBATCH", str(training_image))

mobile_net = tensorflow.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

feature_map_batch = mobile_net(training_image)
logger.log('## SHAPE ##', str(feature_map_batch.shape))

model = tensorflow.keras.Sequential(
    [
        mobile_net,
        tensorflow.keras.layers.GlobalAveragePooling2D(),
        tensorflow.keras.layers.Dense(len(training_label))
    ])

logit_batch = model(training_image).numpy()

logger.log("min logit:", str(logit_batch.min()))
logger.log("max logit:", str(logit_batch.max()))
logger.log("Shape:", str(logit_batch.shape))

model.compile(optimizer=tensorflow.train.AdamOptimizer(), 
              loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

logger.log("MODEL SUMMARY", str(model.summary()))

steps_per_epoch = tensorflow.ceil(len(trainingLoader.getFiles())/trainingLoader.getBatchSize()).numpy()
logger.log("#STEPS PER EPOCH#", str(steps_per_epoch))
model.fit(trainingLoader.getDS(), epochs=3, steps_per_epoch=3, validation_data=(test_image, test_label))

# model.fit(train_images, train_labels, epochs=args.epochs, validation_data = (test_images, test_labels), callbacks=[cp_callback])


#sess = tensorflow.Session()
#init = tensorflow.global_variables_initializer()
#sess.run(init)

#(train_x, train_y), (test_x, test_y)

#my_feature_columns = []
#for key in train_x.keys():
#    my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))
