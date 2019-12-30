import tensorflow
import math
import numpy
import os
import sys
import matplotlib.pyplot as pyplot

import lib.logger.logger as logger

import Loader
import ImageConverter

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BATCH_SIZE = 256

################################################# 
OLD_LOSS = 100.0
OLD_ACC = 0.0
OLD_VLOSS = 100.0
OLD_VACC = 0.0

TRAIN = False
EPOCHS = 1000

class VisualizerCB(tensorflow.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        global OLD_LOSS, OLD_ACC, OLD_VLOSS, OLD_VACC

        l1 = logger.YELLOW if(OLD_LOSS == logs["loss"]) else logger.GREEN if(OLD_LOSS > logs["loss"]) else logger.RED
        l2 = logger.YELLOW if(OLD_ACC == logs["accuracy"]) else logger.GREEN if(OLD_ACC < logs["accuracy"]) else logger.RED
        l3 = logger.YELLOW if(OLD_VLOSS == logs["val_loss"]) else logger.GREEN if(OLD_VLOSS > logs["val_loss"]) else logger.RED
        l4 = logger.YELLOW if(OLD_VACC == logs["val_accuracy"]) else logger.GREEN if(OLD_VACC < logs["val_accuracy"]) else logger.RED

        # logger.log(str(epoch), str("Loss: " + l1 + str(logs["loss"]).ljust(25) + logger.RESET + " Accuracy: " + l2 + str(logs["accuracy"]).ljust(15) + logger.RESET + " Val_Loss: " + l3 + str(logs["val_loss"]).ljust(25) + logger.RESET + " Val_Accuracy: " + l4 + str(logs["val_accuracy"]).ljust(15) + logger.RESET))
        logger.log(logger.CYAN + str(epoch) + logger.RESET, str(", " + l1 + str(logs["loss"]) + logger.RESET + ", " + l2 + str(logs["accuracy"]) + logger.RESET + ", " + l3 + str(logs["val_loss"]) + logger.RESET + ", " + l4 + str(logs["val_accuracy"]) + logger.RESET))
        OLD_LOSS = logs["loss"]
        OLD_ACC = logs["accuracy"]
        OLD_VLOSS = logs["val_loss"]
        OLD_VACC = logs["val_accuracy"]

class MyModelV2(tensorflow.keras.Model):
    def __init__(self):
        super(MyModelV2, self).__init__()

        self.lyr = {
            "lyr": [
                tensorflow.keras.layers.Dense(4, activation=tensorflow.nn.relu)
                ]
            }

        self.parallelLayers = {
            "x": [
                tensorflow.keras.layers.Conv2D(4, 2, strides=1, activation=tensorflow.nn.log_softmax),
                #tensorflow.keras.layers.MaxPooling2D(pool_size=2, strides=2),
                tensorflow.keras.layers.Conv2DTranspose(4, 2, strides=1),
            ],
            "y": [
                tensorflow.keras.layers.Conv2D(4, 2, strides=1, activation=tensorflow.nn.sigmoid),
                #tensorflow.keras.layers.MaxPooling2D(pool_size=2, strides=2),
                tensorflow.keras.layers.Conv2DTranspose(4, 2, strides=1)
            ],
            "z": [
                tensorflow.keras.layers.Conv2D(4, 2, strides=1, activation=tensorflow.nn.selu),
                #tensorflow.keras.layers.MaxPooling2D(pool_size=2, strides=2),
                tensorflow.keras.layers.Conv2DTranspose(4, 2, strides=1)
            ]
        }

    def call(self, inputs):
        tensors = []
        idx = 0
        for t in self.parallelLayers:
            tensors.append(self.parallelLayers[t][0](inputs))
            for i in range(1, len(self.parallelLayers[t])):
                tensors[idx] = self.parallelLayers[t][i](tensors[idx])
            idx += 1

        combined = tensorflow.concat(tensors, 3)

        return self.lyr["lyr"][0](combined)

    def getResult(self, x, y, p, images):
        for t in self.parallelLayers:
            for i in self.parallelLayers[t]:
                try:
                    img = i.call(images) #self.get_layer(index=i).call(images[0:20])
                    pyplot.subplot(x,y,p)
                    pyplot.imshow(img[0])
                    logger.log("Model[" + str(t) + "]", str(logger.GREEN + "RENDER" + logger.RESET))
                except Exception as e:
                    logger.log("Model[" + str(t) + "]", str(logger.RED + str(e) + logger.RESET))
                p += 1
            p += 1

        for t in self.lyr:
            for i in self.lyr[t]:
                try:
                    img = i.call(images) #self.get_layer(index=i).call(images[0:20])
                    pyplot.subplot(x,y,p)
                    pyplot.imshow(img[0])
                    logger.log("Model[" + str(t) + "]", str(logger.GREEN + "RENDER" + logger.RESET))
                except Exception as e:
                    logger.log("Model[" + str(t) + "]", str(logger.RED + str(e) + logger.RESET))
                p += 1
            p += 1

logger.log(logger.WHITE + "TENSORFLOW_VERSION" + logger.RESET, str(tensorflow.__version__))

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
           tensorflow.config.experimental.set_memory_growth(gpu, True)
        tensorflow.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
        logger.log(logger.BLUE + "Physical GPU" + logger.RESET, str(len(gpus)))
        logger.log(logger.BLUE + "Logical GPU" + logger.RESET, str(len(logical_gpus)))
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        logger.log(logger.RED + "EXCEPTIONY" + logger.RESET, str(e))

pyplot.figure(figsize=(4,2))

if(TRAIN == True):
    trainingSet = ImageConverter.ImageConverter(os.path.dirname(__file__) + "\images_training", BATCH_SIZE, BATCH_SIZE)
    trainingImages, trainingLabels = trainingSet.process(True)
 
    pyplot.subplot(4,8,1)
    pyplot.imshow(trainingImages[0])
    pyplot.subplot(4,8,2)
    pyplot.imshow(trainingLabels[0])

    testSet = ImageConverter.ImageConverter(os.path.dirname(__file__) + "\images_test", BATCH_SIZE, BATCH_SIZE)
    testImages, testLabels = testSet.process(True)

# beef = tensorflow.concat([1, 2], 0)

model = MyModelV2()
#mB = DenseModel()
#model = MyModel(mA, mB)

checkpoint_path = "data/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) 

if(TRAIN == True):
    logger.log("Train", "Training Start!")
    model.load_weights('data/weights')
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.Huber(),
              metrics=["accuracy"])
    model.fit(trainingImages, trainingLabels, verbose=0, shuffle=True, batch_size=1024, epochs=EPOCHS, validation_data=(testImages, testLabels), workers=24, use_multiprocessing=True, callbacks=[VisualizerCB()])
    #model.fit(trainingImages, trainingLabels, verbose=0, shuffle=True, epochs=300, validation_data=(testImages, testLabels), workers=8, use_multiprocessing=True)
    model.save_weights("data/weights")
else:
    model.load_weights('data/weights')
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.Huber(),
              metrics=["accuracy"])
    model.build((1, 32, 32, 4))

# logger.log("MODEL SUMMARY", str(model.summary()))

predictSet = ImageConverter.ImageConverter(os.path.dirname(__file__) + "\images", BATCH_SIZE, BATCH_SIZE)
predictImages, junk = predictSet.process(True)

prediction = model.predict(predictImages, steps=42)

model.getResult(4, 8, 9, predictImages)

pIndex = 0

pyplot.subplot(4,8,22)
pyplot.imshow(predictImages[0])

pyplot.subplot(4,8,23)
pyplot.imshow(prediction[pIndex])
pyplot.show()