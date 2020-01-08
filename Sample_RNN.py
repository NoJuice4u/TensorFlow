import tensorflow
import datetime
import math
import numpy
import os
import sys
import matplotlib
import matplotlib.image
import matplotlib.pyplot as pyplot

import lib.logger.logger as logger

import SineWave
import Loader
import ImageConverter

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NEW = True
TRAIN = True
EPOCHS = 1000
BATCH_SIZE = 128

################################################# 
OLD_LOSS = 100.0
LOSS_DELTA = 1
OLD_ACC = 0.0
ACC_DELTA = 1
OLD_VLOSS = 100.0
OLD_VACC = 0.0

NPOS = 8

class VisualizerCB(tensorflow.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        global OLD_LOSS, OLD_ACC, OLD_VLOSS, OLD_VACC, LOSS_DELTA, ACC_DELTA
        global NPOS

        newLossDelta = OLD_LOSS - logs["loss"]
        newAccDelta = OLD_ACC - logs["accuracy"]

        if(LOSS_DELTA == 0):
            lossDeltaPercentage = 0
        else:
            lossDeltaPercentage = newLossDelta / LOSS_DELTA

        l1 = logger.YELLOW if(OLD_LOSS == logs["loss"]) else logger.GREEN if(OLD_LOSS > logs["loss"]) else logger.RED
        l2 = logger.YELLOW if(OLD_ACC == logs["accuracy"]) else logger.GREEN if(OLD_ACC < logs["accuracy"]) else logger.RED
        l3 = logger.YELLOW if(OLD_VLOSS == logs["val_loss"]) else logger.GREEN if(OLD_VLOSS > logs["val_loss"]) else logger.RED
        l4 = logger.YELLOW if(OLD_VACC == logs["val_accuracy"]) else logger.GREEN if(OLD_VACC < logs["val_accuracy"]) else logger.RED
        l5 = logger.YELLOW if(lossDeltaPercentage == 1.0) else logger.CYAN if((lossDeltaPercentage > 1.2) or (lossDeltaPercentage < 0.8)) else logger.GREEN

        # logger.log(str(epoch), str("Loss: " + l1 + str(logs["loss"]).ljust(25) + logger.RESET + " Accuracy: " + l2 + str(logs["accuracy"]).ljust(15) + logger.RESET + " Val_Loss: " + l3 + str(logs["val_loss"]).ljust(25) + logger.RESET + " Val_Accuracy: " + l4 + str(logs["val_accuracy"]).ljust(15) + logger.RESET))
        logger.log(logger.CYAN + str(epoch) + logger.RESET, str(", " + l1 + str(logs["loss"]) + logger.RESET + ", " + l2 + str(logs["accuracy"]) + logger.RESET + ", " + l3 + str(logs["val_loss"]) + logger.RESET + ", " + l4 + str(logs["val_accuracy"]) + logger.RESET) + ", " + l5 + str(lossDeltaPercentage) + logger.RESET)

        if(epoch % (EPOCHS/10) == 0):
            prediction = model.predict(predictImages, steps=42)
            pyplot.subplot(4,8,NPOS)
            pyplot.imshow(prediction[0])
            NPOS += 1

        LOSS_DELTA = newLossDelta
        ACC_DELTA = newAccDelta

        OLD_LOSS = logs["loss"]
        OLD_ACC = logs["accuracy"]
        OLD_VLOSS = logs["val_loss"]
        OLD_VACC = logs["val_accuracy"]

class MyModelV2(tensorflow.keras.Model):
    def __init__(self):
        super(MyModelV2, self).__init__()

        self.lyr = [
                tensorflow.keras.layers.Dense(32),
                tensorflow.keras.layers.Dense(28, activation=tensorflow.nn.leaky_relu),
                tensorflow.keras.layers.Dense(6)
                ]

    def call(self, inputs):
        logger.log(logger.CYAN + "Model Called" + logger.RESET, "")
        tensors = []
        idx = 0

        logger.log(logger.CYAN + "Model Called" + logger.RESET, "Layers Executed. Now doing final layer")

        result = self.lyr[0](inputs)
        for i in range(1, len(self.lyr)):
            result = self.lyr[i](result)

        return result

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
    trainingImages = SineWave.Sin_Data()
    trainingLabels = SineWave.Sin_Label()

    testImages = SineWave.Sin_Data()
    testLabels = SineWave.Sin_Label()

# beef = tensorflow.concat([1, 2], 0)

model = MyModelV2()
#mB = DenseModel()
#model = MyModel(mA, mB)

checkpoint_path = "data/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) 

predictImages = SineWave.Sin_Data()

if(TRAIN == True):
    logger.log("Train", "Training Start!")
    if(NEW is False): 
        model.load_weights('data/weights')
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.Huber(),
              metrics=["accuracy"])
    nowA = datetime.datetime.now()
    model.fit(trainingImages, trainingLabels, verbose=0, shuffle=True, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testImages, testLabels), workers=24, use_multiprocessing=True, callbacks=[VisualizerCB()])
    nowB = datetime.datetime.now()

    deltaTime = nowB - nowA
    logger.log(logger.CYAN + str(deltaTime) + logger.RESET, "Time elapsed for " + str(EPOCHS) + " epochs, averaging: " + str(deltaTime/EPOCHS) + " per epoch")
    #model.fit(trainingImages, trainingLabels, verbose=0, shuffle=True, epochs=300, validation_data=(testImages, testLabels), workers=8, use_multiprocessing=True)
    model.save_weights("data/weights")
else:
    model.load_weights('data/weights')
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.Huber(),
              metrics=["accuracy"])
    model.build((1, 1, 1, 28))

# logger.log("MODEL SUMMARY", str(model.summary()))

prediction = model.predict(predictImages, steps=42)
print(prediction)