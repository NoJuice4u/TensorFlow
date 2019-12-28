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

BATCH_SIZE = 24

################################################# 
OLD_LOSS = 100.0

class VisualizerCB(tensorflow.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        global OLD_LOSS
        if(OLD_LOSS > logs["loss"]):
            logger.log(str(epoch), str(logger.GREEN + str(logs) + logger.RESET))
        else:
            logger.log(str(epoch), str(logger.RED + str(logs) + logger.RESET))
        OLD_LOSS = logs["loss"]

class PixelMapModel(tensorflow.keras.Model):
    def __init__(self):
        super(PixelMapModel, self).__init__()

        self.lyr = [
            tensorflow.keras.layers.Conv2D(4, (2, 2), strides=(1, 1), activation=tensorflow.nn.relu),
            tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            tensorflow.keras.layers.Conv2DTranspose(4, (4, 4), strides=(2, 2))
            ]
        
        self.lyO = [None, None, None]

    def call(self, inputs):
        self.lyO[0] = self.lyr[0](inputs)
        self.lyO[1] = self.lyr[1](self.lyO[0])
        self.lyO[2] = self.lyr[2](self.lyO[1])
        return self.lyO[2]

    def getResult(self, x, y, p, images):
        for i in self.lyr:
            try:
                img = i.call(images[0:20]) #self.get_layer(index=i).call(images[0:20])
                pyplot.subplot(x,y,p)
                pyplot.imshow(img[10])
                p += 1
                logger.log("PixelMapModel", str(logger.GREEN + "RENDER" + logger.RESET))
            except Exception as e:
                logger.log("PixelMapModel", str(logger.RED + str(e) + logger.RESET))

class DenseModel(tensorflow.keras.Model):
    def __init__(self):
        super(DenseModel, self).__init__()

        self.lyr = [
            tensorflow.keras.layers.Dense(4, activation=tensorflow.nn.softmax)
            ]

    def call(self, inputs):
        return self.lyr[0](inputs)

    def getResult(self, x, y, p, images):
        for i in self.lyr:
            try:
                img = i.call(images[0:20]) #self.get_layer(index=i).call(images[0:20])
                pyplot.subplot(x,y,p)
                pyplot.imshow(img[10])
                p += 1
                logger.log("DenseModel", str(logger.GREEN + "RENDER" + logger.RESET))
            except Exception as e:
                logger.log("DenseModel", str(logger.RED + str(e) + logger.RESET))

class MyModel(tensorflow.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__([modelA, modelB])

        self.lyr = [
            tensorflow.keras.layers.Dense(4, activation=tensorflow.nn.softmax)
            ]

    def call(self, inputs):
        return self.lyr[0](inputs)

    def getResult(self, x, y, p, images):
        for i in self.lyr:
            try:
                img = i.call(images[0:20]) #self.get_layer(index=i).call(images[0:20])
                pyplot.subplot(x,y,p)
                pyplot.imshow(img[10])
                p += 1
                logger.log("MyModel", str(logger.GREEN + "RENDER" + logger.RESET))
            except Exception as e:
                logger.log("MyModel", str(logger.RED + str(e) + logger.RESET))

class MyModelV2(tensorflow.keras.Model):
    def __init__(self):
        super(MyModelV2, self).__init__()

    def call(self, inputs):
        x = tensorflow.keras.layers.Conv2D(4, (2, 2), strides=(1, 1), activation=tensorflow.nn.relu)(inputs)
        x = tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tensorflow.keras.layers.Conv2DTranspose(4, (4, 4), strides=(2, 2))(x)
        x = tensorflow.keras.Model(inputs=inputs, outputs=x)

        y = tensorflow.keras.layers.Dense(4, activation=tensorflow.nn.softmax)(inputs)
        y = tensorflow.keras.Model(inputs=inputs, outputs=y)

        combined = tensorflow.concat([x, y], 0)

        return self.lyr[0](inputs)

    def getResult(self, x, y, p, images):
        for i in self.lyr:
            try:
                img = i.call(images[0:20]) #self.get_layer(index=i).call(images[0:20])
                pyplot.subplot(x,y,p)
                pyplot.imshow(img[10])
                p += 1
                logger.log("MyModel", str(logger.GREEN + "RENDER" + logger.RESET))
            except Exception as e:
                logger.log("MyModel", str(logger.RED + str(e) + logger.RESET))


logger.log("TENSORFLOW_VERSION", str(tensorflow.__version__))

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
           tensorflow.config.experimental.set_memory_growth(gpu, True)
        tensorflow.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        logger.log("Physical GPU", str(len(gpus)))
        logger.log("Logical GPU", str(len(logical_gpus)))
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        logger.log("EXCEPTIONY", str(e))

trainingSet = ImageConverter.ImageConverter(os.path.dirname(__file__) + "\images_training", 32, 32)
trainingImages, trainingLabels = trainingSet.process(True)

pyplot.figure(figsize=(4,2))
pyplot.subplot(3,8,1)
pyplot.imshow(trainingImages[0])
pyplot.subplot(3,8,2)
pyplot.imshow(trainingLabels[0])

testSet = ImageConverter.ImageConverter(os.path.dirname(__file__) + "\images_test", 32, 32)
testImages, testLabels = testSet.process(True)

# beef = tensorflow.concat([1, 2], 0)

model = PixelMapModel()
#mB = DenseModel()
#model = MyModel(mA, mB)

checkpoint_path = "data/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) 

train = True
if(train == True):
    logger.log("Train", "TN")
    #model.load_weights('data/weights')
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.Huber(),
              metrics=["accuracy"])
    model.fit(trainingImages, trainingLabels, verbose=0, shuffle=True, epochs=5000, validation_data=(testImages, testLabels), workers=8, use_multiprocessing=True, callbacks=[VisualizerCB()])
    model.save_weights("data/weights")
else:
    model.load_weights('data/weights')
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.Huber(),
              metrics=["accuracy"])
    model.build((1, 32, 32, 4))

logger.log("MODEL SUMMARY", str(model.summary()))

prediction = model.predict(testImages[0:20], steps=42)

#mA.getResult(3, 8, 8, testImages[0:20])
#mB.getResult(3, 8, 16, testImages[0:20])
model.getResult(3, 8, 16, testImages[0:20])
#imm_4 = model.get_layer(index=3).call(testImages[0:20])
#pyplot.subplot(2,8,6)
#pyplot.imshow(imm_4[10])

pIndex = 0

pyplot.subplot(3,8,6)
pyplot.imshow(testImages[0])

pyplot.subplot(3,8,7)
pyplot.imshow(prediction[pIndex])
pyplot.show()

FIG_COLS = 9