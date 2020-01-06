import argparse
import os
import sys
import matplotlib
import matplotlib.image
import matplotlib.pyplot as pyplot

import tensorflow
import numpy

import lib.logger.logger as logger

class ImageConverter:
    def __init__(self, path, x, y, channels=3):
        self.x = x
        self.y = y
        self.path = path
        self.dataset = None

    def process(self, build=False):
        if(build == False):
            pos = self.path.rfind('\\') + 1
            labelGroup = self.path[pos:]
            return self._loadDataSet('data\\' + labelGroup + '_images', 'data\\' + labelGroup + '_labels')
        else:
            return self._buildDataSet()

    def getDataSet(self):
        return self.dataset
     
    def setDataSet(self, images_tensor, labels_tensor):
        images_ds = tensorflow.data.Dataset.from_tensor_slices(images_tensor)
        labels_ds = tensorflow.data.Dataset.from_tensor_slices(tensorflow.cast(labels_tensor, tensorflow.int32))

        self.dataset = tensorflow.data.Dataset.zip((images_ds, labels_ds)).shuffle(buffer_size=256)

    def _buildDataSet(self):
        labelset = {}
        files = {}
        labelArray = []
        resultNormalArray = []

        imgset = []
        for r, d, f in os.walk(self.path):
            logger.log(logger.MAGENTA + "READING FILES" + logger.RESET, str(r))
            for file in f:
                if '.png' in file:
                    logger.log(logger.GREEN + "FILE" + logger.RESET, str(file))
                    path = os.path.join(r, file)
                    if(file[0:3] == "SRC"):
                        img = matplotlib.image.imread(path)
                        imgset.append(img)
                    elif(file[0:3] == "TAR"):
                        labelArray.append(matplotlib.image.imread(path))
                    else:
                        pass

        imgsetNPARR = numpy.asarray(imgset)
        labelsetNPARR = numpy.asarray(labelArray)

        pos = self.path.rfind('\\') + 1
        labelGroup = self.path[pos:]

        numpy.save('data\\' + labelGroup + '_images', imgsetNPARR)
        numpy.save('data\\' + labelGroup + '_labels', labelsetNPARR)

        # Map labelset with resultNormalArray
        images_tensor = tensorflow.convert_to_tensor(imgsetNPARR)
        labels_tensor = tensorflow.convert_to_tensor(labelsetNPARR)  ## USE INTEGER

        self.setDataSet(images_tensor, labels_tensor)

        label_names = sorted(list(labelset))
        label_to_index = dict((index, name) for name, index in enumerate(label_names))
        
        return images_tensor, labels_tensor

    def _loadDataSet(self, images, labels):
        imgsetNPARR = numpy.load(images + ".npy")
        labelsetNPARR = numpy.load(labels + ".npy")

        # Map labelset with resultNormalArray
        images_tensor = tensorflow.convert_to_tensor(imgsetNPARR)
        labels_tensor = tensorflow.convert_to_tensor(labelsetNPARR)  ## USE INTEGER

        self.setDataSet(images_tensor, labels_tensor)

        #label_names = sorted(list(labelset))
        #label_to_index = dict((index, name) for name, index in enumerate(label_names))

        return images_tensor, labels_tensor

    def _process_image(self, filename, label):
        image_string = tensorflow.read_file(filename)
        image_decoded = tensorflow.image.decode_jpeg(image_string, channels=3)
        logger.log("TRank", str(tensorflow.rank(image_decoded)))
        image_resized = tensorflow.image.resize_images(image_decoded, [self.x, self.y])  # Parameterize the size
        result = tensorflow.cast(image_resized, tensorflow.float64)/255.0
        
        return result, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, default=os.path.dirname(__file__) + "\images_training",     help='Data file to Load/Save')

    args = parser.parse_args()

    try:
        imageConverter = ImageConverter(args.path, 256, 256)
        imageConverter.process()
    except SystemExit as e:
        print(e) 