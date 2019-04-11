import tensorflow
import numpy
import argparse
import os
import sys
import matplotlib.pyplot as pyplot

import lib.logger.logger as logger

tensorflow.enable_eager_execution()
class ImageConverter:
    def __init__(self, path, x, y):
        self.x = x
        self.y = y
        self.path = path

    def process(self):
        dataset = self.getDataSet()

        pyplot.figure(figsize=(8, 8))
        for image, label in dataset.take(4):
            pyplot.imshow(image)
            pyplot.grid(False)
            pyplot.xticks([])
            pyplot.yticks([])
            pyplot.show()

    def getDataSet(self):
        labelset = set()
        files = {}
        fileArray = []
        labelArray = []
        all_image_labels = []

        for r, d, f in os.walk(self.path):
            logger.log("Reading JPG Files", str(r))
            for file in f:
                if '.jpg' in file:
                    path = os.path.join(r, file)
                    pos = os.path.dirname(path).rfind('\\') + 1
                    labelGroup = os.path.dirname(path)[pos:]
                    if(labelGroup not in labelset):
                        labelset.add(labelGroup)
                        labelArray.append(labelGroup)
                    all_image_labels.append(labelGroup)

                    fileArray.append(os.path.join(r, file))

        label_names = sorted(list(labelset))
        label_to_index = dict((index, name) for name, index in enumerate(label_names))
        
        filenames = tensorflow.constant(fileArray)
        labels = tensorflow.constant(all_image_labels)

        dataset = tensorflow.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._process_image)

        return dataset

    def _process_image(self, filename, label):
        image_string = tensorflow.read_file(filename)
        image_decoded = tensorflow.image.decode_jpeg(image_string, channels=3)
        image_resized = tensorflow.image.resize_images(image_decoded, [self.x, self.y])  # Parameterize the size
        result = tensorflow.cast(image_resized, tensorflow.float64)/255.0
        
        return result, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, default=os.path.dirname(__file__) + "\images_training",     help='Data file to Load/Save')

    args = parser.parse_args()

    try:
        imageConverter = ImageConverter(args.path, 28, 28)
        imageConverter.process()
    except SystemExit as e:
        print(e) 