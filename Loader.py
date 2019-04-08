import os
import tensorflow

import lib.logger.logger as logger

'''
COMMENTS
'''

class Loader:
    def __init__(self, xRes, yRes, batchSize):
        self.xRes = xRes
        self.yRes = yRes
        self.batchSize = batchSize
        self.files = []

    def getDataSet(self, path):
        labelset = set()
        for r, d, f in os.walk(path):
            logger.log("Reading JPG Files", str(r))
            for file in f:
                if '.jpg' in file:
                    path = os.path.join(r, file)
                    pos = os.path.dirname(path).rfind('\\') + 1
                    labelset.add(os.path.dirname(path)[pos:])
                    self.files.append(os.path.join(r, file))

        label_names = list(labelset)
        label_to_index = dict((name, index) for index,name in enumerate(label_names))
        all_image_labels = []
        for path in self.files:
            pos = os.path.dirname(path).rfind('\\') + 1
            all_image_labels.append(label_to_index[os.path.dirname(path)[pos:]])

        path_ds = tensorflow.data.Dataset.from_tensor_slices(self.files)
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)

        label_ds = tensorflow.data.Dataset.from_tensor_slices(tensorflow.cast(all_image_labels, tensorflow.int32))
        image_label_ds = tensorflow.data.Dataset.zip((image_ds, label_ds))

        logger.log('Image Shape', str(image_label_ds.output_shapes[0]))

        image_count = len(path)
        ds = image_label_ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.batch(self.batchSize)
        ds = ds.prefetch(buffer_size=tensorflow.data.experimental.AUTOTUNE)

        ds = image_label_ds.apply(tensorflow.data.experimental.shuffle_and_repeat(buffer_size=image_count))
        ds = ds.batch(self.batchSize)
        self.ds = ds.prefetch(buffer_size=tensorflow.data.experimental.AUTOTUNE)

        keras_ds = self.ds.map(self.change_range)
        return next(iter(keras_ds))

    def load_and_preprocess_image(self, path):
        image = tensorflow.read_file(path)
        return self.preprocess_image(image)

    def preprocess_image(self, image):
        image_decoded = tensorflow.image.decode_jpeg(image, channels=3)
        image_final = tensorflow.image.resize_images(image_decoded, [self.xRes, self.yRes])
        image_final = image_final/255.0
        return image_final

    def change_range(self, image, label):
        return 2*image-1, label

    def getBatchSize(self):
        return self.batchSize

    def getFiles(self):
        return self.files

    def getDS(self):
        return self.ds