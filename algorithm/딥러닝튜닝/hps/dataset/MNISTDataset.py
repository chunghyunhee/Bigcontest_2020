
import tensorflow as tf
import tensorflow_datasets as tfds

# class : MNISTDataset
class MNISTDataset(object):
    @staticmethod
    def get():
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        return (ds_train, ds_test), ds_info

    @staticmethod
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    @staticmethod
    def normalize_img_1d(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        # return tf.cast(tf.reshape(image, [-1]), tf.float32) / 255., label
        return tf.cast(tf.reshape(image, [-1]), tf.float32), label

    @classmethod
    def make_dataset(cls, normalize_fn):
        (ds_train, ds_test), ds_info = cls.get()

        # train dataset
        ds_train = ds_train.map(
            normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        # test dataset
        ds_test = ds_test.map(
            normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_test

    @classmethod
    def get_tf_dataset_2d(cls):
        return cls.make_dataset(cls.normalize_img)


    @classmethod
    def get_tf_dataset_1d(cls):
        return cls.make_dataset(cls.normalize_img_1d)

if __name__ == '__main__':
    print(MNISTDataset.get_tf_dataset_1d())
