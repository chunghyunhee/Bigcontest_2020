from hps.dataset.MNISTDataset import MNISTDataset
from hps.dataset.EraDataset import EraDataset

# class : DatasetFactory
class DatasetFactory(object):
    @staticmethod
    def create(data_nm, dim=1):
        data_nm = data_nm.lower()
        if data_nm == "mnist":
            if dim == 1:
                return MNISTDataset.get_tf_dataset_1d()
        elif data_nm == "era_dataset":
            return EraDataset.get()


if __name__ == '__main__':
    name = "MNIST"
    ds_train, ds_test = DatasetFactory.create(name)
    print(ds_train, ds_test)