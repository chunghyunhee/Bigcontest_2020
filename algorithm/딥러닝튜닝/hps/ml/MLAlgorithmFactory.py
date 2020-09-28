
### keras neural network
from hps.ml.neural.DNN import DNN
from hps.ml.neural.CNN import CNN
from hps.ml.neural.RNN import RNN

# class : MLAlgorithmFactory
class MLAlgorithmFactory(object):
    @staticmethod
    def create(algorithm_name, param_dict):
        if algorithm_name == "DNN":
            return DNN(param_dict)
        elif algorithm_name == "CNN":
            return CNN(param_dict)
        elif algorithm_name == "RNN":
            return RNN(param_dict)

        raise NotImplementedError
