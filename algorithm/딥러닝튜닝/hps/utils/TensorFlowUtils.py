

import os
import tensorflow as tf

# class : TensorFlowUtils
class TensorFlowUtils(object):
    @staticmethod
    def get_active_fn(act_fn):
        active = act_fn.lower()
        if active == "relu":
            return  tf.nn.relu
        elif active == "sigmoid":
            return tf.nn.sigmoid
        elif active == "tanh":
            return tf.nn.tanh
        else:
            raise NotImplementedError

    @staticmethod
    def get_optimizer_fn(optimizer_fn):
        optimizer = optimizer_fn.lower()
        if optimizer == "adam":
            return tf.keras.optimizers.Adam
        elif optimizer == "adadelta":
            return tf.keras.optimizers.Adadelta
        elif optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop
        else:
            raise NotImplementedError

    @staticmethod
    def get_pooling_fn(pooling_fn):
        pooling = pooling_fn.lower()
        if pooling == "max1d":
            return tf.keras.layers.MaxPool1D
        elif pooling == "average1d" :
            return tf.keras.layers.AveragePooling1D
        else :
            raise NotImplementedError

    @staticmethod
    def get_conv_fn(conv_fn):
        conv = conv_fn.lower()
        if conv == "conv1d":
            return tf.keras.layers.Conv1D
        else :
            raise NotImplementedError

    @staticmethod
    def get_rnn_cell(rnn_cell):
        cell = rnn_cell.lower()
        if cell == "rnn":
            return tf.keras.layers.SimpleRNN
        elif cell == "gru":
            return tf.keras.layers.GRU
        elif cell == "lstm":
            return tf.keras.layers.LSTM
        else:
            raise NotImplementedError

    @staticmethod
    def get_units(input_units, hidden_units, output_units):
        unit_list = list()
        unit_list.append(input_units)

        # for i in range(num_layer):
        #     unit_list.append(hidden_units)
        for unit in hidden_units:
            unit_list.append(unit)

        unit_list.append(output_units)
        return unit_list

    @staticmethod
    def mlp_block(
            model, units, act_fn, dropout_prob, initial_weight,
            name="mlp", algorithm_type="classifier"
    ):
        for i in range(len(units) - 2):
            layer_nm = "{}_{}".format(name, str(i + 1))

            initializer = tf.keras.initializers.RandomUniform(
                minval=-initial_weight, maxval=initial_weight, seed=None
            )
            model.add(tf.keras.layers.Dense(
                units[i + 1], activation=act_fn, name=layer_nm,
                kernel_initializer=initializer
            ))
            model.add(tf.keras.layers.Dropout(dropout_prob))

        final_act_fn = act_fn
        if algorithm_type == "classifier":
            final_act_fn = tf.nn.softmax
            if units[-1] == 1:
                final_act_fn = tf.nn.sigmoid

        model.add(tf.keras.layers.Dense(units[-1], activation=final_act_fn, name=name+"_predict", ))
        return model

    @staticmethod
    def compile_model(model, algorithm_type, output_units, optimizer_fn, learning_rate):
        if algorithm_type == "classifier":
            # loss_fn_nm = 'categorical_crossentropy'
            loss_fn_nm='sparse_categorical_crossentropy'
            if output_units == 1:
                loss_fn_nm = "binary_crossentropy"
            model.compile(
                loss=loss_fn_nm,
                optimizer=optimizer_fn(learning_rate),
                metrics=['accuracy']
            )

        elif algorithm_type == "regressor":
            model.compile(
                loss="mse",
                optimizer=optimizer_fn(learning_rate),
            )

        else:
            raise NotImplementedError

    @staticmethod
    def device_memory_limit(gpu_idx, mem_limit):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        if str(gpu_idx) != "-1":
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            print(physical_devices)
            tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
