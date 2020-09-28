
import os

from hps.utils.Singleton import Singleton
from hps.utils.Configurations import Configurations

# class : Constants
class Constants(metaclass=Singleton):
    __FILE_REAL_PATH = os.path.dirname(os.path.realpath(__file__))

    ### early stop
    EARLY_TYPE_NONE = "none"
    EARLY_TYPE_MIN = "min"
    EARLY_TYPE_MAX = "max"
    EARLY_TYPE_VAR = "var"

    ### default config
    DEFAULT = Configurations(config_path=__FILE_REAL_PATH+"/../conf/default.conf")

    ### DIR SETTING
    DIR_DATA = __FILE_REAL_PATH + "/../.." + DEFAULT.get("DIR_CONFIG", "DIR_DATA")
    DIR_PARAMS = DIR_DATA + DEFAULT.get("DIR_CONFIG", "DIR_PARAMS")


    ### ML DEVICES
    DEVICE_MODE = DEFAULT.get("ML_DEVICE", "MODE")
    NUM_DEVICES = DEFAULT.get("ML_DEVICE", "VISIBLE_DEVICES")
    DEVICE_MEM  = DEFAULT.get("ML_DEVICE", "DEVICE_MEM")

if __name__ == '__main__':
    print(Constants.DIR_DATA)