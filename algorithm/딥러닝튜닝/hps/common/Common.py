
import os

from hps.common.Constants import Constants
from hps.utils.Singleton import Singleton
from hps.utils.MPLogger import MPLogger
from hps.utils.CommonUtils import CommonUtils

# class : Common
class Common(metaclass=Singleton):
    __FILE_REAL_PATH = os.path.dirname(os.path.realpath(__file__))

    # make directories
    CommonUtils.mkdir(Constants.DIR_DATA)
    CommonUtils.mkdir(Constants.DIR_PARAMS)

    # make multi-process logger
    __DIR_LOG = __FILE_REAL_PATH + "/../../" + Constants.DEFAULT.get("LOG_CONFIG", "LOG_DIR")
    CommonUtils.mkdir(__DIR_LOG)
    LOGGER = MPLogger(
        log_dir=__DIR_LOG, log_name=Constants.DEFAULT.get("LOG_CONFIG", "LOG_NAME"),
        log_level=Constants.DEFAULT.get("LOG_CONFIG", "LOG_LEVEL")
    )
