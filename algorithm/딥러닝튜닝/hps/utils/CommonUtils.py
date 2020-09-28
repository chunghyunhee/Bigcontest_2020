
import os

# class : CommonUtils
class CommonUtils(object):
    @staticmethod
    def mkdir(dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
