
import configparser

# class : Configurations
class Configurations(object):
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get(self, section, key):
        return self.config.get(section, key)

if __name__ == '__main__':
    default_config = Configurations("../conf/default.conf")
    print(default_config.get("LOG_CONFIG", "LOG_LEVEL"))
