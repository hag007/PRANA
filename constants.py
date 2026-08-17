import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONF = "config/config.json"
config_json = json.load(open(os.path.join(dir_path, PATH_TO_CONF)))
BASE_DIR = config_json["BASE_PROFILE"]
PRS_DATASETS = config_json["PRS_DATASETS"]
PRS_PRSS = config_json["PRS_PRSS"]
PRS_OUTPUT = config_json["PRS_OUTPUT"]
