import os
from config.root import ROOT_DIR

TEST_SIZE = 0.2
RANDOM_STATE = 2000

polarity_map = {
    "NEG": 0,
    "NEU": 1,
    "POS": 2
}
emotion_map = {
    "others":   0,
    "joy":      1,
    "fear":     2,
    "anger":    3,
    "sadness":  4,
    "disgust":  5,
    "surprise": 6
}

REPORT_SVM_PATH = os.path.join(ROOT_DIR, 'reports', 'svm_report.csv')
REPORT_NB_PATH = os.path.join(ROOT_DIR, 'reports', 'nb_report.csv')
REPORT_LR_PATH = os.path.join(ROOT_DIR, 'reports', 'lr_report.csv')