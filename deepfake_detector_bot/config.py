import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

SAMPLE_RATE = 22050
N_MFCC = 40

TEST_SIZE = 0.2
RANDOM_STATE = 42

BOT_TOKEN = '#:#'

THRESHOLDS = {
    'REAL_MAX': 0.75,      
    'UNCERTAIN_MIN': 0.75, 
    'UNCERTAIN_MAX': 0.85, 
    'FAKE_MIN': 0.85,      

}
