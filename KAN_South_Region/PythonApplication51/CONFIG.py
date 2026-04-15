# CONFIG.py

NUM_CLASSES = 7

POS_AXES = ['X', 'Y', 'Depth']
FAC_LSIT = ['DEM',
            #'Slope',
            #'Aspect',
            'WaterSys_Dis',
            'Faults_Dis',
            #'Eq_Dis',
            #'Plan_Cur',
            #'Prof_Cur',
            #'Comp_Cur',
            ]

NUM_FEATURE_POS = len(POS_AXES)
NUM_FEATURE_FAC = len(FAC_LSIT)

EPOCHS = 512
TRAIN_BATCH_SIZE = 32*8
TEST_BATCH_SIZE = 256*4
PRED_BATCH_SIZE = 1024 * 4

TRAINSET_RATIO = 60 / 100.
VALSET_RATIO = 20 / 100.
TESTSET_RATIO = 20 / 100.

INITIAL_LR = 1e-3
INITIAL_SEED = 0

GRID_RESOLUTION_3D =300      
SECTION_RESOLUTION_2D = 200    
DEM_MAX_POINTS = 300000            
BOREHOLE_PROJECTION_THRESHOLD = 200 


SECTIONS = [
    {'name': '剖面线 1', 'start': (12150355.98,4404513), 'end': (12175278.43,4379015.514)},
    {'name': '剖面线 2', 'start': (12140029.53,4408424.757), 'end': (12177391.25,4370169.905)},
    {'name': '剖面线 3', 'start': (12139045.49, 4402804.661), 'end': (12177940.99, 4362928.75)},
    {'name': '剖面线 4', 'start': (12155146.24,4389706.854), 'end': (12169480.28,4398177.587)},
    {'name': '剖面线 5', 'start': (12154809.21,4366906.173), 'end': (12175278.43,4379015.514)},
    {'name': '剖面线 6', 'start': (12156945.5,4357948.259), 'end': (12177391.25,4370169.905)}
]

COLOR_DIM_FACTOR = 0.9

VIS_COLOR_MAP = {
      0: '#ffd6a5',
    1: '#ffadad',
    2: '#bdb2ff',
    3: '#caffbf',
    4: '#9bf6ff',
    5: '#a0c4ff',
    6: '#fdffb6'
}