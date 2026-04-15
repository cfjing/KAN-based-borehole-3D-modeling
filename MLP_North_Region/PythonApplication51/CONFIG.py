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

EPOCHS =512
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
PRED_BATCH_SIZE = 1024 * 4

TRAINSET_RATIO = 60 / 100.
VALSET_RATIO = 20 / 100.
TESTSET_RATIO = 20 / 100.

INITIAL_LR = 1e-4
INITIAL_SEED = 0

GRID_RESOLUTION_3D =300      
SECTION_RESOLUTION_2D = 200 
DEM_MAX_POINTS = 300000            
BOREHOLE_PROJECTION_THRESHOLD = 200 


SECTIONS = [
    {'name': 'Section Lines 1', 'start': (12159315.13, 4437421.1), 'end': (12169310.14, 4435350.15)},
    {'name': 'Section Lines 2', 'start': (12163051.97, 4432526.958), 'end': (12164719.88, 4438374.973)},
    {'name': 'Section Lines 3', 'start': (12158130.55, 4433721.19), 'end': (12169352.05, 4439114.941)}
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