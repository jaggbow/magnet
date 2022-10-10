from .fno_1d import FNO1d
from .fno_2d import FNO2d
from .magnet_cnn import MAgNetCNN
from .magnet_cnn_2d import MAgNetCNN_2d
from .magnet_cnn_no_interaction import MAgNetCNN_no_interaction
from .magnet_gnn import MAgNetGNN
from .mpnn import MPNN
from .mpnn_2d import MPNN_2d

FACTORY = {
    'fno_1d': FNO1d,
    'fno_2d': FNO2d,
    'mpnn': MPNN,
    'mpnn_2d': MPNN_2d,
    'magnet_cnn_no_interaction': MAgNetCNN_no_interaction,
    'magnet_cnn': MAgNetCNN,
    'magnet_cnn_2d': MAgNetCNN_2d,
    'magnet_gnn': MAgNetGNN
    }