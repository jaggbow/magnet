from .fno_1d import FNO1d
from .magnet_cnn import MAgNetCNN
from .magnet_cnn_no_int import MAgNetCNN_no_interaction
from .magnet_gnn import MAgNetGNN
from .mpnn import MPNN

FACTORY = {
    'fno_1d': FNO1d,
    'mpnn': MPNN,
    'magnet_cnn_no_int': MAgNetCNN_no_interaction,
    'magnet_cnn': MAgNetCNN,
    'magnet_gnn': MAgNetGNN
}