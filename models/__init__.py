from .bilstm import BiLSTM
from .cnn import ResNet1D
from .mlp import MLP
from .transformer import ECGTransformer

SIGNAL_MODELS = {"cnn": ResNet1D, "bilstm": BiLSTM, "transformer": ECGTransformer}
FEATURE_MODELS = {"mlp": MLP}


def build_model(name, **kw):
    if name in SIGNAL_MODELS:
        return SIGNAL_MODELS[name](**kw)
    return FEATURE_MODELS[name](**kw)


def input_kind(name):
    return "signal" if name in SIGNAL_MODELS else "features"
