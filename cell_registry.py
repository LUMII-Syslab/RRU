# Importing all cells
from cells.RRUCell import RRUCell
from cells.GatedRRUCell_a import RRUCell as GRRUACell
from cells.GRUCell import GRUCell
from cells.BasicLSTMCell import BasicLSTMCell
from cells.MogrifierLSTMCell import MogrifierLSTMCell

cell_registry = {
    "RRU": {  # ReZero version
        "cell_fn": RRUCell,
        "model_name": "rru_model",
        "has_separate_output_size": True,
        "state_is_tuple": False
    },
    "GRRUA": {  # Gated version with separate output size
        "cell_fn": GRRUACell,
        "model_name": "grrua_model",
        "has_separate_output_size": True,
        "state_is_tuple": False
    },
    "GRU": {
        "cell_fn": GRUCell,
        "model_name": "gru_model",
        "has_separate_output_size": False,
        "state_is_tuple": False
    },
    "LSTM": {
        "cell_fn": BasicLSTMCell,
        "model_name": "lstm_model",
        "has_separate_output_size": False,
        "state_is_tuple": True
    },
    "MogrifierLSTM": {  # For this you have to have dm-sonnet, etc. installed
        "cell_fn": MogrifierLSTMCell,
        "model_name": "mogrifier_lstm_model",
        "has_separate_output_size": False,
        "state_is_tuple": True
    }
}


def get_cell_information(cell_name):
    if cell_name not in cell_registry.keys():
        raise ValueError(f"No such cell ('{cell_name}') has been implemented!")

    cell_information = cell_registry[cell_name]

    cell_fn = cell_information["cell_fn"]
    model_name = cell_information["model_name"]
    has_separate_output_size = cell_information["has_separate_output_size"]
    state_is_tuple = cell_information["state_is_tuple"]

    return cell_fn, model_name, has_separate_output_size, state_is_tuple
