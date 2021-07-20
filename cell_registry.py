# In this file you can register your RNN cells and then you will be able to get the cell configuration from the cell you
# will choose in other files

# Importing all the cells that we need
from cells.BasicLSTMCell import BasicLSTMCell
from cells.GRUCell import GRUCell
from cells.MogrifierLSTMCell import MogrifierLSTMCell
from cells.RRUCell import RRUCell

# This dictionary holds the configuration for each cell we have implemented (each variable is explained in the function
# below), here you can also implement your cell by adding it's configuration (cell function / class, model name, whether
# it has a separate output size variable and whether the cell's state is tuple (as it is for LSTM based cells)) to the
# registry below, then importing the necessary function / class above.
cell_registry = {
    "RRU": {  # RRU ReZero version
        "cell_fn": RRUCell,
        "model_name": "rru_model",
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
    # A competitor's cell that we test our cells against, for this you have to have dm-sonnet, etc. installed
    "MogrifierLSTM": {
        "cell_fn": MogrifierLSTMCell,
        "model_name": "mogrifier_lstm_model",
        "has_separate_output_size": False,
        "state_is_tuple": True
    }
}


def get_cell_information(cell_name):
    """
        This function returns the requested cell's configuration.

        Input:
            cell_name: string, RNN cell's name.

        Output:
            cell_fn: class, a function / class that implements the corresponding cell;
            model_name: string, name that will be used for the cell when running the tasks and logging the results;
            has_separate_output_size: bool, does the cell have a separate output_size (we need this, because we need to
                know whether or not to send output_size to the implemented cell, because, if it has no such argument, it
                will raise an error);
            state_is_tuple: bool, is the cell's state a tuple (as is the case for the LSTM based cells).
    """

    # If the passed cell name isn't in the cell registry, we raise an error
    if cell_name not in cell_registry.keys():
        raise ValueError(f"No such cell ('{cell_name}') has been implemented!")

    # Access the information of the cell from the registry
    cell_information = cell_registry[cell_name]

    # Setting the information to it's corresponding variables
    cell_fn = cell_information["cell_fn"]
    model_name = cell_information["model_name"]
    has_separate_output_size = cell_information["has_separate_output_size"]
    state_is_tuple = cell_information["state_is_tuple"]

    return cell_fn, model_name, has_separate_output_size, state_is_tuple
