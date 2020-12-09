# This is a small program in which we explain what HyperOpt library functions our code can handle

# Importing the necessary stuff for hyperparameter optimization
from hyperopt import hp, tpe, Trials, fmin
from utils import print_trials_information
import numpy as np  # We need this for hp.loguniform

if __name__ == '__main__':  # Main function
    times_to_evaluate = 100

    # What to do with hp.choice variables
    activation_choice = ["relu", "gelu"]
    choices = {  # We need this, so we can print the hp.choice answers normally
        'activation': activation_choice
    }
    # What to do with hp.uniforms that need to be rounded
    round_uniform = ['number_of_parameters', 'output_size']
    # What to do with reverse hp.loguniform variables
    reverse_log_uniforms = {
        'residual_weight_initial_value': [0.0001, 1]  # It can't be 0
    }

    space = [
        # hp.choice
        hp.choice('activation', activation_choice),
        # hp.uniform
        hp.uniform('number_of_parameters', 1000000, 10000000),
        hp.uniform('output_size', 32, 256),
        hp.uniform('middle_layer_size_multiplier', 0.5, 8),
        hp.uniform('gate_bias', -1, 3),
        # hp.loguniform
        hp.loguniform('learning_rate', np.log(0.0004), np.log(0.004)),
        # Reverse hp.loguniform
        hp.loguniform('residual_weight_initial_value', np.log(0.0001), np.log(1))  # It can't be 0
    ]


    def objective(activation, number_of_parameters, output_size, middle_layer_size_multiplier, gate_bias, learning_rate,
                  residual_weight_initial_value):
        # Formatting special cases
        # hp.uniform - nothing to do, but if you want integers use round()
        number_of_parameters = round(number_of_parameters)
        output_size = round(output_size)
        # Reverse hp.loguniform - swap difference
        residual_weight_initial_value = \
            reverse_log_uniforms['residual_weight_initial_value'][1] - \
            (residual_weight_initial_value - reverse_log_uniforms['residual_weight_initial_value'][0])

        plus = 2 if activation == "relu" else 1

        minus = gate_bias + learning_rate + residual_weight_initial_value

        # Returning a random function to minimize
        return (number_of_parameters // output_size) + middle_layer_size_multiplier + plus - minus

    # https://github.com/hyperopt/hyperopt/issues/129
    def objective2(args):
        return objective(*args)

    # Create the algorithm
    tpe_algo = tpe.suggest
    # Create trials object
    tpe_trials = Trials()

    # Run times_to_evaluate evaluations with the tpe algorithm
    tpe_best = fmin(fn=objective2, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=times_to_evaluate)

    print_trials_information(hyperopt_trials=tpe_trials,
                             round_uniform=round_uniform,
                             hyperopt_choices=choices,
                             reverse_hyperopt_loguniforms=reverse_log_uniforms,
                             metric="Random minimum")
