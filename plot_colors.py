import argparse
import matplotlib.pyplot as plt

supported_cells = ["RRU", "GRU", "LSTM", "MogrifierLSTM"]
supported_datasets = ["Nottingham"]
supported_modes = ["values", "epochs"]


def parse_args():
	"""
        This function parses and returns the passed arguments.
        Output:
            vars: dict, a dictionary to find whether or not an argument was present and their corresponding values.
    """
	
	parser = argparse.ArgumentParser(description="Plot the 2D matrix with a heatmap.")
	parser.add_argument("-c", "--cell", type=str, default="RRU", help="Choose the cell for which to plot the data.")
	parser.add_argument("-d", "--dataset", type=str, default="Nottingham", help="Choose the dataset for which to plot the data.")
	parser.add_argument("-m", "--mode", type=str, default="values", help="Choose the mode to plot in – 'values' or 'epochs'.")
	parser.add_argument("-s", "--save", action="store_true", help="If present, save the plot to a file instead.")

	return vars(parser.parse_args())


def main():
	"""
        This is the main function. It controls the workflow of our program.
    """
	
	args = parse_args()  # Parse the passed argument(s)
	save_plot = args["save"]  # Should we save the plot instead
	cell_name = args["cell"]  # The dataset for which to plot the data
	data_set_name = args["dataset"]  # Choose the mode to plot in – 'value' or 'epochs'.
	mode_name = args["mode"]  # Choose the mode to plot in – 'value' or 'epochs'.

	if cell_name not in supported_cells:
		print("ERROR: No results for the given cell!")
		return
		
	if data_set_name not in supported_datasets:
		print("ERROR: No results for the given dataset!")
		return
		
	if mode_name not in supported_modes:
		print("ERROR: Given mode is not correct!")
		return

	name = f"{cell_name}_{data_set_name}_{mode_name}"

	if data_set_name == "Nottingham":
		# RRU Nottingham
		if cell_name == "RRU":
			if mode_name == "values":
				numbers = [
					[3.069, 3.079, 3.062, 3.062, 3.084, 3.037, 3.076, 3.061, 3.090, 3.071],
					[3.027, 3.010, 3.000, 3.049, 3.025, 3.032, 3.042, 3.053, 3.044, 3.051],
					[3.065, 3.009, 3.006, 3.013, 3.020, 3.028, 3.037, 3.016, 3.010, 3.023],
					[3.001, 3.006, 2.966, 2.972, 2.972, 2.984, 3.009, 3.026, 3.034, 3.019],
					[3.001, 2.961, 2.993, 2.988, 3.028, 2.978, 3.003, 3.014, 3.001, 3.025],
					[2.969, 2.998, 2.970, 2.965, 3.000, 3.022, 3.002, 3.018, 3.032, 3.016],
					[3.008, 2.997, 3.020, 3.024, 3.022, 3.039, 3.043, 3.017, 3.041, 3.067],
					[3.038, 3.043, 3.027, 3.034, 3.056, 3.053, 3.067, 3.038, 3.031, 3.057],
					[3.081, 3.075, 3.051, 3.025, 3.016, 3.026, 3.058, 3.031, 3.028, 3.028],
					[3.135, 3.011, 3.088, 3.071, 3.011, 3.071, 3.027, 3.087, 3.141, 3.082]
				]
				min_value = 2.961
				max_value = 3.814
			elif mode_name == "epochs":
				numbers = [
					[162, 85, 85, 68, 55, 69, 52, 45, 45, 50],
					[121, 65, 68, 39, 44, 44, 45, 33, 39, 37],
					[60, 48, 52, 39, 44, 42, 38, 33, 33, 36],
					[58, 45, 37, 36, 35, 33, 30, 32, 32, 29],
					[51, 43, 41, 33, 25, 31, 25, 30, 27, 21],
					[62, 37, 34, 34, 29, 28, 28, 27, 26, 25],
					[51, 34, 36, 24, 30, 28, 27, 26, 19, 21],
					[51, 37, 32, 35, 24, 31, 28, 31, 28, 24],
					[41, 30, 36, 35, 26, 34, 27, 29, 31, 38],
					[36, 64, 36, 35, 45, 31, 49, 34, 23, 30]
				]
				min_value = 7
				max_value = 360
		if cell_name == "GRU":
			if mode_name == "values":
				numbers = [
					[3.612, 3.579, 3.523, 3.513, 3.514, 3.511, 3.507, 3.502, 3.543, 3.481],
					[3.568, 3.522, 3.510, 3.500, 3.514, 3.462, 3.482, 3.484, 3.456, 3.477],
					[3.579, 3.506, 3.483, 3.491, 3.495, 3.462, 3.477, 3.444, 3.421, 3.445],
					[3.562, 3.503, 3.486, 3.481, 3.462, 3.456, 3.444, 3.441, 3.405, 3.434],
					[3.564, 3.511, 3.441, 3.434, 3.407, 3.382, 3.366, 3.377, 3.404, 3.373],
					[3.584, 3.481, 3.393, 3.330, 3.355, 3.368, 3.322, 3.302, 3.329, 3.357],
					[3.535, 3.361, 3.364, 3.302, 3.277, 3.292, 3.303, 3.267, 3.283, 3.285],
					[3.559, 3.519, 3.292, 3.298, 3.303, 3.302, 3.308, 3.312, 3.270, 3.292],
					[3.442, 3.465, 3.456, 3.450, 3.450, 3.492, 3.468, 3.522, 3.522, 3.501],
					[3.787, 3.790, 3.664, 3.763, 3.765, 3.732, 3.802, 3.776, 3.814, 3.791]
				]
				min_value = 2.961
				max_value = 3.814
			elif mode_name == "epochs":
				numbers = [
					[360, 213, 218, 177, 169, 154, 144, 137, 96, 126],
					[233, 157, 121, 112, 101, 89, 83, 84, 73, 75],
					[146, 113, 85, 84, 77, 75, 67, 61, 56, 56],
					[116, 88, 69, 57, 50, 57, 50, 42, 42, 44],
					[95, 58, 53, 54, 51, 45, 43, 38, 35, 37],
					[66, 45, 54, 42, 42, 44, 34, 34, 31, 36],
					[54, 53, 44, 39, 38, 39, 32, 33, 28, 27],
					[39, 27, 41, 29, 31, 34, 28, 36, 45, 28],
					[72, 58, 58, 51, 52, 37, 40, 26, 31, 32],
					[32, 16, 72, 30, 35, 30, 27, 32, 17, 15]
				]
				min_value = 7
				max_value = 360
		if cell_name == "LSTM":
			if mode_name == "values":
				numbers = [
					[3.704, 3.716, 3.784, 3.758, 3.633, 3.667, 3.770, 3.739, 3.725, 3.680],
					[3.638, 3.578, 3.610, 3.557, 3.546, 3.638, 3.533, 3.557, 3.546, 3.638],
					[3.588, 3.545, 3.497, 3.503, 3.480, 3.554, 3.506, 3.481, 3.496, 3.484],
					[3.502, 3.421, 3.436, 3.457, 3.428, 3.439, 3.441, 3.423, 3.440, 3.439],
					[3.428, 3.408, 3.396, 3.375, 3.395, 3.385, 3.399, 3.399, 3.407, 3.404],
					[3.400, 3.358, 3.354, 3.356, 3.351, 3.381, 3.404, 3.360, 3.373, 3.364],
					[3.381, 3.306, 3.303, 3.342, 3.322, 3.294, 3.351, 3.328, 3.315, 3.304],
					[3.331, 3.289, 3.261, 3.318, 3.309, 3.296, 3.306, 3.304, 3.344, 3.322],
					[3.300, 3.274, 3.253, 3.270, 3.295, 3.272, 3.257, 3.290, 3.266, 3.299],
					[3.307, 3.305, 3.292, 3.263, 3.289, 3.298, 3.284, 3.335, 3.306, 3.278]
				]
				min_value = 2.961
				max_value = 3.814
			elif mode_name == "epochs":
				numbers = [
					[360, 236, 190, 185, 195, 189, 140, 157, 139, 151],
					[189, 164, 140, 143, 106, 89, 112, 111, 95, 99],
					[146, 121, 111, 95, 97, 74, 76, 72, 77, 72],
					[126, 97, 70, 66, 57, 54, 60, 60, 51, 52],
					[87, 76, 58, 49, 54, 42, 46, 38, 36, 35],
					[76, 47, 41, 35, 37, 34, 28, 26, 29, 27],
					[42, 35, 30, 27, 25, 28, 24, 24, 21, 21],
					[33, 26, 23, 19, 23, 18, 20, 16, 19, 20],
					[28, 21, 19, 18, 20, 20, 16, 17, 19, 20],
					[22, 18, 19, 19, 17, 17, 17, 23, 17, 15]
				]
				min_value = 7
				max_value = 360
		if cell_name == "MogrifierLSTM":
			if mode_name == "values":
				numbers = [
					[3.522, 3.474, 3.476, 3.520, 3.523, 3.468, 3.539, 3.537, 3.671, 3.665],
					[3.496, 3.456, 3.452, 3.468, 3.476, 3.462, 3.554, 3.497, 3.638, 3.536],
					[3.477, 3.415, 3.413, 3.432, 3.396, 3.443, 3.483, 3.531, 3.516, 3.545],
					[3.446, 3.411, 3.405, 3.375, 3.369, 3.426, 3.449, 3.462, 3.484, 3.422],
					[3.418, 3.349, 3.408, 3.347, 3.352, 3.378, 3.393, 3.449, 3.380, 3.471],
					[3.422, 3.328, 3.319, 3.340, 3.335, 3.400, 3.348, 3.402, 3.396, 3.404],
					[3.355, 3.348, 3.310, 3.305, 3.325, 3.333, 3.335, 3.383, 3.384, 3.411],
					[3.402, 3.338, 3.377, 3.353, 3.327, 3.354, 3.321, 3.377, 3.359, 3.390],
					[3.316, 3.309, 3.316, 3.321, 3.338, 3.320, 3.328, 3.308, 3.324, 3.316],
					[3.303, 3.275, 3.331, 3.350, 3.348, 3.313, 3.300, 3.315, 3.322, 3.305]
				]
				min_value = 2.961
				max_value = 3.814
			elif mode_name == "epochs":
				numbers = [
					[204, 155, 112, 104, 101, 93, 84, 87, 82, 79],
					[123, 76, 75, 64, 65, 59, 55, 54, 46, 44],
					[92, 66, 55, 47, 44, 43, 39, 37, 38, 35],
					[63, 42, 34, 33, 33, 29, 30, 26, 31, 25],
					[43, 32, 29, 29, 23, 23, 21, 19, 20, 22],
					[34, 27, 22, 18, 18, 18, 16, 17, 15, 15],
					[30, 18, 16, 16, 13, 14, 14, 13, 11, 10],
					[21, 15, 14, 13, 12, 10, 10, 9, 9, 8],
					[22, 13, 9, 12, 10, 9, 11, 11, 9, 8],
					[20, 11, 14, 8, 15, 10, 10, 11, 9, 7]
				]
				min_value = 7
				max_value = 360

	if mode_name == "epochs":
		max_value = 100
		name += "_100"

	plt.imshow(numbers, cmap='viridis', vmin=min_value, vmax=max_value)
	# plt.matshow(numbers, cmap=plt.get_cmap('gray'), vmin=min_value, vmax=max_value)
	plt.colorbar()
	plt.title(name)
	if save_plot:
		plt.savefig(name + ".png")
	else:
		plt.show()


if __name__ == '__main__':
	main()
