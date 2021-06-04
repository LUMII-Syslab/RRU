import argparse
import matplotlib.pyplot as plt

supported_cells = ["RRU", "GRU", "LSTM", "MogrifierLSTM", "RRU-nonorm"]
supported_datasets = ["Nottingham", "Penn Treebank"]
supported_modes = ["values", "epochs", "bananas", "RRUbattle"]


data = {
	"Nottingham": {
		"min_value": 2.961,
		"max_value": 3.814,  # If ignore NaNs – 3.79,
		"min_epochs": 7,
		"max_epochs": 360,
		"metric": "NLL",
		"x_labels": ["1.0", "2.6", "4.1", "5.7", "7.2", "8.8", "10.3", "11.9", "13.4", "15.0"],
		"y_labels": list(reversed(["100", "60", "36", "22", "13", "8", "5", "3", "2", "1"])),
		"RRU": {
			"values": list(reversed([
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
			])),
			"epochs": list(reversed([
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
			]))
		},
		"GRU": {
			"values": list(reversed([
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
			])),
			"epochs": list(reversed([
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
			]))
		},
		"LSTM": {
			"values": list(reversed([
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
			])),
			"epochs": list(reversed([
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
			]))
		},
		"MogrifierLSTM": {
			"values": list(reversed([
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
			])),
			"epochs": list(reversed([
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
			]))
		},
		"RRU-nonorm": {
			"values": list(reversed([
				[3.357, 3.361, 3.354, 3.101, 3.141, 3.419, 3.342, 3.112, 3.140, 3.382],
				[3.062, 3.055, 3.024, 3.049, 3.039, 3.013, 3.074, 3.023, 3.034, 3.048],
				[3.024, 3.003, 3.026, 3.017, 3.029, 3.002, 2.991, 3.053, 3.025, 3.024],
				[3.004, 2.999, 2.982, 2.970, 2.984, 3.011, 2.981, 3.007, 2.953, 2.992],
				[3.009, 2.980, 2.977, 2.959, 2.983, 3.008, 2.985, 2.994, 2.986, 2.988],
				[2.996, 2.967, 2.970, 2.948, 2.982, 2.988, 2.983, 3.002, 2.986, 2.986],
				[2.993, 2.959, 2.998, 2.995, 2.971, 2.986, 2.970, 2.982, 2.990, 3.022],
				[3.024, 3.044, 3.042, 3.042, 3.048, 3.042, 3.032, 3.065, 3.048, 3.055],
				[3.150, 3.128, 9.999, 3.180, 9.999, 9.999, 9.999, 3.183, 9.999, 3.232],
				[3.317, 3.397, 3.435, 3.465, 3.545, 3.573, 9.999, 9.999, 9.999, 9.999]
			])),
			"epochs": list(reversed([
				[262, 187, 159, 208, 177, 123, 126, 154, 145, 109],
				[216, 130, 140, 110, 114, 99, 94, 100, 112, 93],
				[157, 116, 96, 94, 90, 78, 81, 66, 73, 68],
				[116, 80, 78, 68, 67, 52, 62, 55, 57, 59],
				[85, 70, 65, 51, 46, 58, 45, 44, 46, 47],
				[80, 54, 54, 45, 45, 39, 34, 39, 42, 43],
				[68, 44, 42, 43, 36, 37, 32, 29, 31, 27],
				[54, 29, 45, 32, 39, 22, 26, 28, 29, 30],
				[33, 25, 1, 24, 1, 1, 1, 15, 1, 18],
				[30, 15, 16, 13, 9, 10, 1, 1, 1, 1]
			]))
		}
	},
	"Penn Treebank": {
		"min_value": 104.5,
		"max_value": 795.5,
		"min_epochs": 1,
		"max_epochs": 551,
		"metric": "Perplexity",
		"x_labels": ["10.0", "12.2", "14.4", "16.7", "18.9", "21.1", "23.3", "25.6", "27.8", "30.0"],
		# "y_labels": list(reversed(["0.01", "0.006", "0.0036", "0.0022", "0.0013",
		# "0.0008", "0.0005", "0.0003", "0.0002", "0.0001"])),
		"y_labels": list(reversed(["100", "60", "36", "22", "13", "8", "5", "3", "2", "1"])),
		"RRU": {
			"values": list(reversed([
				[122.2, 122.7, 121.3, 122.9, 122.3, 123.8, 123.9, 123.1, 124.2, 123.9],
				[120.1, 120.6, 121.2, 122.2, 121.8, 122.7, 122.2, 122.6, 124.3, 123.7],
				[119.8, 121.4, 119.7, 120.9, 122.2, 121.2, 121.4, 123.6, 124.2, 123.5],
				[117.9, 118.0, 120.1, 121.3, 119.1, 122.0, 121.7, 122.3, 121.6, 122.7],
				[117.9, 117.5, 117.3, 118.2, 119.3, 119.0, 120.1, 122.4, 119.3, 119.7],
				[115.4, 116.1, 117.2, 117.5, 118.2, 118.3, 118.0, 118.4, 117.7, 118.6],
				[113.6, 113.9, 114.0, 114.0, 113.8, 112.8, 116.5, 114.8, 113.6, 115.4],
				[109.0, 110.0, 109.4, 111.7, 110.3, 109.8, 110.1, 111.5, 112.1, 109.2],
				[106.7, 104.5, 106.8, 105.3, 105.0, 105.6, 105.8, 107.8, 107.1, 104.6],
				[107.6, 107.6, 114.4, 109.8, 109.2, 108.6, 108.8, 109.8, 109.4, 108.1]
			])),
			"epochs": list(reversed([
				[59, 53, 53, 43, 43, 38, 38, 32, 30, 35],
				[42, 35, 30, 28, 28, 24, 24, 21, 18, 23],
				[32, 25, 25, 20, 21, 21, 21, 17, 16, 16],
				[33, 28, 22, 15, 16, 14, 14, 12, 15, 13],
				[23, 22, 19, 16, 13, 14, 12, 11, 13, 12],
				[18, 16, 16, 14, 11, 13, 12, 12, 12, 11],
				[22, 15, 19, 14, 15, 16, 10, 12, 12, 10],
				[21, 18, 21, 14, 16, 15, 16, 15, 12, 15],
				[25, 29, 20, 22, 28, 22, 24, 17, 22, 23],
				[37, 34, 33, 27, 34, 24, 33, 29, 40, 37]
			]))
		},
		"GRU": {
			"values": list(reversed([
				[128.5, 136.4, 137.9, 141.4, 143.5, 142.7, 144.0, 146.6, 147.5, 149.4],
				[133.5, 140.2, 141.4, 141.2, 141.3, 144.5, 143.4, 147.5, 144.0, 145.5],
				[133.8, 136.0, 138.7, 139.3, 141.1, 141.8, 144.9, 142.1, 143.9, 145.9],
				[136.3, 136.6, 140.0, 140.9, 141.2, 139.1, 137.5, 143.3, 141.8, 141.7],
				[135.4, 136.9, 136.1, 138.2, 140.3, 137.5, 139.0, 144.3, 143.9, 145.3],
				[132.1, 131.7, 135.2, 135.1, 136.9, 137.9, 143.4, 149.8, 146.0, 147.2],
				[129.9, 138.0, 140.9, 144.0, 142.2, 141.4, 140.5, 147.8, 143.4, 147.0],
				[132.5, 138.7, 136.2, 137.2, 142.0, 138.7, 141.4, 146.6, 143.8, 147.0],
				[137.6, 137.7, 144.0, 141.6, 146.1, 145.2, 149.8, 150.2, 723.0, 148.9],
				[153.2, 144.1, 142.7, 278.8, 440.3, 147.6, 795.5, 207.3, 778.2, 606.6]
			])),
			"epochs": list(reversed([
				[283, 245, 214, 163, 192, 168, 158, 150, 133, 130],
				[150, 130, 120, 92, 88, 88, 76, 71, 62, 66],
				[109, 79, 72, 61, 58, 46, 53, 43, 45, 42],
				[47, 55, 39, 39, 38, 32, 26, 28, 27, 24],
				[38, 35, 30, 23, 24, 20, 20, 23, 19, 20],
				[26, 24, 17, 19, 17, 15, 13, 16, 13, 17],
				[17, 18, 15, 15, 12, 11, 10, 12, 10, 7],
				[15, 10, 12, 12, 10, 11, 11, 7, 7, 10],
				[11, 11, 10, 11, 10, 10, 10, 12, 7, 10],
				[10, 10, 15, 5, 33, 9, 4, 7, 4, 19]
			]))
		},
		"LSTM": {
			"values": list(reversed([
				[196.6, 169.5, 721.5, 143.7, 722.4, 719.1, 148.7, 723.9, 722.2, 723.3],
				[725.4, 726.1, 154.8, 726.0, 727.6, 728.5, 728.5, 725.4, 151.5, 155.4],
				[725.0, 722.9, 726.7, 725.9, 155.5, 151.6, 154.8, 158.1, 155.6, 161.8],
				[157.9, 153.6, 151.9, 142.9, 151.3, 153.4, 152.5, 155.2, 155.5, 154.5],
				[154.4, 145.3, 148.4, 149.5, 150.8, 152.4, 151.4, 153.5, 157.7, 155.8],
				[142.4, 149.2, 152.3, 149.5, 149.5, 151.2, 155.3, 156.1, 160.1, 161.3],
				[148.0, 151.3, 151.0, 154.4, 160.3, 157.5, 155.8, 155.1, 160.4, 156.3],
				[156.4, 150.5, 156.6, 151.1, 152.6, 149.9, 162.2, 161.0, 159.5, 164.0],
				[147.8, 158.0, 163.5, 158.5, 158.1, 163.6, 163.0, 163.0, 169.4, 166.9],
				[155.7, 158.5, 162.1, 157.8, 163.0, 160.0, 159.6, 159.2, 158.9, 163.5]
			])),
			"epochs": list(reversed([
				[551, 478, 2, 278, 2, 10, 213, 2, 3, 3],
				[1, 1, 149, 1, 1, 1, 2, 2, 66, 64],
				[1, 1, 1, 1, 69, 58, 49, 44, 40, 34],
				[89, 62, 50, 35, 34, 29, 29, 28, 27, 25],
				[47, 33, 25, 19, 21, 18, 18, 16, 16, 17],
				[23, 17, 16, 14, 14, 13, 12, 12, 11, 11],
				[12, 12, 11, 10, 9, 8, 8, 8, 7, 8],
				[8, 9, 7, 7, 7, 7, 6, 6, 7, 6],
				[7, 6, 5, 5, 5, 5, 4, 5, 5, 5],
				[5, 5, 4, 4, 4, 5, 4, 5, 6, 8]
			]))
		},
		"MogrifierLSTM": {
			"values": list(reversed([
				[726.7, 738.6, 728.1, 727.3, 727.5, 728.9, 729.8, 143.7, 147.9, 147.1],
				[726.5, 137.6, 148.2, 144.7, 144.7, 145.5, 145.8, 147.0, 142.6, 148.0],
				[143.3, 146.5, 142.0, 146.9, 141.1, 149.4, 150.7, 156.1, 151.1, 156.5],
				[145.5, 144.5, 138.5, 145.6, 140.7, 149.7, 152.8, 154.2, 138.4, 148.9],
				[134.4, 136.0, 137.0, 134.6, 141.4, 141.7, 134.8, 134.3, 130.4, 130.0],
				[133.4, 133.2, 130.1, 134.3, 141.0, 149.2, 145.5, 144.7, 142.4, 143.2],
				[140.3, 145.4, 143.1, 141.1, 143.1, 141.2, 139.2, 139.1, 137.9, 139.3],
				[137.6, 137.6, 138.4, 138.9, 139.6, 145.2, 146.4, 144.4, 144.2, 143.2],
				[137.8, 141.7, 137.9, 142.7, 139.7, 138.2, 139.3, 136.5, 145.9, 143.5],
				[135.8, 136.1, 136.0, 153.2, 138.9, 144.7, 143.5, 153.4, 186.3, 159.4]
			])),
			"epochs": list(reversed([
				[2, 1, 1, 1, 1, 1, 1, 93, 84, 86],
				[1, 118, 92, 93, 75, 70, 56, 60, 52, 55],
				[91, 92, 78, 60, 56, 47, 47, 39, 36, 32],
				[69, 55, 42, 39, 29, 30, 28, 25, 25, 23],
				[35, 27, 24, 22, 22, 20, 18, 18, 16, 15],
				[21, 20, 17, 16, 14, 11, 10, 10, 10, 10],
				[14, 12, 11, 10, 9, 8, 8, 8, 7, 7],
				[10, 9, 8, 7, 7, 6, 6, 5, 5, 5],
				[7, 6, 6, 5, 5, 5, 5, 4, 4, 4],
				[5, 5, 5, 3, 5, 4, 5, 5, 28, 5]
			]))
		}
	}
}


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
	parser.add_argument("-t", "--trim", type=float, default=None, help="Trim max value at what value.")
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
	trim_max = args["trim"]  # Trim max value at what value.

	if cell_name not in supported_cells:
		print("ERROR: No results for the given cell!")
		return
		
	if data_set_name not in supported_datasets:
		print("ERROR: No results for the given dataset!")
		return
		
	if mode_name not in supported_modes:
		print("ERROR: Given mode is not correct!")
		return

	if mode_name not in ["bananas", "RRUbattle"]:
		numbers = data[data_set_name][cell_name][mode_name]
		if mode_name == "values":
			min_value = data[data_set_name]["min_value"]
			max_value = data[data_set_name]["max_value"]
		elif mode_name == "epochs":
			min_value = data[data_set_name]["min_epochs"]
			max_value = data[data_set_name]["max_epochs"]

		if trim_max is not None:
			max_value = trim_max
			name += f"_{max_value}"
			
		''' Old way
		plt.imshow(numbers, cmap='viridis', vmin=min_value, vmax=max_value)
		# plt.matshow(numbers, cmap=plt.get_cmap('gray'), vmin=min_value, vmax=max_value)
		plt.colorbar()
		plt.title(name)
		plt.xlabel("Number of parameters")
		plt.ylabel("Learning rate")
		if save_plot:
			plt.savefig(name + ".png")
		else:
			plt.show()
		'''
		
		name = f"{cell_name}_{data_set_name}_{mode_name}"
		
		fig, ax = plt.subplots(1,1)

		img = ax.imshow(numbers, cmap='viridis', vmin=min_value, vmax=max_value, extent=[0, 10, 0, 10])

		x_label_list = data[data_set_name]["x_labels"]
		y_label_list = data[data_set_name]["y_labels"]

		ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
		ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])

		ax.set_xticklabels(x_label_list, fontsize=8)
		ax.set_yticklabels(y_label_list)

		plt.xlabel("Number of parameters (M)")
		plt.ylabel("Learning rate * 1e-4")

		metric = data[data_set_name]["metric"] if mode_name == "values" else "Epochs"
		fig.colorbar(img, label=metric)

		plt.title(f"{data_set_name} {cell_name} grid of {mode_name}")  # (title, fontsize=8)
		
		if save_plot:
			plt.savefig(name + ".png")
		else:
			plt.show()
	elif mode_name == "RRUbattle":
		RRU_values = data[data_set_name]["RRU"]["values"]
		RRU_epochs = data[data_set_name]["RRU"]["epochs"]
		RRU_nonorm_values = data[data_set_name]["RRU-nonorm"]["values"]
		RRU_nonorm_epochs = data[data_set_name]["RRU-nonorm"]["epochs"]

		min_value = 2.948
		max_value = 3.545

		if trim_max is not None:
			max_value = trim_max
		
		min_epochs = 0  # 9
		max_epochs = 100  # 262
		
		# Let's create the figure in which we will show all the plots
		fig = plt.figure(figsize=(14, 6))  # Width and height in inches		
		#figtile
		# fig, ax = plt.subplots(1,1)

		x_label_list = data[data_set_name]["x_labels"]
		y_label_list = data[data_set_name]["y_labels"]
		extent_list = [0, 10, 0, 10]
		tick_list = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

		ax = fig.add_subplot(231)
		img = ax.imshow(RRU_values, cmap='viridis', vmin=min_value, vmax=max_value, extent=extent_list)
		ax.set_xticks(tick_list)
		ax.set_yticks(tick_list)
		x_label_list1 = [""] * 10
		x_label_list1[0] = x_label_list[0]
		x_label_list1[4] = x_label_list[4]
		x_label_list1[8] = x_label_list[8]
		ax.set_xticklabels(x_label_list1)  # , fontsize=8)
		ax.set_yticklabels(y_label_list)
		ax.xaxis.tick_top()
		ax.xaxis.set_label_position('top') 
		plt.xlabel("RRU\nNumber of parameters (M)")
		plt.ylabel("Learning rate (1e-4)")

		ax2 = fig.add_subplot(232)
		img = ax2.imshow(RRU_nonorm_values, cmap='viridis', vmin=min_value, vmax=max_value, extent=extent_list)
		ax2.set_xticks(tick_list)
		ax2.set_yticks([])
		x_label_list2 = [""] * 10
		x_label_list2[1] = x_label_list[1]
		x_label_list2[5] = x_label_list[5]
		x_label_list2[9] = x_label_list[9]
		ax2.set_xticklabels(x_label_list2)
		ax2.set_yticklabels([])
		ax2.xaxis.tick_top()
		ax2.xaxis.set_label_position('top') 
		plt.xlabel("RRU no normalization\nNumber of parameters (M)")

		ax5 = fig.add_subplot(233)
		ax5.set_visible(False)

		fig.colorbar(img, label=data[data_set_name]["metric"])
		
		ax6 = fig.add_subplot(234)
		img = ax6.imshow(RRU_epochs, cmap='viridis', vmin=min_epochs, vmax=max_epochs, extent=extent_list)
		ax6.set_xticks(tick_list)
		ax6.set_yticks(tick_list)
		x_label_list6 = [""] * 10
		x_label_list6[0] = x_label_list[0]
		x_label_list6[4] = x_label_list[4]
		x_label_list6[8] = x_label_list[8]
		ax6.set_xticklabels(x_label_list6)
		ax6.set_yticklabels(y_label_list)
		plt.xlabel("Number of parameters (M)")
		plt.ylabel("Learning rate (1e-4)")
		
		ax7 = fig.add_subplot(235)
		img = ax7.imshow(RRU_nonorm_epochs, cmap='viridis', vmin=min_epochs, vmax=max_epochs, extent=extent_list)
		ax7.set_xticks(tick_list)
		ax7.set_yticks([])
		x_label_list7 = [""] * 10
		x_label_list7[1] = x_label_list[1]
		x_label_list7[5] = x_label_list[5]
		x_label_list7[9] = x_label_list[9]
		ax7.set_xticklabels(x_label_list7)
		ax7.set_yticklabels([])
		plt.xlabel("Number of parameters (M)")
		
		ax10 = fig.add_subplot(236)
		ax10.set_visible(False)
		
		fig.colorbar(img, label="Epochs")
		
		if save_plot:
			plt.savefig(f"{data_set_name}_diversity.png")
		else:
			plt.show()
	else:
		RRU_values = data[data_set_name]["RRU"]["values"]
		RRU_epochs = data[data_set_name]["RRU"]["epochs"]
		GRU_values = data[data_set_name]["GRU"]["values"]
		GRU_epochs = data[data_set_name]["GRU"]["epochs"]
		LSTM_values = data[data_set_name]["LSTM"]["values"]
		LSTM_epochs = data[data_set_name]["LSTM"]["epochs"]
		MogrifierLSTM_values = data[data_set_name]["MogrifierLSTM"]["values"]
		MogrifierLSTM_epochs = data[data_set_name]["MogrifierLSTM"]["epochs"]

		min_value = data[data_set_name]["min_value"]
		max_value = data[data_set_name]["max_value"]
		
		min_epochs = data[data_set_name]["min_epochs"]
		max_epochs = data[data_set_name]["max_epochs"]

		if trim_max is not None:
			max_value = trim_max
		
		min_epochs = 0
		max_epochs = 100
		
		# Let's create the figure in which we will show all the plots
		fig = plt.figure(figsize=(14, 6))  # Width and height in inches		
		#figtile
		# fig, ax = plt.subplots(1,1)

		x_label_list = data[data_set_name]["x_labels"]
		y_label_list = data[data_set_name]["y_labels"]
		extent_list = [0, 10, 0, 10]
		tick_list = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

		ax = fig.add_subplot(251)
		img = ax.imshow(RRU_values, cmap='viridis', vmin=min_value, vmax=max_value, extent=extent_list)
		ax.set_xticks(tick_list)
		ax.set_yticks(tick_list)
		x_label_list1 = [""] * 10
		x_label_list1[0] = x_label_list[0]
		x_label_list1[4] = x_label_list[4]
		x_label_list1[8] = x_label_list[8]
		ax.set_xticklabels(x_label_list1)  # , fontsize=8)
		ax.set_yticklabels(y_label_list)
		ax.xaxis.tick_top()
		ax.xaxis.set_label_position('top') 
		plt.xlabel("RRU\nNumber of parameters (M)")
		plt.ylabel("Learning rate (×1e-4)")

		ax2 = fig.add_subplot(252)
		img = ax2.imshow(GRU_values, cmap='viridis', vmin=min_value, vmax=max_value, extent=extent_list)
		ax2.set_xticks(tick_list)
		ax2.set_yticks([])
		x_label_list2 = [""] * 10
		x_label_list2[1] = x_label_list[1]
		x_label_list2[5] = x_label_list[5]
		x_label_list2[9] = x_label_list[9]
		ax2.set_xticklabels(x_label_list2)
		ax2.set_yticklabels([])
		ax2.xaxis.tick_top()
		ax2.xaxis.set_label_position('top') 
		plt.xlabel("GRU\nNumber of parameters (M)")
		
		ax3 = fig.add_subplot(253)
		img = ax3.imshow(LSTM_values, cmap='viridis', vmin=min_value, vmax=max_value, extent=extent_list)
		ax3.set_xticks(tick_list)
		ax3.set_yticks([])
		x_label_list3 = [""] * 10
		x_label_list3[2] = x_label_list[2]
		x_label_list3[6] = x_label_list[6]
		ax3.set_xticklabels(x_label_list3)
		ax3.set_yticklabels([])
		ax3.xaxis.tick_top()
		ax3.xaxis.set_label_position('top') 
		plt.xlabel("LSTM\nNumber of parameters (M)")
		
		ax4 = fig.add_subplot(254)
		img = ax4.imshow(MogrifierLSTM_values, cmap='viridis', vmin=min_value, vmax=max_value, extent=extent_list)
		ax4.set_xticks(tick_list)
		ax4.set_yticks([])
		x_label_list4 = [""] * 10
		x_label_list4[3] = x_label_list[3]
		x_label_list4[7] = x_label_list[7]
		ax4.set_xticklabels(x_label_list4)
		ax4.set_yticklabels([])
		ax4.xaxis.tick_top()
		ax4.xaxis.set_label_position('top') 
		plt.xlabel("MogrifierLSTM\nNumber of parameters (M)")

		ax5 = fig.add_subplot(255)
		ax5.set_visible(False)

		fig.colorbar(img, label=data[data_set_name]["metric"])
		
		ax6 = fig.add_subplot(256)
		img = ax6.imshow(RRU_epochs, cmap='viridis', vmin=min_epochs, vmax=max_epochs, extent=extent_list)
		ax6.set_xticks(tick_list)
		ax6.set_yticks(tick_list)
		x_label_list6 = [""] * 10
		x_label_list6[0] = x_label_list[0]
		x_label_list6[4] = x_label_list[4]
		x_label_list6[8] = x_label_list[8]
		ax6.set_xticklabels(x_label_list6)
		ax6.set_yticklabels(y_label_list)
		plt.xlabel("Number of parameters (M)")
		plt.ylabel("Learning rate (×1e-4)")
		
		ax7 = fig.add_subplot(257)
		img = ax7.imshow(GRU_epochs, cmap='viridis', vmin=min_epochs, vmax=max_epochs, extent=extent_list)
		ax7.set_xticks(tick_list)
		ax7.set_yticks([])
		x_label_list7 = [""] * 10
		x_label_list7[1] = x_label_list[1]
		x_label_list7[5] = x_label_list[5]
		x_label_list7[9] = x_label_list[9]
		ax7.set_xticklabels(x_label_list7)
		ax7.set_yticklabels([])
		plt.xlabel("Number of parameters (M)")
		
		ax8 = fig.add_subplot(258)
		img = ax8.imshow(LSTM_epochs, cmap='viridis', vmin=min_epochs, vmax=max_epochs, extent=extent_list)
		ax8.set_xticks(tick_list)
		ax8.set_yticks([])
		x_label_list8 = [""] * 10
		x_label_list8[2] = x_label_list[2]
		x_label_list8[6] = x_label_list[6]
		ax8.set_xticklabels(x_label_list8)
		ax8.set_yticklabels([])
		plt.xlabel("Number of parameters (M)")
		
		ax9 = fig.add_subplot(259)
		img = ax9.imshow(MogrifierLSTM_epochs, cmap='viridis', vmin=min_epochs, vmax=max_epochs, extent=extent_list)
		ax9.set_xticks(tick_list)
		ax9.set_yticks([])
		x_label_list9 = [""] * 10
		x_label_list9[3] = x_label_list[3]
		x_label_list9[7] = x_label_list[7]
		ax9.set_xticklabels(x_label_list9)
		ax9.set_yticklabels([])
		plt.xlabel("Number of parameters (M)")
		
		ax10 = fig.add_subplot(2, 5, 10)
		ax10.set_visible(False)
		
		fig.colorbar(img, label="Epochs")
		
		if save_plot:
			plt.savefig(f"{data_set_name}_diversity.png")
		else:
			plt.show()


if __name__ == '__main__':
	main()
