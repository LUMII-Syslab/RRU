# Gates are not what you need in RNNs

This repository is the official implementation of [Gates are not what you need in RNNs](https://www.google.com/search?q=google+arxiv). 

<!-- 📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

### 1. Download the data sets (177 MB total)

To get the data sets, you must run the script which downloads the data by:
```
bash get_data.sh
```


P.S. If you are on Windows, it might be a little tricky and not work, in that case you can use WSL, etc.
<!--```
cd '/mnt/c/Users/YOUR PATH/RRU'
bash get_data.sh
```-->
	
When these files were added to this Git repository, it threw a message "warning: LF will be replaced by CRLF in get_data.sh.
", if this script doesn't work, the newline symbols have to be changed to match the system.

Language modeling data sets are downloaded from:

http://mattmahoney.net/dc/textdata

and

http://www.fit.vutbr.cz/~imikolov/rnnlm/ .

Music modeling data sets are downloaded from:

http://www-etud.iro.umontreal.ca/~boulanni/icml2012 .

### 2. Prepare your environment

DISCLAIMER: Parts of this project may work with your current TensorFlow environment, if you have one (especially if you comment
out the MogrifierLSTM code)

1 – Install Anaconda (if you do not have it yet)

a) Download the installer
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
```

or

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

b) Run the installer
```
bash Anaconda3-2020.07-Linux-x86_64.sh
```

or

```
bash Miniconda3-latest-Linux-x86_64.sh
```

c) Set the path (corresponding to the location you put it in):
```
export PATH=/host-dir/ronalds/anaconda3/bin:$PATH
```

or  

```
export PATH=/host-dir/ronalds/miniconda3/bin:$PATH
```

d) Test if the "conda" command works for you, for example, with:

```
conda env list
```

If it shows that it doesn't understand the command "conda", refreshing the shell usually helps:

```
source .bashrc
```

2 - Create a new "conda" environment and activate it

a) "RRU" environment with some things installed
```
conda create -n RRU python=3.6.9 numpy scipy
```

b) Activate the environment
```
conda activate RRU
```

3 – Install the remaining requirements (dm-sonnet is needed for the MogrifierLTSM)

```
conda install tensorflow-gpu=1.15
/host-dir/ronalds/miniconda3/envs/RRU/bin/pip3 install sklearn
/host-dir/ronalds/miniconda3/envs/RRU/bin/pip3 install --upgrade --upgrade-strategy only-if-needed dm-sonnet==1.36
/host-dir/ronalds/miniconda3/envs/RRU/bin/pip3 install hyperopt
```

## Training & Evaluation

The models are evaluated automatically after the training has been finished.

### Sentiment Analysis
1 - To start training / validation / testing, run:
```
python -u sa_main.py
```

You change the training / validation / testing hyperparameters at the top of the "sa_main.py" file.

### Language modeling

1 - To prepare the data, run:
```
python -u lm_utils.py
```

You can change the vocabulary size for each data set in the main function in "lm_utils.py". After any changes, you must
rerun the lm_utils.py file.

2 - To start training / validation / testing, run:
```
python -u lm_main.py
```

You change the training / validation / testing hyperparameters at the top of the "lm_main.py" file.

### Music modeling

1 - To prepare the data, run:
```
python -u mm_utils.py
```

2 - To start training/validation/testing, run:
```
python -u mm_main.py
```

You change the training / validation / testing hyperparameters at the top of the "mm_main.py" file.

### MNIST image classification

1 - To prepare the data, run:
```
python -u mnist_utils.py
```

2 - To start training/validation/testing, run:
```
python -u mnist_main.py
```

You change the training / validation / testing hyperparameters at the top of the "mnist_main.py" file.

## Results

Our model achieves the following performance on aforementioned data sets:

<!-- ### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet) -->

| Task                                  | Tuned | RRU        | GRU        | LSTM   | Mogrifier LSTM |
| ------------------------------------- | ----- | ---------- | ---------- | ------ | -------------- |
| Music JSB Chorales (NLL)              | +     | **7.72**   | 8.20       | 8.33   | 8.18           |
| Music Nottingham (NLL)                | +     | **2.92**   | 3.22       | 3.25   | 3.28           |
| Music MuseData (NLL)                  | +     | **7.03**   | 7.31       | 7.24   | 7.22           |
| Music Piano-midi.de (NLL)             | +     | **7.38**   | 7.58       | 7.54   | 7.52           |
| Word-level Penn Treebank (Perplexity) | +     | **102.56** | 122.21     | 140.35 | 126.88         |
| Character-level Penn Treebank (BPC)   | +/-   | **1.27**   | 1.28       | 1.34   | 1.29           |
| Character-level enwik8 (BPC)          | -     | 1.37       | 1.53       | 1.49   | **1.36**       |
| Character-level text8 (BPC)           | -     | **1.35**   | 1.55       | 1.44   | 1.37           |
| Sentiment analysis IMDB (Accuracy)    | -     | **87.20**  | 87.04      | 85.89  | 86.23          |
| Sequential MNIST (Accuracy)           | -     | **98.74**  | 98.36      | 92.88  | 98.14          |
| Permuted MNIST (Accuracy)             | -     | 97.67      | **98.68**  | 97.39  | 97.81          |

<!--
## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
-->