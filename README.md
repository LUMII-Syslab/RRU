# RRU – Residual Recurrent Unit

This is project is made for creating an environment in which we can test the performance of our RRU cell against it's
competitors – GRU, LSTM, MogrifierLSTM

## Usage

### 1. Download the data sets (177 MB total)

To get the data sets, you must run get_data.sh.

P.S.: If you are on Windows, it might be a little tricky and not work, in that case you can use WSL, etc.
Code if you use WSL:
```
cd '/mnt/c/Users/YOUR PATH/RRU'
bash get_data.sh
```
	
When I added these files to git, it threw a message "warning: LF will be replaced by CRLF in get_data.sh.
", so I hope the script still works.

Language modeling data sets downloaded from:

http://mattmahoney.net/dc/textdata

and

http://www.fit.vutbr.cz/~imikolov/rnnlm/ .

Music modeling data sets downloaded from:

http://www-etud.iro.umontreal.ca/~boulanni/icml2012 .

### 2. Prepare your environment

DISCLAIMER: Parts of this project may work with your current TensorFlow environment (especially if you comment
out MogrifierLSTM code)

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

c) Set path (corresponding to the location you put it in):
```
export PATH=/host-dir/ronalds/anaconda3/bin:$PATH
```

or  

```
export PATH=/host-dir/ronalds/miniconda3/bin:$PATH
```

d) Test if the conda command works for you, for example, with:

```
conda env list
```

If it shows that it doesn't understand the command conda, refresh the shell, so you can use conda command

```
source .bashrc
```

2 - Create a new conda environment and activate it

a) "RRU" environment with some things installed
```
conda create -n RRU python=3.6.9 numpy scipy
```

b) Activate the environment
```
conda activate RRU
```

3 – Install the remaining requirements (dm-sonnet is needed for MogrifierLTSM; we don't use tensorflow_datasets anymore,
 but it's being imported in imdb_utils.py, in a function, that we don't use anymore)

```
conda install tensorflow-gpu=1.15
/host-dir/ronalds/miniconda3/envs/RRU/bin/pip3 install sklearn
/host-dir/ronalds/miniconda3/envs/RRU/bin/pip3 install --upgrade --upgrade-strategy only-if-needed dm-sonnet==1.36
/host-dir/ronalds/miniconda3/envs/RRU/bin/pip3 install tensorflow_datasets==3
```

### 3. Run the experiments (change the inner configuration if you wish)

### IMDB Sentiment Analysis

1 – Run imdb_main.py

### Language modeling

1 - To prepare the data, you must run lm_utils.py

You can change vocabulary_size for it in the main function, then you have to run it, for it to save them.

2 - To start training/validation/testing, run lm_main.py

Change hyperparameters inside to run different data sets, possibilities at the moment: enwik8, text8,
penn (Penn Treebank word-level), pennchar (Penn Treebank character-level)

### Music modeling

1 - To prepare the data, you must run music_utils.py

2 - To start training/validation/testing, run music_main.py

Change hyperparameters inside to run different data sets, possibilities at the moment: "JSB Chorales", "MuseData",
"Nottingham", "Piano-midi.de".
