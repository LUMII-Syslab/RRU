# RRU – Residual Recurrent Unit

This is project is made for creating an environment in which we can test the performance of our RRU cell against it's competitors – GRU, LSTM, MogrifierLSTM

## Usage

DISCLAIMER: Parts of this project may work with your current environment (especially if you comment out MogrifierLSTM code)

1 – Install Anaconda (if you do not have it yet)

a) Download the installer
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
```

b) Run the installer
```
bash Anaconda3-2020.07-Linux-x86_64.sh
```

c) Refresh the shell, so you can use conda command
```
source .bashrc
```

2 - Create a new conda environment and activate it

a) "test_mogrifier" environment with some things installed
```
conda create -n test_mogrifier python=3.6.9 numpy scipy
```

b) Activate the environment
```
conda activate test_mogrifier
```

3 – Install the remaining requirements
```
conda install cudatoolkit
conda install cudnn
conda install tensorflow-gpu=1.15
/root/anaconda3/envs/test_mogrifier/bin/pip3 install sklearn
/root/anaconda3/envs/test_mogrifier/bin/pip3 install --upgrade --upgrade-strategy only-if-needed dm-sonnet==1.36
```

4 – You might have to download the data, if you haven't done it yet (depends on the task)

You can see the instructions for each task below.

5 – Run the experiments (change the inner configuration if you wish)
```
python3 -u lm_efficient_main.py
```

## Tasks

### Language modeling

#### Newest way (less memory usage)

1 - To get the data, you must run lm_get_data.sh

If you are on Windows, it might be a little tricky and not work, in that case you can use WSL, etc.:
```
cd '/mnt/c/Users/YOUR PATH/RRU'
bash lm_get_data.sh
```
	
When I added these files to git, it threw message "warning: LF will be replaced by CRLF in get_data.sh.
", so I hope the script still works.

2 - To prepare the data, you must run lm_efficient_utils.py

You can change vocabulary_size for it in the main function, then you have to run it, for it to save them.

3 - To start training/validation/testing, run lm_efficient_main.py

Change hyperparameters inside to run different data sets, possibilites at the moment: enwik8, text8, penn, pennchar, defaultly picks enwik8

#### Old way (more memory usage)

1 - To get the data, you must run lm_get_data.sh

If you are on Windows, it might be a little tricky and not work, in that case you can use WSL, etc.:
```
cd '/mnt/c/Users/YOUR PATH/RRU'
bash lm_get_data.sh
```
	
When I added these files to git, it threw message "warning: LF will be replaced by CRLF in get_data.sh.
", so I hope the script still works.

2 - To prepare the data, you must run lm_utils.py

You can change vocabulary_size, window_size, step_size for it in main function, then you have to run it, for it to save it.

3 - To start training/validation/testing, run lm_main.py

Change hyperparameters inside to run different data sets, possiblites atm: enwik8, text8, penn, pennchar, defaultly picks enwik8

### IMDB Sentiment Analysis

#### Newest way (Easy to run other cells)

Run cell_network.py

#### Old way (LSTM programmed by hand, so harder to run other cells)

Run LSTM_with_tf_scan.py
