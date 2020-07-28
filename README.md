# RRU comparison versus GRU and LSTM

### IMDB Sentiment Analysis

##### Newest way

Run cell_network.py

##### Old way

Run LSTM_with_tf_scan.py

### Language modeling

1 - To get the data, you must run lm_get_data.sh

If you are on Windows, it might be a little tricky and not work, in that case you can use WSL, etc.:
```
cd '/mnt/c/Users/YOUR PATH/rnn-lm-tensorflow'
bash get_data.sh
```
	
When I added these files to git, it threw message "warning: LF will be replaced by CRLF in get_data.sh.
", so I hope that it still works.

2 - To prepare the data, you must run lm_utils.py

You can change vocabulary_size, window_size, step_size for it in main function, then you have to run it, for it to save it.

3 - To start training/validation/testing, run lm_main.py

Change hyperparameters inside to run different data sets, possiblites atm: enwik8, text8, penn, pennchar, defaultly picks enwik8
