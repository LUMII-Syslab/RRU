echo "Downloading data sets!"
# Create the directory which will hold the data sets
# -p, --parents -> No error if the folder(s) exist, make parent directories as needed
mkdir -p data
cd data

echo "  Downloading language modeling data sets!"
mkdir -p lm
cd lm

mkdir -p unchanged
mkdir -p ready
cd unchanged

echo "    Downloading enwik8 data set!"
# --continue -> Continue getting a partially downloaded file (if there already is a file named like that in the current directory
# , wget will assume that it is the first portion of the remote file, and will ask the server to continue the retrieval from an offset
# equal to the length of the local file)
wget --continue http://mattmahoney.net/dc/enwik8.zip

echo "    Downloading text8 data set!"
wget --continue http://mattmahoney.net/dc/text8.zip

echo "    Downloading Penn Treebank (PTB) data set!"
# -q, --quiet -> Turn off wget's output
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# -x -> Extract files from an archive; -z -> Filter the archive through gzip(1); -f -> the next argument is the file name of the archive
# In a few words, it extracts it to a folder
tar -xzf simple-examples.tgz

echo "      Organizing Penn Treebank data set files!"

mkdir -p penn
cd penn
# mv -> Rename or move files
mv ../simple-examples/data/ptb.train.txt train.txt
mv ../simple-examples/data/ptb.valid.txt valid.txt
mv ../simple-examples/data/ptb.test.txt test.txt
cd ..

mkdir -p pennchar
cd pennchar
mv ../simple-examples/data/ptb.char.train.txt train.txt
mv ../simple-examples/data/ptb.char.valid.txt valid.txt
mv ../simple-examples/data/ptb.char.test.txt test.txt
cd ..

# -r -> recursive removal; -f -> remove files without prompting you to confirm that you wish to have them removed
rm -rf simple-examples/

cd ..
cd ..

echo "  Downloading music modeling data sets!"
mkdir -p mm
cd mm

mkdir -p unchanged
mkdir -p ready
cd unchanged

echo "    Downloading Piano-midi.de data set!"
wget --continue http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle

echo "    Downloading Nottingham data set!"
wget --continue http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle

echo "    Downloading MuseData data set!"
wget --continue http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle

echo "    Downloading JSB Chorales data set!"
wget --continue "http://www-etud.iro.umontreal.ca/~boulanni/JSB Chorales.pickle"

echo "Downloading finished!"

