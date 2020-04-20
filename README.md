![University of Colorado Colorado Springs](https://github.com/kadengriffith/MWP-Automatic-Solver/blob/master/data/util/UCCS.png)

# MWP Automatic Solver

---

This repo is the home to an automatic solver for math word problems using the 2017 Transformer model. This research was conducted at the University of Colorado Colorado Springs by Kaden Griffith and Jugal Kalita in 2019-2020. This research was a part of an undergraduate program through the National Science Foundation and grant research done through our university. This Transformer arrangement translates a word problem into a useable expression. We did not provide comprehensive code to solve the output. Our best model surpassed state-of-the-art, receiving an average of 86.7% accuracy among all test sets.

This version supports some features that were not used in the publication, mostly because they were not done at the time! This repo will soon be updated to use new approaches.

## Quickstart Guide

---

This quickstart is aimed to help you create your very own question to equation translator! Our configuration shown here is the model which performed the best among our tests. The translations are simple, so don't expect this to finish your calculus homework.

#### Step 1

---

For consistency, we provide a ready-to-use Docker image in this repo. To download all necessary packages and a version of TensorFlow that uses GPU configurations, follow this step. If you do not have Docker installed, you're going to want to install that before proceeding here.

Build the Docker image by:

```
chmod a+x build-docker && ./build-docker.sh
```

OR

```
sh build-docker.sh
```

This command builds the Docker image that we can train and test within. It includes the libraries necessary for running all forms of pretraining. Use the following command if you wish to open a command prompt within the image.

```
./bash.sh
```

#### Step 2 (Optional if using the build-docker.sh script)

---

To clean and create the data for your translations use the following:

```
./make-data.sh
```

Data collected is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Common Core](https://cogcomp.org/page/resource_view/98), [Illinois](https://cogcomp.org/page/resource_view/98), and [MaWPS](http://lang.ee.washington.edu/MAWPS/). For more details on these sets, please refer to their authors.

All MWPs in the testing and training data are similar to the following:

```
There are 37 short bushes and 30 tall trees currently in the park. Park workers will plant 20 short bushes today. How many short bushes will the park have when the workers are finished?
57
X = 37 + 20
```

Numerous binary files containing training and testing data will be created and housed in the data folder. Depending on your configuration, you will only use some of these files.

#### Step 3

---

Our approach has tested various vanilla Transformer models. Our most successful model is speedy and uses two Transformer layers.

The trainable model code lives in the translator.py file. This model is similar to the Transformer example discussed in the [TensorFlow online tutorial](https://www.tensorflow.org/tutorials/text/transformer).

Everything in the translator.py file is controllable through configuration files. To use this model create a config YAML file like this:

```
# A minimal configuration example

dataset: train_all_postfix.p
duplication: False
test: postfix
# Model specification
model: False
layers: 2
heads: 8
d_model: 256
dff: 1024
lr: scheduled
dropout: 0.1
# Training
epochs: 300
batch: 128
# Options: "dolphin", "imdb", "wikipedia_en_DUMPNUM"
pretrain: False
# Adam optimizer params
beta_1: 0.95
beta_2: 0.99
# Data pre-processing
pos: False
pos_words: False
remove_stopwords: False
as_lemmas: True
reorder: False
tagging: True
# Other behavior
input: False
seed: 420365
save: True
```

There is already a created configuration present in this repo for ease of use.

For pretraining you can set the _pretrain_ setting to `imdb`, `dolphin`, `wikipedia_en_DUMPNUM` (where DUMPNUM is the dump date from [here](https://dumps.wikimedia.org/backup-index.html)), or false.

Testing can occur on specific sets if you use the .p filename for the _test_ field. Using settings like shown in the above example tests all of the prefix test sets.

#### Step 4

---

From this point, you can run the following command in the container.

```
python translator.py config.yaml
```

Alternatively, you can run the _trial.sh_ script, which iterates over all of the config files found in the root directory and completes all epoch iterations you specify.

A more user friendly approach that does not require you to use the bash script, is the _run.sh_ script. This executes the _trial.sh_ script inside your image. From your host terminal use the following command:

```
./run.sh
```

This script is set to use only 1 GPU device if you are on a CUDA system. There is a comment within the script that shows how you can use a GPU on CUDA <10.1 environments and also how you can customize the GPU device list you want available.

#### Step 5

---

After training, the program saves your model, and your configuration file is updated to refer to this model in future training/testing. You can train from the checkpoint generated by repeating step 4.

#### Tips

---

- Once trained, setting _epochs_ to 0 skips training if you wish to retest your model. This mode also enables the command line input testing when _input_ is set to True. The translator works well when fed data from the training sets, but highly unique user input will most likely not result in the best output equations.

---

We hope that your interest in math word problems has increased, and encourage any suggestions for future work or bug fixes.

Happy solving!

#### Cite The Paper

```
@inproceedings{griffith2019solving,
  title     = {Solving Arithmetic Word Problems Automatically Using Transformer and Unambiguous Representations},
  author    = {Griffith, Kaden and Kalita, Jugal},
  booktitle = {Proceedings of the 2019 International Conference on Computational Science and Computational Intelligence (CSCI'19)},
  pages     = {526--532},
  year      = {2019}
}
```
