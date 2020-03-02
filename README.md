![University of Colorado Colorado Springs](https://github.com/kadengriffith/MWP-Automatic-Solver/blob/master/data/util/UCCS.png)

# MWP Automatic Solver

---

This repo is the home to an automatic solver for math word problems using the 2017 Transformer model. This research was conducted at the University of Colorado Colorado Springs by Kaden Griffith and Jugal Kalita over the Summer in 2019. This research was a part of an undergraduate program through the National Science Foundation. This Transformer arrangement translates a word problem into a useable expression. We did not provide comprehensive code to solve the output, but you can use our _EquationConverter_ class to solve infix expressions via the sympy package. Our best model surpassed state-of-the-art, receiving an average of 86.7% accuracy among all test sets.

This version supports some features that were not used in the publication, mostly because they were not done at the time!

## Quickstart Guide

---

This quickstart is aimed to help you create your very own question to equation translator! Our configuration shown here is the model which performed the best among our tests. The translations are simple, so don't expect this to finish your calculus homework.

#### Step 1

---

For consistency, we provide a ready-to-use Docker image in this repo. To download all necessary packages and a version of TensorFlow that uses GPU configurations, follow this step. If you do not have Docker installed, you're going to want to install that before proceeding here.

Build the Docker image by:

```
chmod a+x build-docker && ./build-docker
```

OR

```
sh build-docker
```

This command builds the image, and start up a bash environment that we can train and test within. It includes the libraries necessary for running all forms of pretraining (Wikipedia inlcuded). Use the following command after you exit the container and wish to start it again.

```
./run
```

The script above is set to mount the working directory and uses GPU0 on your system. Please alter the script to utilize more GPUs if you want to speed up the training process (i.e., `-e NVIDIA*VISIBLE_DEVICES=0,1*` to use 2 GPUs, on newer versions of CUDA, you will need the `--gpus` flag. Edit the run script as necessary.).

#### Step 2

---

So that we don't require a large download to use this software, compilers and generators exist that need to be used before training your model. Run the following command to both generate 50000 problems using our problem generator and compile the training and test sets we used in our work. We did not end up using the 50000 generated problems in our publication, but they're fun to train with and allow for custom applications if you want to use this code for something else.

```
./make-data
```

Data collected is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Common Core](https://cogcomp.org/page/resource_view/98), [Illinois](https://cogcomp.org/page/resource_view/98), and [MaWPS](http://lang.ee.washington.edu/MAWPS/). For more details on these sets, please refer to their authors.

All MWPs in the testing and training data are similar to the following:

```
There are 37 short bushes and 30 tall trees currently in the park. Park workers will plant 20 short bushes today. How many short bushes will the park have when the workers are finished?
57
X = 37 + 20
```

#### Step 3

---

Our approach has tested various vanilla Transformer models. Our most successful model is speedy and uses two Transformer layers.

The trainable model code lives in the translator.py file. This model is similar to the Transformer example discussed in the [TensorFlow online tutorial](https://www.tensorflow.org/beta/tutorials/text/transformer).

Everything in the translator.py file is controllable through configuration files. To use this model create a config JSON file like this:

```
{
 "dataset": "train_all_prefix.p",
 "test": "prefix",
 "input": false,
 "pretrain": false,
 "seed": 6271996,
 "model": false,
 "layers": 2,
 "heads": 8,
 "d_model": 256,
 "dff": 1024,
 "lr": "scheduled",
 "dropout": 0.1,
 "epochs": 300,
 "batch": 128,
 "beta_1": 0.95,
 "beta_2": 0.99
}

// output to config.json
```

For pretraining you can set the _pretrain_ setting to `imdb`, `dolphin`, `wikipedia_en_DUMPNUM` (where DUMPNUM is the dump date from [here](https://dumps.wikimedia.org/backup-index.html)), or false.

Testing can occur on specific sets if you use the .p filename for the _test_ field. Using settings like shown in the above example tests all of the prefix test sets.

#### Step 4

---

From this point, you can run the following command in the container.

```
python translator.py config.json
```

Alternatively, you can run the _trial_ script, which iterates over all of the config files found in the root directory and completes all epoch iterations you specify.

#### Step 5

---

After training, the program saves your model, and your configuration file is updated to refer to this model. You can train from the checkpoint generated by repeating step 4.

#### Tips

---

- Once trained, setting _epochs_ to 0 skips training if you wish to retest your model. This mode also enables the command line input testing when _input_ is set to True. The translator works well when fed data from the training sets, but highly unique user input will most likely not result in the best output equations.

---

We hope that your interest in math word problems has increased, and encourage any suggestions for future work or bug fixes.

Happy solving!
