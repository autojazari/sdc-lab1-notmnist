# Assignment 1 of Udacity.com Self-Drive Course

Udacity.com course uses Jupyter Notebook, however I perfer to run all code from the terminal.  This guide will help you setup OS X and Ubuntu / Kali-Rolling with `Python 3.5`, `Matplotlib`, `TensorFlow` and `sklearn`.

## Description
The purpose of this assignment was to practice the following:

* `Linear Regression`
* `TensorFLow`
* `Stochastic Gradient Descent`
* `Learning Rate Decay`
* `Cross Validation`

Most of the code was provided but as a jupyter notebook.  The excercise uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) data set

## OS X
```
brew install python3
brew install pkg-config
brew link pkg-config
brew install pygtk freetype libpng
pip3 install -r requirements.txt
pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl

/usr/local/Cellar/python3/3.5.2_3/bin/python3.5 main.py
```
## Summary

* `cross validation` : 
  * batch size : ~0.1% - ca. 1700
* `epochs` : 50 iterations


## Example Plot

![alt tag](https://raw.githubusercontent.com/autojazari/sdc-lab1-notmnist/master/SDC-Assignment-1-Learn-Rate-Decay.png)

