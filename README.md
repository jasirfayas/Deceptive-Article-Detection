
# Deceptive Article Detection

This is a Machine Learning based project to predict whether an article is potentially deceptive or not.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to install the following tools in your machine:

```
Python3 or Anaconda
```
### Datasets
The datasets used for this project are from various sources. Below is some description about the data files used for this project.
#### Dataset #1
LIAR dataset which contains 3 files with .tsv format for test, train and validation.

LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, ["Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection](https://arxiv.org/abs/1705.00648), to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.

#### Dataset #2
Training data has been used from this [study](http://cse.iitkgp.ac.in/~abhijnan/papers/chakraborty_clickbait_asonam16.pdf) - data posted [here](https://github.com/bhargaviparanjape/clickbait/tree/master/dataset)

#### Dataset #3
["Getting Real about Fake News: Text & metadata from fake & biased news sources around the web"](https://www.kaggle.com/mrisdal/fake-news) by Meg Risdal
#### Dataset #4
[Fake or Real News ](https://github.com/GeorgeMcIntire/fake_real_news_dataset) tagged dataset created by Miguel Martinez Alvarez. 

### Download the datasets
 - [fake.csv](https://www.kaggle.com/mrisdal/fake-news)
 - [liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
 - [fr_news.zip](https://github.com/docketrun/Detecting-Fake-News-with-Scikit-Learn/blob/master/fake_or_real_news.csv)
 - [clickbait](https://github.com/bhargaviparanjape/clickbait/blob/master/dataset/clickbait_data.gz)
 - [non_clickbait_data](https://github.com/bhargaviparanjape/clickbait/blob/master/dataset/non_clickbait_data.gz)
 

### Installing

Links to install necessary softwares:

* *Python 3* - [Click here](https://www.python.org/downloads/)
* *Anaconda* - [Click here](https://www.anaconda.com/distribution/)

## Running the tests

You can run this project on your machine by doing the following steps:

### First Step

First of all, you have to run the **models.py** file, because that is where  we import datasets and train the classifiers.
To do that run the following command on your **python terminal** or in **Jupyter Lab**:
```
python models.py
```
After it has successfully run completely, there will be eight **.pkl** saved in the project folder.

Move those **.pkl** to the **static** folder.

### Second Step

After you have done the first step, now you are ready to do the next step.
Now you should run the **main.py**. To do that run the following command on the terminal:
```
python main.py
```
When this command has run completely, you will be shown a **web address** which is **127.0.0.1:8083**. Type that in your web browser. 
( Note: You should not close the python terminal until you want to stop the program.)

## Using the program
Now your browser will show a webpage which has fields to type in article headlines and body.

Type in the suspecting articles into the text fields and press **Submit**. 

**Results** will be shown below in the webpage.


## Modules used

 - sklearn
 - numpy
 - pandas
 - joblib
 - os
 - pdb
 - sys
 - flask

## Contributers

 - Aleena Thankachen
 - Jasir Fayas
 - Jeena Varghese
 - Joe Jose

## Acknowledgments

* kaggle.com
* towardsdatascience.com
* fakenewschallenge.com
