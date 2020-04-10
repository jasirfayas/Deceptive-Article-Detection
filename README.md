# Deceptive Article Detection

This is a Machine Learning based project to predict whether an article is potentially deceptive or not.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites

You need to install the following tools in your machine:

```
Python3 or Anaconda
```

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

 - ALEENA THANKACHEN
 - JASIR FAYAS
 - JEENA VARGHESE
 - JOE JOSE

## Acknowledgments

* kaggle.com
* towardsdatascience.com
* fakenewschallenge.com
