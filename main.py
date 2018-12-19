#!/usr/bin/env python
# coding: utf-8

# ## FizzBuzz Project

# In[23]:


import numpy as np                                          
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt


# ## Logic Based FizzBuzz Function [Software 1.0]

# In[24]:

# Function definition for implementation of fizzbuzz in Software 1.0.

# This function takes any integer number as n and returns a string output, 
# based on fizzbuzz logic.
def fizzbuzz(n):                                        
    
    # Logic Explanation
    # If n is divisble by both 3 and 5, the function returns 'FizzBuzz' as output.
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    # If n is divisible by 3 but not 5, the function returns 'Fizz' as output.
    elif n % 3 == 0:
        return 'Fizz'
    # If n is divisible by 5 but not 3, the function returns 'Buzz' as output. 
    elif n % 5 == 0:
        return 'Buzz'
    # If n is not divisble by 5 or 3, the function returns 'Other' as output.
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format

# In[25]:

# This function definition creates .csv files using pandas.DataFrame 
def createInputCSV(start,end,filename):
    
    # Why list in Python?
    # Since we are using numbers in sequence or order from 1 to 1000, 
    # we choose list as our collection to store these numbers, which is ordered and changeable. 
    # Both list and tuple are ordered, but tuple is unchangeable. 
    # list is preferred over tuple because in future it will be easy to change 
    # our training or test data set if required.
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    # In Machine Learning we divide our dataset into training dataset and test dataset.
    # We train our model using training dataset, so that this model learns and then test it on test dataset.
    # We cannot just design a predictive system without any learning. 
    # We can compare this to our brain, which first learns from various organs such as eyes, ears etc. 
    # For example we did not just started to write when we are born. We learn from practice,
    # which is like training dataset and after that we are able to write anything based on this learning.
    # We can elaborate this by considering how one learns to speak and write a specific language. \
    # To create training data we use the fizzbuzz function from software 1.0 to calculate output for us.
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe?
    # Dataframes are multidimensional data structures that is present in pandas module/package. 
    # Pandas consists of many inbuilt functions/method to work with dataframes.
    # Here we are creating a dataset with two columns named input and label 
    # and assigning input data to the input column and ouput data to the label column. 
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    # to_csv() writes dataframes to a csv file. 
    # It also stores dataframe indices as the first column in .csv file.
    # To avoid writting indices we can use .to_csv(filename, index=False).
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[26]:


def processData(dataset):
    
    # Why do we have to process?
    # Since input data is in categorical, labels or numbers format (other than binary),
    # which is not understood by machine learning algorithms.
    # System only understands binary representation.
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    # encodeData(data) returns encoded binary values for each number.
    processedData  = encodeData(data)
    # encodeLabel(labels) returns encoded binary values for each label.
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[27]:

import numpy as np

# Here we are encoding data [i.e 1-100 in test dataset and 101 to 1000 in training dataset],
# into 10 bit binary numbers. 
def encodeData(data):
    
    # Encoded list
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        # We have max number as 1000 which requires 10 bits to represent in binary form.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    # np.array() converts tuple or list to an array.
    return np.array(processedData)


# In[28]:

# Here we are processing categorical labels into numerical forms,
# so that our machine learning algorithms can easily understand.
def encodeLabel(labels):
    
    # Encoded list
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    # np_utils.to_categorical() is used to convert an array of labelled data to one-hot vector.
    # Here 4 is the total number of classes
    # one-hot encoding method is used to handle categorical variables 
    # by transforming them into numerical value.
    return np_utils.to_categorical(np.array(processedLabel),4)


# In[29]:


# Create datafiles
# Calling createInputCSV(start,end,filename) which is defined above,
# to create two files named training.csv (training data) and testing.csv (testing data)
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# In[30]:


# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData  = pd.read_csv('testing.csv')

# Process Dataset
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel   = processData(testingData)


# ## Tensorflow Model Definition

# In[31]:


# Defining Placeholder
# Placeholders are like variables which are assigned data at a later date.
# By creating placeholders, we only assign memory(optional) where data is stored later on.
# [None, 10] defines the shape of this placeholder as an array of size 10.
inputTensor  = tf.placeholder(tf.float32, [None, 10])
outputTensor = tf.placeholder(tf.float32, [None, 4])


# In[32]:

# Defining number of neurons in the hidden layer of neural netwok.
NUM_HIDDEN_NEURONS_LAYER_1 = 100
# It defines how quickly a model learns.
LEARNING_RATE = 0.08

# Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

# Initializing the input to hidden layer weights
input_hidden_weights  = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1])
# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])

# Computing values at the hidden layer
# relu to convert to linear data
hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))
# Computing values at the output layer
output_layer = tf.matmul(hidden_layer, hidden_output_weights)

# Defining Error Function
# It computes inaccuracy of predictions in classification problems.
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function
prediction = tf.argmax(output_layer, 1)


# ## Training the Model

# In[141]:

# Epochs is defined as the number of times training data is traversed through by the model.
NUM_OF_EPOCHS = 5000
# Total data is divided in sub samples of 32 or 64 or 128 etc.
BATCH_SIZE = 128

# Accuracy for each iteration of training data is stored in a list. 
training_accuracy = []

# TensorFlow session is an object where all operations are run.
with tf.Session() as sess:
    
    # Set Global Variables ?
    # We initialize all the global variables before training starts.
    tf.global_variables_initializer().run()
    
    # Iterating each EPOCHS.
    # tqdm uses smart algorithms to predict remaining time, skip unnecessary iteration displays and displays a progress bar.
    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
        
        #Shuffle the Training Dataset at each epoch
        # We shuffle data to avoid sequential data, that a model can learn and reduce training error.
        # Making data more random can make model to learn and converge more efficiently.
        p = np.random.permutation(range(len(processedTrainingData)))
        processedTrainingData  = processedTrainingData[p]
        processedTrainingLabel = processedTrainingLabel[p]
        
        # Start batch training 
        # we loop each batch and start training.
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
            # By running a session we evaluate tensor variables.
            # feed_dict initializes tensor variables.
            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], 
                                          outputTensor: processedTrainingLabel[start:end]})
        # Training accuracy for an epoch
        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                             outputTensor: processedTrainingLabel})))
    # Testing
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})


# In[142]:

df = pd.DataFrame()
df['acc'] = training_accuracy
df.plot(grid=True)
plt.show()

# In[143]:

# Function to decode labels back from numerical value.
def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# # Testing the Model [Software 2.0]

# In[144]:

# wrong = Predicted label not correct.
wrong   = 0
# right = Predicted label is correct.
right   = 0

# List to store all predicted labels.
predictedTestLabelList = []

# Checking how many matches occured between predicted and processed labels. 
for i,j in zip(processedTestingLabel,predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))
    
    if np.argmax(i) == j:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

testDataInput = testingData['input'].tolist()
testDataLabel = testingData['label'].tolist()

# Please input your UBID and personNumber 
testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "****")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "************")

predictedTestLabelList.insert(0, "")
predictedTestLabelList.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabelList

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')
