# IELTSWritingPrediction

## Keywords
* **NLP**
* **Text Classification**
* **RNN**
* **LSTM**
* **DistilBERT**
* **Transfer Learning**
* **IELTS**
* **BERT**

## 1.0 Problem Statement
  - Based on the essay, what is the IELTS Writing score?

## 2.0 Dataset
The data is collected from Kaggle (link: https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset). There are ~1400 datapoints with various scores. There are 6 columns in the .csv file. The columns are:

* Task_Type: This column categorizes the essays into their respective IELTS writing tasks, distinguishing between "Task 1" and "Task 2."

* Question: The "Question" column contains the specific writing prompts or questions assigned to the candidates for each essay.

* Essay: The heart of the dataset, the "Essay" column, contains the actual written responses submitted by the IELTS candidates.

* Examiner_Comment: This column includes comments and feedback provided by the examiners who evaluated the essays.

* Task_Response, Coherence_Cohesion, Lexical_Resource, Range_Accuracy columns includes the respective scores based on the evaluation criteria. These scores will be added to the dataset in the subsequent version.

* Overall: The "Overall" column provides the final scores assigned to each essay.

## 3.0 Challenge
With a small dataset (~1400 data points), it is hard to create a model that generalizes well on unseen data as there are ~100 data points for each category.

## 4.0 Methods
### 4.1 Data Preprocessing
The data is imbalanced in for each score range which add a layer of difficulty when training the data. It was originally splitted into scores of 4.5, 5.0, 5.5, 6.0,..., 9.0. However, it was visible that the model could not generalize even on the training set. Thus, a more extreme approach was taken where the data was splitted into 2 classes: <6.5 and >=6.5. The reason behind the specific 6.5 score is that Academic Institutions mostly look for international students who are able to obtain scores a minimal of 6.0 in writing and 6.5 average in all categories. As the data 

![image](https://github.com/mart1428/IELTSWritingPrediction/assets/60026413/ead64a22-e736-4e43-9f24-16e147d7d692)



### 4.2 Model
The initial model was created using a simple **RNN** layer with 2 fully connected layers as classifiers. The **RNN** model seems to find it hard as it is a multi-class model with only ~100 data points per class. Thus, the model overfitted with over ~80% accuracy in training set and ~60% accuracy in validation set. 

A **LSTM** model was then created as it is a more complex model and it can better understand the context of the essays. The **LSTM** model used 150 hidden size, 1 layer, 0.4 drop out. The model is also **bidirectional** as to capture and understand the context better.
A 6B **GloVe** embedding with 50 dimensions was also used to tokenize the data and obtain its vectors. The model was able to obtain a better Training and Validation loss but the accuracy was still ~65% in validation set.

A **Transfer Learning** approach was then used to extract features from the input data. A **DistilBERT** Transformer was then used as it was a smaller variation on **BERT**. In theory, the model should be able to capture the context and features from the essays even though the size of the dataset is small. The transformer was then accompanied by 4 fully connected layers of size 64. 
![image](https://github.com/mart1428/IELTSWritingPrediction/assets/60026413/76e06e65-00ff-4a7a-9e17-000194ff459c)

The model reached better losses and accuracies on training, validation and test set. With about 0.714 loss and 73.61% accuracy on test set, the model reached its best state without overfitting and underfitting the dataset. 

![image](https://github.com/mart1428/IELTSWritingPrediction/assets/60026413/0860811f-5c4b-4404-a48e-9284d522a562)

## 5.0 Next Steps

Here are some next steps that I would like to consider:
* Include Task Type and Question features in the model
* Obtain more data with better distribution between classes

