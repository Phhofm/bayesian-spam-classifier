# Binarized Multinomial Naive Bayes Spam Classifier

## Program

This program takes as arguments the paths to the training and testing folders [PathToSpamTrainingFolder, PathToHamTrainingFolder, PathToSpamTestingFolder, PathToHamTestingFolder]. The program trains on, then classifies the corresponding messages and outputs a evaluation, based on folder structure and classification.

## Background

This was a short spam-filter exercise of the lecture "Practical Artificial Intelligence" at UZH. The task was to implement the methods of the BayesianClassifier class in the course-provided framework. For training and testing, the **Enron dataset** is used, as found in pre-processed form here: <http://www.aueb.gr/users/ion/data/enron-spam/>
Each dataset contains user-specific messages and should only be used one at a time.
In pre-processed form, the messages will not have any header; **training and classification happens on subject and content of emails**.
Each data set needs to be split into training and testing messages by placing them into the corresponding folders.

## Approach

I chose the **Binarized Multinomial Naive Bayes**, because according to the paper _Spam filtering with Naive Bayes - Which Naive Bayes? - Metsis, Androutsopoulos, Paliouras, CEAS 2006 - Third Conference on Email and Anti-Spam, 2006, Mountain View, California USA_ this approach gave the best overall results.
This program extracts all words from all messages in the training data and then builds a **vector of the most used words** _# of words used is declared by the user, see the "NUMBEROFTOKENS" variable._
Afterwards it **builds a binarized vector for every single message in the training data, corresponding to the tokens in the declared vector**.
It then uses the Binarized Multinomial Naive Bayes to classify each message in the testing folders.
For more information, have a look into the src/BayesianClassifier.java file.

## Suggested Improvements

We could implement **preprocessing**: **Stop Words removal** if it improves correct classification. _depending on the dataset. If a certain stop word only appears in one Category (Ham or Spam), training and classifying on that token could improve correct classification._ Implementing the **Porter Stemmer Algorithm** to cut words to a specified base form. Or even using **lemmatization instead of stemming**.

Maybe rather than building a vector on the most used terms, we could build a vector that uses the **terms with the highest information gain**.

By using a **inverted index**, instead of building a full-term vector per message, where we explicitly record the absence of a term (mostly stored zeroes), we could build a vector per term, storing the messageID's of the messages containing
that term, thereby not explicitly storing the absence of a term.

We could make use of **more optimal datasctructures**.
