from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

df = pd.read_table('smsspamcollection/SMSSpamCollection',
                    sep     = '\t',
                    header  = None,
                    names   = ['label', 'sms_message'])

df['label'] = df.label.map({'ham': 0, 'spam' : 1})

#print rows and columns
#print(df.shape)

#output printing out first 5 columns
#print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
                                        df['sms_message'],
                                        df['label'],
                                        random_state = 1)

print('Number of rows in the total set : {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

print(testing_data)