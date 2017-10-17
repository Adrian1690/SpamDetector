import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

df = pd.read_table('smsspamcollection/SMSSpamCollection',
                    sep     = '\t',
                    header  = None,
                    names   = ['label', 'sms_message'])

df['label'] = df.label.map({'ham': 0, 'spam' : 1})

#print rows and columns
print(df.shape)

#output printing out first 5 columns
print(df.head())