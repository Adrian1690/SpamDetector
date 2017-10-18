documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []

for i in documents:
    lower_case_documents.append(i.lower())

#print(lower_case_documents)

sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))

#print(sans_punctuation_documents)

preprocessed_documents = []

for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))

print(preprocessed_documents)

frecuency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frecuency_counts = Counter(i)
    frecuency_list.append(frecuency_counts)

pprint.pprint(frecuency_list)

