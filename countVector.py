from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

count_vector = CountVectorizer()

count_vector.fit(documents)

count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()

frecuency_matrix = pd.DataFrame(doc_array,
                                columns = count_vector.get_feature_names())
                                
print(frecuency_matrix)