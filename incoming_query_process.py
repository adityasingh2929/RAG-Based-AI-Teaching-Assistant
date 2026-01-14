import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from read_chunks import create_embedding
import numpy as np
import joblib

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a question: ")
embedded_query = create_embedding([incoming_query])[0]   # just cause ye list ki list deta hai and ek hee element hota hai soo using '[0]' helps us just extract a simple list with embed values.
print("question query created\n\n")

# Find similarities of question_embedding with other embeddings.
similarities = cosine_similarity(np.vstack(df['embedding']),[embedded_query]).flatten()
# print(similarities)
top_results = 3
max_indx = similarities.argsort()[::-1][:top_results]    # this basically gives the similarities's vector embedding's indexes, that too...the top 3 in the sorted order.
# print(max_indx)

new_df = df.loc[max_indx]   # i.e load all the rows given in the max_indx list from the df dataframe and put them into the new_df dataframe.

print(new_df)