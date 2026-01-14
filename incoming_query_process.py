import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model" : "bge-m3",
        "input" : text_list
    })

    embedding = r.json()['embeddings']
    return embedding

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a question: ")
embedded_query = create_embedding([incoming_query])[0]   # just cause ye list ki list deta hai and ek hee element hota hai soo using '[0]' helps us just extract a simple list with embed values.
print("question query created\n\n")

# Find similarities of question_embedding with other embeddings.
similarities = cosine_similarity(np.vstack(df['embedding']),[embedded_query]).flatten()
# print(similarities)
top_results = 5    # just soo we get top 5 results so that we give max. possible context to the LLMs.
max_indx = similarities.argsort()[::-1][:top_results]    # this basically gives the similarities's vector embedding's indexes, that too...the top 3 in the sorted order.
# print(max_indx)

new_df = df.loc[max_indx]   # i.e load all the rows given in the max_indx list from the df dataframe and put them into the new_df dataframe.

# Printing results:
for index, item in new_df.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"],item["end"])


# Using the model to answer based on the RAG Data.

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json={
        "model" : "deepseek-r1:1.5b",
        "prompt" : prompt,
        "stream" : False
    })

    response = r.json()
    return response

# PROMPT

prompt = f'''I'm teaching web development in my Sigma Web development course. Here are video chunks containing video title, video number, start time in second, end time in second, the text in between the given timestamps:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}  
------------------------------------
{incoming_query} 

User asked this question related to the video chunks, you have to answer in a human way (do not show your thinking and also dont mention the above format, its just for you) where and how much content is taught where (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated questions, tell him that you can only answer questions related to the course. Do not ask any question at the end, just answer the question. If you're asked about timestamps, give them in minutes format, not seconds, its less confusing that way.

'''

with open("prompt.txt","w") as f:
    f.write(prompt)


# Using the model to answer based on the RAG Data.

response = inference(prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)