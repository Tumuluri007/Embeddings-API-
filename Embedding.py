import numpy as np
from openai import OpenAI

client = OpenAI()
# All here it is happening is we are passing a string for our embedding vector and gives us back a list of Numbers.Remember is that vecotr of numbers is basically a point in Multi Dimensional Space and how things are similar between each other. 
prompt = input("Enter a string to create an embedding vector for: ")
response = client.embeddings.create(
	input = prompt,
	model = "text-embedding-ada-002")

print("\n")	
print(response)

print("\nLet's find the similarity score between 'potato' and 'rhubarb'.") # we can Find how similar one thing is similar to each other. 

response = client.embeddings.create(
	input=["potato", "rhubarb"], # we can passs a whole bumch of stuff like N number of things.
	model="text-embedding-ada-002")
	
potato = response.data[0].embedding
rhubarb = response.data[1].embedding

simScore = np.dot(potato, rhubarb) #dot product. of cosine Similarity 

print("\nScore is " + str(simScore) + "\n") # and print the dot product.

print("How about 'potato' and 'The starship Enterprise'") # Now we are seeing if 2 things are not similar to each other. 

response = client.embeddings.create(
	input=["potato", "The starship Enterprise"],
	model="text-embedding-ada-002")
	
potato = response.data[0].embedding
enterprise = response.data[1].embedding

simScore = np.dot(potato, enterprise)

print("\nScore is " + str(simScore) + "\n")

