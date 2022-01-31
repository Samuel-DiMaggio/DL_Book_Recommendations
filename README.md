# Simple Deep Learning Book Recommender 
## Contents

### Books File

Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site.

### Ratings File

Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

These files were found: https://www.kaggle.com/arashnic/book-recommendation-dataset

## Import libraries and Fix Seeds
```python
import os
import math
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import random
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

SEED = 101
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

## Load and Clean the Data
```python
ratings = pd.read_csv('Ratings.csv', low_memory=False)
books = pd.read_csv('Books.csv', low_memory=False)
```
```python
ratings.head()
```
![image](https://user-images.githubusercontent.com/47721595/151726810-bbf6d41a-f22a-42b8-af66-007e1f535e29.png)

```python
ratings.isnull().sum()/len(ratings)*100
```
![image](https://user-images.githubusercontent.com/47721595/151726903-329ac147-830a-4792-9ad9-0e7a4f77538d.png)

```python
books.head()
```
![image](https://user-images.githubusercontent.com/47721595/151726984-5682cef3-3056-4f3f-9a2a-731c4c2c54a9.png)

```python
books.isnull().sum()/len(books)*100
```
![image](https://user-images.githubusercontent.com/47721595/151727225-f8c8b7d6-34e4-4097-9c5c-393ad14cf70a.png)

```python
books.dropna(inplace = True)
books.isnull().sum()
```
![image](https://user-images.githubusercontent.com/47721595/151727251-14d1752c-5e41-4dbd-97d2-3bdac64f0b0d.png)

```python
ratings.head()
```
![image](https://user-images.githubusercontent.com/47721595/151727273-d0787ade-3666-4668-9c14-779b43506372.png)

## Summarize the Data
```python
sns.countplot(x='Book-Rating',data = ratings)
```
![image](https://user-images.githubusercontent.com/47721595/151727340-b9f43993-0233-45d4-913a-afe4bdcf9d7a.png)

Since there is so much noise for the values of 0, which according to the information provided with the dataset indicates "implicit", which in this case is slightly ambiguous. In regards to this issue, below I will remove these rows.
```python
ratings = ratings[ratings["Book-Rating"] != 0]
```
```python
sns.countplot(x='Book-Rating',data = ratings)
```
![image](https://user-images.githubusercontent.com/47721595/151727406-23a813a4-6779-4de3-b442-3be994509c39.png)

```python
ratings.head()
```
![image](https://user-images.githubusercontent.com/47721595/151727536-57cc5099-59b3-46b2-8970-9f0a05051f0a.png)

```python
# Get the unique userIds
user_ids = ratings["User-ID"].unique().tolist()
# Map userIds to the corresponding indices
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# Inverse transform from index to userId
userencoded2user = {i: x for i, x in enumerate(user_ids)}
# Get the unique ISBN
books_ids = ratings["ISBN"].unique().tolist()
# Map ISBN to the corresponding index
books2books_encoded = {x: i for i, x in enumerate(books_ids)}
# Inverse transform from index to userId
books_encoded2books = {i: x for i, x in enumerate(books_ids)}
ratings["User-ID"] = ratings["User-ID"].map(user2user_encoded)
ratings["ISBN"] = ratings["ISBN"].map(books2books_encoded)

num_users = len(user2user_encoded)
num_books = len(books_encoded2books)
unique_rating = ratings["Book-Rating"].unique().tolist()
ratings["Book-Rating"] = ratings["Book-Rating"].values.astype(np.float32)


print(f"Number of users: {num_users}, Number of Books: {num_books}")
print(f"The unique rating values = {sorted(unique_rating)}")
```
![image](https://user-images.githubusercontent.com/47721595/151727576-a9c27a37-1554-4ba0-9d0a-e5011eb390d0.png)

## Split the Data into Training and Test Dataset

We take out the features and label. We convert the label to the range of [0, 1] to improve the network's performance.

```python
X = ratings[["User-ID", "ISBN"]].values
# Normalize the targets between 0 and 1. 
scaler = MinMaxScaler()
y = scaler.fit_transform(ratings["Book-Rating"].values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 101)
```
```python
# Set the dimensionaliy of the embedding space
EMBEDDING_SIZE = 50

# Create a subclass from Model class
class RecommenderNet(keras.Model):
    # construct the layers in the constructor
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        # Map userIds to  smaller vector space
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1),
        )
        # Generate a bias 
        self.user_bias = layers.Embedding(num_users, 1)
        # Map ISBN to  smaller vector space
        self.books_embedding = layers.Embedding(
            num_books,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1),
        )
        # Book bias
        self.books_bias = layers.Embedding(num_books, 1)
    # The forward pass to do computation
    def call(self, inputs):
        # Take out the first feature of userIds and do embedding
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        # Take out the 2nd feature of ISBN and do embedding
        books_vector = self.books_embedding(inputs[:, 1])
        books_bias = self.books_bias(inputs[:, 1])
        # 
        dot_user_books = tf.tensordot(user_vector, books_vector, 2)
        # Add all the components (including bias)
        x = dot_user_books + user_bias + books_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

# Create the model based on the sublass defined above
model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)
```
## Configure the Model

There are only two cases for the label. It either recommends the book or doesn't recommend it. It is a binary classification. The loss function should be binary cross-entropy. We also need to specify the optimizer.
```python
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001))
```
## Train the Model

We train the model using early stopping.
```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    min_delta=0.001, 
    mode='min')
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size= 128,
    epochs=100,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping])
 ```
 ![image](https://user-images.githubusercontent.com/47721595/151727721-de632c63-7dec-4cd7-b285-c4c26f6bec9d.png)

```python
model.summary()
```

![image](https://user-images.githubusercontent.com/47721595/151727764-4fd05d96-edbb-4e8f-9b4c-f85b89192ee4.png)

## Model Diagnostics

Let's check the in sample fit and out sample fit.

```python
train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
sns.lineplot(x='epoch', y ='loss', data =train_history)
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
plt.legend(labels=['train_loss', 'val_loss'])
```
![image](https://user-images.githubusercontent.com/47721595/151727798-ece64f44-e056-4bb3-8b8b-ba915e647fb9.png)

## Recommend the Books

The recommender system only can forecast it for the existing users in the dataset. Let's randomly select a user and recommend several books for him/her.

```python
# Randomly select a user from the rating dataset
from random import sample
user_id = str(random.sample(set(user_ids), 1))[1:-1]
print(user_id)
```
![image](https://user-images.githubusercontent.com/47721595/151727870-5736789d-6235-4d89-a7b5-e25edbcc9ce3.png)

```python
# Subset the books read and not read by this user
books_read_by_user = ratings[ratings["User-ID"] == user_id]
books_not_read = books[~books["ISBN"].isin(books_read_by_user.ISBN.values)]["ISBN"]
# Find the books not read that contains in the rating dataset
books_not_read = list(set(books_not_read).intersection(set(books2books_encoded.keys())))
# Map  the books to the indices
books_not_read = [[books2books_encoded.get(x)] for x in books_not_read]
# Map the user Id to the index
user_encoder = user2user_encoded.get(int(user_id))
# Generate a dataset with two columns
# The first column is the user_encoder that is the same for all rows
# The 2nd column is the books not read by this user.
user_books_array = np.hstack(([[user_encoder]] * len(books_not_read), books_not_read))
# Forecast the ratings(probablity of this user will read it )
ratings = model.predict(user_books_array).flatten()
# Find the indices of the top 10 ratings by using argsort
top_ratings_indices = ratings.argsort()[-10:][::-1]
# Find the recommened book Ids
recommended_books_ids = [books_encoded2books.get(books_not_read[x][0]) for x in top_ratings_indices]
```
Below, I only chose to show the Top 10 Recommendations, however if somebody wanted to show the "Top 10 Books rated by the User" I provided the code below as well. 

```python
# print("Showing recommendations for user: {}".format(user_id))
#print(f"Showing recommendations for user: {user_id}")
#print("====" * 20)
#print("Top 10 Books with high ratings from user")
#print("----" * 20)
#top_books_user = (books_read_by_user.sort_values(by="Book-Rating", ascending=False).head(10).ISBN.values)
#books_rows = books[books["ISBN"].isin(top_books_user)]
#for row in books_rows.itertuples():
#print(f"The title: {row.Title}; Author: {row.Author}; Published: {row.Year}")
print("----" * 20)
print("Top 10 Books recommendations")
print("----" * 20)
recommended_books = books[books["ISBN"].isin(recommended_books_ids)]
for row in recommended_books.itertuples():
    print(f"The title: {row.Title}; Author: {row.Author}; Published: {row.Year}")
```
![image](https://user-images.githubusercontent.com/47721595/151728025-d9d6ef74-2b55-4a5d-b72d-7d6f1557f8fd.png)

