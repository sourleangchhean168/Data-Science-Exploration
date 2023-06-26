# Select the textual features for clustering
textual_features = data[['title']].astype(str)

# Perform feature extraction using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
textual_features = vectorizer.fit_transform(textual_features['title'])

# Perform k-means clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(textual_features)

# Assign cluster labels to each book
data['cluster_label'] = kmeans.labels_

# Make a prediction for the next book to read
next_book_summary = "Voices from Chernobyl: The Oral History of a Nuclear Disaster"
next_book_summary_vectorized = vectorizer.transform([next_book_summary])
predicted_cluster = kmeans.predict(next_book_summary_vectorized)[0]
recommended_books = data[data['cluster_label'] == predicted_cluster]

# Display recommended books
print("Recommended Books:")
print(recommended_books[['title', 'genre']])

# Display predicted cluster for the next book
print("Predicted Cluster for the Next Book:")
print(predicted_cluster)


# Show the title of the next book
next_book_title = recommended_books.iloc[0]['title']
print("Next Book Title:")
print(next_book_title)
# for _, book in recommended_books.iterrows():
#     print(book['title'])



The code is performing text clustering using the k-means algorithm. Here is a detailed explanation of each step:

1. **Select the textual features for clustering**: The code selects the 'title' column from the 'data' dataframe and converts it to a string format. This column will be used as the input for the clustering algorithm.

2. **Perform feature extraction using TF-IDF vectorizer**: The code uses the TfidfVectorizer class from the scikit-learn library to convert the textual data into a matrix of TF-IDF features. The TF-IDF algorithm is used to convert a collection of raw documents to a matrix of TF-IDF features. It is a technique used for information retrieval to represent how important a specific word or phrase is to a given document[1][2][3].

3. **Perform k-means clustering**: The code uses the KMeans class from the scikit-learn library to perform k-means clustering on the TF-IDF features. The number of clusters is set to 5[1][3].

4. **Assign cluster labels to each book**: The code assigns a cluster label to each book in the 'data' dataframe based on the cluster they belong to[1].

5. **Make a prediction for the next book to read**: The code takes a summary of a book as input and predicts which cluster it belongs to using the k-means model. It then recommends books from the same cluster as the predicted cluster[1].

6. **Display recommended books**: The code displays the recommended books along with their genre.

7. **Display predicted cluster for the next book**: The code displays the predicted cluster for the next book.

8. **Show the title of the next book**: The code displays the title of the recommended book.

The TF-IDF algorithm is used to extract important keywords from a document to get a sense of what characterizes a document. It can be used for a wide range of tasks including text classification, clustering, topic modeling, search, keyword extraction, and more[4][2][5].

Citations:
[1] https://www.geeksforgeeks.org/sklearn-feature-extraction-with-tf-idf/
[2] https://www.datainsightonline.com/post/my-roadmap-into-preprocessing-data-an-example-of-a-text-classification-with-naive-bayes
[3] https://pythonprogramminglanguage.com/kmeans-text-clustering/
[4] https://www.freecodecamp.org/news/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667/
[5] https://kavita-ganesan.com/python-keyword-extraction/

By Perplexity at https://www.perplexity.ai/search/36b56406-e747-4283-b769-3c74b84ed928
