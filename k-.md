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

The line `vectorizer = TfidfVectorizer()` creates an instance of the TfidfVectorizer class from the scikit-learn library. This class is used to convert a collection of raw documents to a matrix of TF-IDF features. The `TfidfVectorizer()` method takes several optional parameters that can be used to customize the vectorizer. By default, it uses the following settings:

- `lowercase=True`: Convert all characters to lowercase before tokenizing.
- `stop_words=None`: Remove stop words from the text. Stop words are common words that are usually removed from text because they do not carry much meaning, such as "the", "and", and "a".
- `token_pattern='(?u)\\b\\w\\w+\\b'`: Use a regular expression to tokenize the text. This pattern matches all words that are at least two characters long.
- `max_df=1.0`: Ignore terms that have a document frequency strictly higher than the given threshold.
- `min_df=1`: Ignore terms that have a document frequency strictly lower than the given threshold.
- `use_idf=True`: Enable inverse-document-frequency reweighting.
- `norm='l2'`: Apply L2 normalization to the output vectors.

The line `textual_features = vectorizer.fit_transform(textual_features['title'])` uses the `fit_transform()` method of the `TfidfVectorizer` class to convert the textual data into a matrix of TF-IDF features. The `fit_transform()` method fits the vectorizer to the data and transforms the data into a matrix of TF-IDF features. In this case, it takes the 'title' column of the 'data' dataframe and converts it to a matrix of TF-IDF features. The resulting matrix is assigned to the 'textual_features' variable.

Citations:
[1] https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/
[2] https://programminghistorian.org/en/lessons/analyzing-documents-with-tfidf
[3] https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html
[4] https://pythonprogramminglanguage.com/kmeans-text-clustering/
[5] https://www.geeksforgeeks.org/sklearn-feature-extraction-with-tf-idf/

The line `k = 5  # Number of clusters` sets the number of clusters to 5. This means that the k-means algorithm will group the data into 5 clusters. The number of clusters is a hyperparameter that needs to be set before running the algorithm. The optimal number of clusters depends on the data and the problem at hand. There are several methods to determine the optimal number of clusters, such as the elbow method, the silhouette method, and the gap statistic method[1][2].

The line `kmeans = KMeans(n_clusters=k, random_state=42)` creates an instance of the KMeans class from the scikit-learn library and sets the number of clusters to 5 using the 'n_clusters' parameter. The 'random_state' parameter is set to 42 to ensure reproducibility of the results. The KMeans class is used to perform k-means clustering on the TF-IDF features.

The line `kmeans.fit(textual_features)` fits the k-means model to the TF-IDF features. This means that the algorithm is trained on the data to learn the cluster centers and assign each data point to a cluster. The resulting model is stored in the 'kmeans' variable.

The k-means algorithm is an unsupervised learning algorithm that groups data points into k clusters based on their similarity. The algorithm works by iteratively assigning data points to the nearest cluster center and updating the cluster centers based on the new assignments. The algorithm stops when the assignments no longer change or a maximum number of iterations is reached[3][4].

Citations:
[1] https://stackabuse.com/k-means-clustering-with-scikit-learn/
[2] https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
[3] https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
[4] https://en.wikipedia.org/wiki/K-means_clustering


