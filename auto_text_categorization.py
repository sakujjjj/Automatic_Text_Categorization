from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 訓練數據集
train_data = ["I love this sandwich.", "This is an amazing place!", "I feel very good about these beers.", "This is my best work.", "What an awesome view",
              "I do not like this restaurant", "I am tired of this stuff.", "I can't deal with this", "He is my sworn enemy!", "My boss is horrible."]
train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# 特徵提取
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data)

# 建立和訓練模型
classifier = MultinomialNB()
classifier.fit(train_vectors, train_labels)

# 預測新數據
test_data = ["The beer was good.", "I do not enjoy my job", "I ain't feeling dandy today.",
             "I feel amazing!", "Gary is a friend of mine.", "I can't believe I'm doing this."]

test_vectors = vectorizer.transform(test_data)
predictions = classifier.predict(test_vectors)

# 打印預測結果
for i, prediction in enumerate(predictions):
    print(test_data[i], " - ", "positive" if prediction == 1 else "negative")
