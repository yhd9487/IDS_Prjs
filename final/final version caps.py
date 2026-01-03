#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Due to complexity of code and variable names, we decided to leave it as group members part instead of sticking them together.


# In[ ]:


# Start of Chen Hua's code Question 1,4,5,extra credit


# In[ ]:


import random
import numpy as np

N_number = 15601674


# In[ ]:


#1


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Users\YHD\Documents\PycharmProjects\DS Prj\spotify52kData.csv'
spotify_data = pd.read_csv(file_path)

# Converting duration from milliseconds to minutes
spotify_data['duration_min'] = spotify_data['duration'] / 60000

# Calculating the correlation between song duration and popularity
correlation = spotify_data['duration_min'].corr(spotify_data['popularity'])
print(f"The correlation between song duration and popularity is: {correlation}.There is no obvious strong relationship between song duration and popularity. The duration of most songs is concentrated in a short range, and the popularity distribution is relatively wide. Although the correlation coefficient we calculated before shows a slight negative correlation, it can be seen more clearly from this figure that this relationship is not obvious Other factors may have a bigger impact on a song's popularity")

# visualization
plt.figure(figsize=(12, 6))
plt.hexbin(spotify_data['duration_min'], spotify_data['popularity'], gridsize=50, cmap='viridis')
plt.colorbar(label='Number of Songs')
plt.title("Hexbin plot of Song Duration (in minutes) vs Popularity")
plt.xlabel("Duration (minutes)")
plt.ylabel("Popularity")
plt.grid(True)
plt.show()


# In[ ]:


spotify_data.head()


# In[ ]:


#4


# In[ ]:


# Check missing values and data types
missing_values = spotify_data.isnull().sum()
data_types = spotify_data.dtypes

# Select relevant features for the analysis
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']

# Extracting the relevant features
spotify_data_selected = spotify_data[features]

missing_values_selected = missing_values[features]
data_types_selected = spotify_data_selected.dtypes

missing_values_selected, data_types_selected


# In[ ]:


# Correlation matrix
correlation_matrix = spotify_data_selected.corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Features and Popularity")
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Selecting relevant features and the target variable
X = spotify_data[features[:-1]]
y = spotify_data['popularity']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= N_number)

# Creating and training the Random Forest model
rf_model = RandomForestRegressor(n_estimators = 100, random_state = N_number)
rf_model.fit(X_train, y_train)

# Predicting on the test set
y_pred = rf_model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Feature Importance
feature_importance = rf_model.feature_importances_

# Creating a DataFrame for feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

mse, r2, feature_importance_df


# In[ ]:


from sklearn.linear_model import LinearRegression
# Fitting the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating performance metrics
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Getting the coefficients of the features
feature_coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

r2, adjusted_r2, rmse, feature_coefficients.sort_values(by='Coefficient', ascending=False)


# In[ ]:


features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'popularity'

r2_scores = {}
rmse_scores = {}

for feature in features:

    X = spotify_data[[feature]]
    y = spotify_data[target]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting and evaluating the model
    y_pred = model.predict(X_test)
    r2_scores[feature] = r2_score(y_test, y_pred)
    rmse_scores[feature] = np.sqrt(mean_squared_error(y_test, y_pred))

# Displaying the R² and RMSE scores for each feature
r2_scores, rmse_scores


# In[ ]:


#5


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


combined_features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']


X = spotify_data[combined_features]
y = spotify_data['popularity']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=N_number)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predicting and evaluating the linear regression model
lr_y_pred = lr_model.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_y_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))

# Create and fit the ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Predicting and evaluating the ridge regression model
ridge_y_pred = ridge_model.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, ridge_y_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_y_pred))

lr_r2, lr_rmse, ridge_r2, ridge_rmse

model_names = ['Linear Regression', 'Ridge Regression']
r2_values = [lr_r2, ridge_r2]
rmse_values = [lr_rmse, ridge_rmse]

results_df = pd.DataFrame({
    'Model Type': model_names,
    'R² Value': r2_values,
    'RMSE Value': rmse_values
})


results_df


# In[ ]:


#extra credit: The five most popular singers included in 
#the dataset and trying to analyze how their musicality is 
#similar to each other


# In[ ]:


# Grouping by artist and calculating the mean popularity
artist_popularity = spotify_data.groupby('artists')['popularity'].mean().sort_values(ascending=False)

# Getting the top 5 artists with the highest average popularity
top_5_artists = artist_popularity.head(5).index.tolist()

# Extracting the rows corresponding to these top 5 artists
top_artists_data = spotify_data[spotify_data['artists'].isin(top_5_artists)]

# Analyzing the features of these top artists
top_artists_features = top_artists_data.groupby('artists')[features].mean()

top_5_artists, top_artists_features


# In[ ]:


#Danceability is generally high, which may be related to their popularity.
#Energy varies from artist to artist, but generally tends to be in the medium to high range.
#Loudness also varies from artist to artist, but generally tends to be higher.


# In[ ]:


#Perform PCA to identify the three features most strongly correlated with danceability and then build a 
#model to predict danceability based on these features.


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Selecting audio features except danceability
audio_features = ['duration', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence',
                  'tempo', 'time_signature', 'key', 'mode', 'explicit']

X_audio = spotify_data[audio_features]

# Standardizing the features
scaler = StandardScaler()
X_audio_scaled = scaler.fit_transform(X_audio)

# Applying PCA
pca = PCA()
X_audio_pca = pca.fit_transform(X_audio_scaled)

# Analyzing the PCA components' relationship with Danceability
# We need to correlate each PCA component with Danceability
danceability = spotify_data['danceability']
pca_correlation_with_danceability = np.abs([np.corrcoef(X_audio_pca[:, i], danceability)[0, 1] for i in range(len(audio_features))])

# Sorting the components by their correlation with Danceability and selecting top 3
top_3_components_indices = np.argsort(pca_correlation_with_danceability)[-3:]
top_3_components_indices, pca_correlation_with_danceability[top_3_components_indices]



# In[ ]:


# Selecting the top 3 principal components
X_top_3_pca = X_audio_pca[:, top_3_components_indices]

# Splitting the dataset into training and testing sets
X_train_pca, X_test_pca, y_train_danceability, y_test_danceability = train_test_split(
    X_top_3_pca, danceability, test_size=0.2, random_state=N_number)

# Training a linear regression model using the top 3 principal components
model_pca_danceability = LinearRegression()
model_pca_danceability.fit(X_train_pca, y_train_danceability)

# Predicting danceability
y_pred_danceability_pca = model_pca_danceability.predict(X_test_pca)

# Evaluating the model
r2_pca_danceability = r2_score(y_test_danceability, y_pred_danceability_pca)
rmse_pca_danceability = np.sqrt(mean_squared_error(y_test_danceability, y_pred_danceability_pca))

r2_pca_danceability, rmse_pca_danceability


# In[ ]:


# End of Chen Hua's part, Start of Haoda Yu's Part. Question 2, 6, 8


# In[ ]:


# random seed
NNumber = 11375906


# In[ ]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

file_path = r'C:\Users\YHD\Documents\PycharmProjects\DS Prj\spotify52kData.csv'
df = pd.read_csv(file_path)


# In[ ]:


num_rows = df.shape[0]
print("The total number of rows in the dataset is：", num_rows)

has_nan = df.isna().any().any()
print("Whether the entire data set contains NaN values:", has_nan)


# In[ ]:


#--------------------------------------Start of Question 2------------------------------------------


# In[ ]:


true_false_counts = df.iloc[:, 6].value_counts()
print(true_false_counts)


# In[ ]:


# Divide songs into two groups: E and non-E
explicit_songs = df[df['explicit'] == True]['popularity']
non_explicit_songs = df[df['explicit'] == False]['popularity']


# In[ ]:


from scipy.stats import levene

# Levene test
statistic, p_value = levene(explicit_songs, non_explicit_songs)

print("Levene test result:")
print(f"Test Statistic: {statistic}")
print(f"p-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("The variances of the two sets of data are significantly different")
else:
    print("The variances of the two sets of data are similar")


# In[ ]:


from scipy.stats import mannwhitneyu


u_test_results = []

np.random.seed(NNumber)
random_seeds = np.random.randint(0, len(explicit_songs) , 1000)

# Perform 1000 down sampling and U tests
for seed in random_seeds:
    # Set a random seed for this iteration
    np.random.seed(seed)

    # Randomly sample from E songs, the same number as non-E songs
    down_sampled_non_explicit = non_explicit_songs.sample(n=len(explicit_songs), replace=False)

    # Perform Mann-Whitney U test
    u_stat, p_val = mannwhitneyu(explicit_songs, down_sampled_non_explicit)

    # Save U-statistics and p-values
    u_test_results.append((u_stat, p_val))

# Calculate the average U statistic and average p value of 1000 tests
average_u_stat = np.mean([result[0] for result in u_test_results])
average_p_val = np.mean([result[1] for result in u_test_results])

print(f"Average U statistic: {average_u_stat}")
print(f"Average p-value: {average_p_val}")


# In[ ]:


#--------------------------------------End of Q2, Start of Q6---------------------------------------


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
# Standardized data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# PCA
pca = PCA(n_components=0.95)
principal_components = pca.fit_transform(scaled_features)

# K-means clustering
kmeans = KMeans(n_clusters=20)
clusters = kmeans.fit_predict(principal_components)

# Add clustering results to original data frame
df['cluster'] = clusters


# In[ ]:


pca = PCA().fit(scaled_features)

# Draw cumulative variance curve
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')
plt.grid(True)
plt.show()


# In[ ]:


# There are 3 meaningful principal components


# In[ ]:


# PCA
pca = PCA(n_components=0.95)
principal_components = pca.fit_transform(scaled_features)

# Get the proportion of explained variance
explained_var_ratio = pca.explained_variance_ratio_

# Calculate the cumulative variance of each principal component
cumulative_var_ratio = explained_var_ratio.cumsum()

# Print the proportion of explained variance and cumulative variance of each principal component
for idx, (var_ratio, cum_var_ratio) in enumerate(zip(explained_var_ratio, cumulative_var_ratio)):
    print(f"Principal Component {idx+1}:")
    print(f"  - Explained Variance Ratio: {var_ratio:.4f}")
    print(f"  - Cumulative Variance Ratio: {cum_var_ratio:.4f}\n")


print(f"Total variance explained by the selected components: {cumulative_var_ratio[-1]:.4f}")


# In[ ]:


eigenvalues = pca.explained_variance_

for idx, eigenvalue in enumerate(eigenvalues):
    print(f"Eigenvalue of Principal Component {idx+1}: {eigenvalue:.4f}")


# In[ ]:


# There are 3 of these principal components account for 57.36% of the variance.


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Try different number of clusters from 2 to 20
silhouette_scores = []
for n_clusters in range(2, 21):
    kmeans = KMeans(n_clusters=n_clusters, random_state=NNumber)
    clusters = kmeans.fit_predict(principal_components)

    # Calculate silhouette coefficient
    score = silhouette_score(principal_components, clusters)
    silhouette_scores.append((n_clusters, score))

# Find the number of clusters with the highest silhouette coefficient
best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
best_score = max(silhouette_scores, key=lambda x: x[1])[1]

print(f"Best number of clusters: {best_n_clusters}")
print(f"Best silhouette score: {best_score}")
# Takes ~6 min


# In[ ]:


# cant reasonably correspond to the genre labels in column 20 of the data 


# In[ ]:


# -----------------------------------End of Q6 Start of Q8------------------------------------------


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]  # 特征数据
y = df['track_genre']

# Feature scaling and standardization using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Partition the data set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=NNumber)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=NNumber)

# Training model
clf.fit(X_train, y_train)

# Predict test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[ ]:


# 10 times 30.41%, 100 times 35.78%, 1000 times 36.50%
# 100 times has the best balance between time and effect, and there will not be much improvement when the number is increased.


# In[ ]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[features]
y = df['track_genre']

# 将流派标签转换为整数
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=NNumber)

# 创建XGBoost分类器
model = xgb.XGBClassifier(objective='multi:softmax', num_class=52, random_state=NNumber)

# 训练模型
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


# End of Haoda Yu's code part, Start of Haomin Yu’s code part. Question 3, 7, 9, 10


# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np

np.random.seed(10572260)


# In[ ]:


df = pd.read_csv(r'C:\Users\YHD\Documents\PycharmProjects\DS Prj\spotify52kData.csv')


# In[ ]:


major = df[df['mode'] == 1]
minor = df[df['mode'] == 0]


# In[ ]:


major['popularity'].mean(), minor['popularity'].mean()


# In[ ]:


from scipy import stats

def is_gaussian(data):
    stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    # print(p_value)
    alpha = 0.005
    if p_value > alpha:
        return True
    else:
        return False


# In[ ]:


is_gaussian(major['popularity']), is_gaussian(minor['popularity'])


# In[ ]:


major['popularity'].hist()
minor['popularity'].hist()
plt.legend(['Major', 'Minor'])
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.show()


# In[ ]:


u_stat, p_value = stats.mannwhitneyu(minor['popularity'], major['popularity'], alternative='greater')

# Print the results
print("U-statistic:", u_stat)
print("P-value:", p_value)
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis, meaning that the popularity of minor songs is greater than the popularity of major songs")
else:
    print("Fail to reject the null hypothesis, meaning that the popularity of minor songs is not greater than the popularity of major songs")


# In[ ]:


valence = df['valence']  # X
mode = df['mode']  # y


# In[ ]:


# logistic regression and support vector machine
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(valence, mode, test_size=0.2, random_state=10572260)
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)


# In[ ]:


# scatter plot
plt.scatter(x_train, y_train)
plt.xlabel('valence')
plt.ylabel('mode')
plt.show()


# In[ ]:


logreg_model = LogisticRegression(random_state=10572260)
logreg_model.fit(x_train, y_train)
y_pred_logreg = logreg_model.predict(x_test)
accuracy_score(y_test, y_pred_logreg)


# In[ ]:


svc_model = SVC(random_state=10572260)
svc_model.fit(x_train, y_train)
y_pred_svc = svc_model.predict(x_test)
accuracy_score(y_test, y_pred_svc)


# In[ ]:


y_pred_svc.mean(), y_pred_logreg.mean()


# In[ ]:


ratings = pd.read_csv(r'C:\Users\YHD\Documents\PycharmProjects\DS Prj\starRatings.csv', header=None)
avg_ratings = ratings.mean(axis=0)
popularity = df.iloc[:5000]['popularity']
popularity.shape, avg_ratings.shape


# In[ ]:


# scatter plot
plt.scatter(avg_ratings, popularity)
plt.xlabel('avg_ratings')
plt.ylabel('popularity')
plt.show()


# In[ ]:


avg_ratings.corr(popularity)


# In[ ]:


# 9)
greatest_hits = df.iloc[(~ratings.isna()).sum(axis=0).sort_values(ascending=False)[:10].index]
display(greatest_hits)


# In[ ]:


import numpy as np
from tqdm import tqdm

distance_matrix = np.zeros((10000,10000))
for i, j in tqdm([(i,j) for i in range(10000) for j in range(10000)]):
        if j < i:
            distance_matrix[i][j] = distance_matrix[j][i]
        else:
            person1 = ratings.iloc[i]
            person2 = ratings.iloc[j]
            mask = ~ratings.iloc[i].isna() & ~ratings.iloc[j].isna()
            diff = person1[mask] - person2[mask]
            distance_matrix[i][j] = diff.abs().mean()
np.save('distance_matrix.npy', distance_matrix)

distance_matrix = np.load('distance_matrix.npy')


# In[ ]:


avg_rec_ratings = []
avg_gh_ratings = []
intersects_with_greatest_hits = []
for i in tqdm(range(10000)):
    top_10_similar_users = (-distance_matrix[i]).argsort()[-11:][::-1]
    top_10_similar_users = top_10_similar_users[top_10_similar_users != i][:10]
    # index = ratings.iloc[top_10_similar_users].mean(axis=0)[ratings.iloc[i].isna()].sort_values(ascending=False)[:10].index
    index = ratings.iloc[top_10_similar_users].mean(axis=0).sort_values(ascending=False)[:10].index
    songs = df.loc[index]  # 10 songs to recommand
    user_ratings = ratings.iloc[i]  # ratings of user i
    avg_rec_ratings.append(user_ratings.loc[songs.index].mean())  # average ratings of recommanded songs
    intersects_with_greatest_hits.append(len(set(greatest_hits.index).intersection(set(songs.index))))
    avg_gh_ratings.append(user_ratings.loc[greatest_hits.index].mean())  # average ratings of greatest hits
print('mean of average ratings of recommanded songs:', np.nanmean(np.array(avg_rec_ratings)))
print('mean of average ratings of greatest hits:', np.nanmean(np.array(avg_gh_ratings)))
print('mean of number of intersections with greatest hits:', np.nanmean(np.array(intersects_with_greatest_hits)))

