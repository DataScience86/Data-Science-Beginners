
# coding: utf-8

# # Building Recommender System

# In this article, we will be covering solid essentials of building recommender sytems with python. We will practice building all the different types of methods used in building the recommendation systems. First, we will discuss about the core concepts and ideas behind the recommender systems and then we will see how to build these systems using different python libraries. We will be covering the following approaches to recommender systems:-
# 
# 1. Popularity based recommender systems using pandas library
# 2. Correlation based recommender systems using pandas library
# 3. Classification based recommender systems [Machine Learning] using scikit learn library
# 4. Model based recommender system [Machine Learning] using scikit learn library
# 5. Content based recommender system [Machine Learning] scikit learn library
# 
# You will also learn how to evaluate each of these models. The data used in this article is provided here. Data set from UCI.

# # Why build recommender systems
# 
# Recommender systems are built in order to find out the items that a user is most likely to purchase or show interest in. Almost all the ecommerce websites these days use recommender systems to make product recommendation at their site. For example, Netflix uses it to make movie recommendations. If you use Amazon music then you must have seen the music recommendations which may have helped you in finding new music. Companies like Facebook, linkedIn, or other social media platforms also use recommnder systems to help you connect with new people. 

# # Different approaches to recommender systems
# 
# To build recommender systems we use the following techniques.
# 
# 1. **Collaborative filtering** - They are also called as crowdsource models. As it is based on what items most user prefer over certain others items. Collaborative filtering further have two approaches - 
# 
#     a. **User based** - These systems make recommendations based upon the similarity between the users. The similarity could be defined based upon infromations like age, martial status, net worth, geographical location, places you visit, number of children you have, etc. Let say a user who is 34, married and have two children was offered a credit card and he accepted that offer. Then the next customer who is also married and has children will be made the same offer.
#     
#     b. **Item based** - Item based systems are also called as item-to-item systems. They generate recommendations based on ratings given by user to the similar items. Think of this as the recommendations given to you by ecommerce websites stating people who purchased this also purchased A, B, and C items. For example, User X and Y have given high rating to a mobile phone and the charger from from brand XX. Then you purchased the same Mobile phone and gave high rating to the phone. Now based upon the similarity of preference between user X and Y and that you also liked the phone, system will make a recommendation for charger to you as well.
# 
# 2. **Popularity based Systems** - These sytems can be thought as the elementary form of collaborative filtering. The items are recommended based upon how popular those items are among other buyers or users. For example, a restaurant may be recommended to you because it has been rated high or has received most number of positive reviews by the users. So these systems require a historical data in order to make a suggestion. They are mostly, used by websites like Forbes, bloomber, or other news sites. Note - These systems cannot make personalized recommendations as they do not take into account the user infomration.
#    
# 3. **Content Based Systems** - These recommenders recommend items or products based upon the feature similarity of products. For example, if you have given high rate to the hotel facing beach then similar hotels will be recommened to you.
# 

# # Example Populartity Based Recommender System
# 
# In this example, we will see how to generate a recommendation using popularity based recommender technique. The data we are using here is downloaded from UCI Machine Learning Repository. You can get the data from the https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data. We used the chefmozaccepts.csv, rating_final.csv and geoplace.csv files.

# In[51]:


import pandas as pd
import numpy as np

# Reading the cuisine data - this file was created by merging the chefmozaccepts.csv and rating_final.csv on "placeID"

cuisine = pd.read_csv("C:/Users/mosha/OneDrive/Documents/datasciencebeginners/courses/Recommender Systems/data/cuisine.csv",
                     sep = ",")

cuisine.head()


# To generate recommendation based on counts, getting the count of ratings given to each place.

# In[57]:


# Using groupby to group the restaurants and getting the count by rating
count_by_rating = pd.DataFrame(cuisine.groupby(['placeID'])['rating'].count())

# Arranging the output in descending order and taking head to get the top 5 most popular restaurants
count_by_rating.sort_values('rating', ascending=False).head(5)


# From the above table of top 5 restaurants. The system will recommend the restaurant with id 135032 over the restaurant with id 135052. This is somewhat every naive let us see how to make recommendation based upon the type of cuisine. As part of example, we are showing top rated placeID for top 3 most popular cuisines.

# In[26]:


# getting top 3 most popular cuisines
count_by_cuisine = pd.DataFrame(cuisine.groupby(['Rcuisine'])['rating'].count())

# Arranging the output in descending order and taking head to get the top 10 most popular restaurants
count_by_cuisine_top10 = count_by_cuisine.sort_values('rating', ascending=False).head(10)


# In[62]:


# Creating a list of top 3 cuisine
top_3_cuisine = ["Mexican", "Bar", "Cafeteria"]

for i in top_3_cuisine:
    # Subsetting the data
    cuisines_subset = cuisine[cuisine["Rcuisine"] == i]
    # getting top 3 most popular cuisines
    count_by_restaurant = pd.DataFrame(cuisines_subset.groupby(['placeID'])['rating'].count())
    count_by_restaurant.reset_index(level = 0, inplace=True)
    
    # Arranging the output in descending order and taking head to get the top 10 most popular restaurants
    count_by_restaurant_top_5 = count_by_restaurant.sort_values('rating', ascending=False).head(5)
    print(f"\nMost popular restaurants for {i} food are: ")
    print(count_by_restaurant_top_5)


# # Example Correlation Based Recommender System
# In correlation based systems, recommendations are made based upon the similarity of the ratings/reviews given by users. So, for these systems, we use pearson correlation to suggest an item which is most similar to the item which user has already reviewed. In this sense, this technique takes user preference into account. If you want to refresh on Pearson correlation read here(https://datasciencebeginners.com/2018/09/30/05-statistics-and-branches-of-statistics-part-2/). Correlation based recommender systems are also called as item-based systems.
# 
# Now let us see how to create correlation based recommendation system in python

# In[67]:


# Importing libraries which we require
import numpy as np
import pandas as pd

# Reading the cuisine data - this file was created by merging the chefmozaccepts.csv and rating_final.csv on "placeID"
cuisine = pd.read_csv("C:/Users/mosha/OneDrive/Documents/datasciencebeginners/courses/Recommender Systems/data/cuisine.csv",
                     sep = ",")

places_geo = pd.read_csv("C:/Users/mosha/OneDrive/Documents/datasciencebeginners/courses/Recommender Systems/data/geoplaces.csv",
                     sep = ",", encoding= 'mbcs')

cuisine.head()


# In[68]:


# Checking the place_geo data
places_geo.head()


# In[70]:


# Subsetting data by required columns
places_geo =  places_geo[['placeID', 'name']]
places_geo.head()


# Lets check the rating these places are getting and see how popular these places are. Once we have this information we would check the summary statistics for cuisines dataset. 

# In[76]:


# Average rating by place
average_rating = pd.DataFrame(cuisine.groupby('placeID')['rating'].mean())
#average_rating.reset_index(level = 0, inplace=True)
average_rating.head()

# We will use count to get how popular these places are
average_rating['rating_count'] = pd.DataFrame(cuisine.groupby('placeID')['rating'].count())
average_rating.head()

# Generating descriptive statistics
average_rating.describe()


# Count indicates we have 95 unique places that have been reviewed with maximum value for rating count being 56. This essentially means that the most popular place in the dataset got 56 reviews. Let's now sort the dataset by using sort_values() method to get the most popular place in the dataset.

# In[77]:


average_rating.sort_values('rating_count', ascending=False).head()


# As restaurant with placeID 135032 is the one which has maximum count. For demo purposes we will see which places can be recommended to users based upon the pearsons correlation and rating given by him to other restaurants. 

# In[109]:


places_geo[places_geo['placeID'] == 135052] # restaurant name is La Cantina Restaurante

# Checking what all cuisines this place serves
cuisine[cuisine['placeID'] == 135052] 

# Most of the matrix is sparse as one person can only review few palces
places_geo_table = pd.pivot_table(data = cuisine, values='rating', index='userID', columns='placeID')
places_geo_table.head()

# Ratings given to el cafetaria restaurant by other users
la_rating = places_geo_table[135052]
la_rating[la_rating>=0]

# Creating the correlation table 
places_similar_to_la = places_geo_table.corrwith(la_rating)

corr_table_la = pd.DataFrame(places_similar_to_la, columns=['PearsonR'])
corr_table_la.dropna(inplace=True) # droping NA values from the sparse table
corr_table_la.head()

# Cominbing with the rating as rating given by other users is required
corr_table_la_summary = corr_table_la.join(average_rating['rating_count'])
corr_table_la_summary[corr_table_la_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10)


# Finally, what we get back here is the list of top 9 places which are similar to el cafeteria restaurant based upon their popularity and correltaion. We can also check what cuisine the top 9 restaurants serve by using the below code. I want to point out here is that you should also check how many people gave the rating to the restaurants. There are fare chances that only one person gave rating to both the restaurant. This can result into high pearson correlation value of one. But this is not meaningful. The places must have more than one review in order to represent meaningful correlation. 
# 
# Lastly, to evaluate how good the recommendation is you can check the type and unique number of items being served by the restaurants recommneded by recommender system. Note - This algorithm can turn out to be bit complicated as you may have to look for many rules to ensure that the final output is relevant and makes sense.

# # Example Classification based recommender systems
# 
# Classification based algorithm are powered by machine learning algorithms like navie bayes, logistic regression, etc. These models are capable of making personalized recommendations, because they take into account purchase history, user attributes, as well as other contextual data. In our example, we will use logistic regression model to build the recommendation system which will help a sales representative to a call on whether to reach a client with product recommendation or not. Basically, the model will predict whether the customer will buy the product or not. This demo is an example of user based recommendation system.

# In[129]:


# loading required libraries
import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression


# In[115]:


bank_data = pd.read_csv('C:/Users/mosha/OneDrive/Documents/datasciencebeginners/courses/Recommender Systems/data/bank.csv')
bank_data.head() # We have 42k observations and 37 variables.


# In[156]:


# Seperating independent and taregt variable
x_vars = bank_data.iloc[:, [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]].values
y_var = bank_data["y"]

# Building the logistic model
Logmod = LogisticRegression()
Logmod.fit(x_vars, y_var)

# Creating x_var data for new user
new_user = [[0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]]
y_pred = Logmod.predict(new_user)
y_pred # The customer will not buy the product if approached.


# In order to evaluate model, divide your data into test and train, then look for accuracy of prediction by generating classification table. 

# # Example Model based recommender systems
# These models use models built on user ratings to make the recommendations. This approach offers speed and scalability unlike classification based models where you have to go back and look into entire dataset to make final predictions.The algorithm here uses Singular Vector Decomposition(SVD) and Utility matrix ( User item matrix). 
# Utility Matix - These matrices contain data about ratings given by each user for each item. As all customers does not review each product these matrices are mostly sparse.
# 
# Singular Vector Decomposition - SVD uncovers the latent variables. A regular singular vector decomposition is a linear algebra method which divides the model matrix into three compressed matrices. For example, let say you have a matrix M. This M matrix will be decomposed into three matrices, U, S and V. The U matrix is left orthogonal matrix which hold non-redundant information about users. V is right orthogonal matrix and hold important information about items. Finally, in the middle, we have S diagonal matrix and contains the information about decomposition. 
# 
# In this example, we will figureout similar movies based upon user ratings for different movies.

# In[158]:


# Loadin required libraries
import sklearn
from sklearn.decomposition import TruncatedSVD

import numpy as np
import pandas as pd
import os


# In[164]:


os.chdir("C:/Users/mosha/OneDrive/Documents/datasciencebeginners/courses/Recommender Systems/data")


# In[167]:


# Reading the reviews data
movie_reviews = pd.read_csv("MovieReviews.csv")

# Checking which movies got highest rating counts
movie_reviews.groupby('item_id')['rating'].count().sort_values(ascending=False).head()

# Getting the movie name of item id 100
filter = movie_reviews['item_id'] == 100
movie_reviews[filter]['movie title'].unique()


# For demo, we will use model based approach to callout movies similar to Fargo. 

# In[171]:


# Building Utility Matrix 
utility_matrix = movie_reviews.pivot_table(values='rating', index='user_id', columns='movie title', fill_value=0)
utility_matrix.head()
utility_matrix_trans = utility_matrix.T
utility_matrix_trans.head()


# In[172]:


# Decompasing the matrix using SVD
svd = TruncatedSVD(n_components = 10, random_state = 86)
decomposed_matrix = svd.fit_transform(utility_matrix_trans)
decomposed_matrix.shape


# In[177]:


# Preparing a correlation matrix
corr_tab = np.corrcoef(decomposed_matrix)

# Substing the data for Fargo movie
names_of_movies = utility_matrix.columns
list_of_movies = list(names_of_movies)

fargo = list_of_movies.index('Fargo (1996)')

# Getting the fargo from corr_tab matrix
fargo_corr = corr_tab[fargo]

# Figuring 10 highly correlated movies - arranged in alphabetic order
list(names_of_movies[(fargo_corr >0.8 ) & (fargo_corr < 0.9)])[0:10]


# # Example Content based recommender systems
# In this final Machine learning based recommender system, we will be using an unsupervised algorithm known as KNN (K Nearest Neighbours). KNN algorithm first memorises the data and then tells us which two or more items are similar based upon mathematical calculation. Let's see how KNN can help us make recommendation for cars based upon the different features of the car. We will build model to get car which has mileage of 16, weighs 3.7, hourse power 180, and disp 250
# 

# In[183]:


mtcars.iloc[12:13,]


# In[194]:


import numpy as np
import pandas as pd

import sklearn
from sklearn.neighbors import NearestNeighbors

mtcars = pd.read_csv('mtcars.csv')
mtcars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

# Setting the features similar to Merc 450SL
t = [16, 250, 160, 3.7]
feature_matix = mtcars.iloc[:,[1, 3, 4, 6]].values

# Recommendation is made based upon 2 similar cars
knn = NearestNeighbors(n_neighbors=1).fit(feature_matix)

# printing the recommendation
print(knn.kneighbors([t]))

# Getting the names of the cars
mtcars.iloc[11:12,[0,1, 3, 4, 6]]


# We hope that you liked this article, if so please do rate us. So in this article, we learned how to build popularity based recommender systems, content based recommender sytems, and both types of collaborative filtering based systems using machine learning algorithms. We also looked at how to evaluate these models.
