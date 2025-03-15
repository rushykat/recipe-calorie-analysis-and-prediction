# Recipe Deep-Dive: Analyzing and Predicting Calorie Counts
*By: Rushyendra Katabathuni*

## Introduction

This project is the final project for the course DSC 80 at the University of California, San Diego. This project analyzes a dataset of recipes which
were scraped from food.com. This project seeks to first understand and answer the question: what types of recipes tend to be healthier and what nutritional
values correlate with healthier foods? For example, do healthy foods have more protein than unhealthy foods? Do they have less carbohydrates than unhealthy
foods? Then, this project will use models to predict the number of calories in recipes.

To begin, I had two datasets: recipes and ratings. The Recipes dataset contained information about recipes. These recipes were uploaded by users on food.com.
The Ratings dataset contains information on ratings that users left for a specific recipe.

***Recipes Schema***
(731927 rows × 5 columns)

| Column Name            | Description                          |
|------------------------|-------------------------------------:|
| name                   | (object) name of recipe              |
| id                     | (int64) unique recipe ID             |
| minutes                | (int64) number of minutes to prepare |
| contributor_id         | (int64) unique contributer ID        |
| submitted              | (object) date submitted              |
| tags                   | (object) relevant recipe tags        |
| nutrition              | (object) [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]|
| n_steps                | (int64) number of steps in the recipe|
| steps                  | (object) text for recipe steps       |
| description            | (object) description of recipe       |
| ingredients            | (object) text for recipe ingredients |
| n_ingredients          | (int64) number of ingredients        |

***Ratings Schema***
(83782 rows × 12 columns)

| Column Name            | Description                            |
|------------------------|---------------------------------------:|
| user_id                | (int64) unique user ID                 |
| recipe_id              | (int64) unique recipe ID               |
| date                   | (object) date rating/ review submitted |
| rating                 | (int64) rating from 1-5 submitted      |
| review                 | (object) review submitted              |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

I start the data cleaning process by left merging the recipes and ratings data. I want to left merge the ratings data onto recipes because I want to keep all recipes even
if they don't have a review for them. I merge the recipes data on the "id" column with the ratings data on the "user_id" column. Once I have my initial merged dataset, I 
want to ensure that values are properly represented. So first, I made sure to replace all occurences of "0" rating scores to "np.nan" rating scores. This is because a "0" 
rating doesn't make sense as a value, as it really just represents an absence of a rating. I wouldn't want to treat "0" ratings as valid ratings because they will skew the
average rating that I calculate for each recipe. Next, I want to calculate the average rating. I can do this by grouping the dataset by the "id", then extracting the
rating and calculating the mean for each group. Then, I want to merge this column with the main dataset. 

The next cleaning I want to do is converting the nutrition column from a list of values to individual values. The nutrition value for each row in a recipe contains
information about the nutritional value of that meal. This information is represented in the shape of a list, however it is actually stored as a string. So, I first 
convert each nutrition string in the column into a list. Then, I can put each value in the list into its own column, and convert that into a dataframe. 

The third cleaning step involves filtering out outliers. This dataset contains some significant outliers that skew the data and significantly impact visualizations. For ease,
I want to exclude these values from my data analysis. Logically, this makes sense, because a recipe with 45,000 calories is outside of regular consumption. The daily caloric
recommendation is 2000 calories, so I think it is valid to exclude these outliers for the sake of better visualizations and analyses. To filter out outliers, I defined a
method, "filter_outliers", which returns only those values within my lower and upper bounds. I then pass through every column in the nutrition dataframe, to remove outliers
for each column. Once I've filtered out the data to exclude outliers, I want to get the indices of the remaining data, and filter out my main dataset to get only the
non-outlier rows. Then, I can merge this filtered dataframe with the nutrition dataframe.

The final cleaning step is to create a healthy column, which contains True or False values depending on if the string "healthy" is in the tags for a recipe or not. 

After all of my data cleaning, my dataset will look like the table below:

***Project Dataset Schema***
(234429 rows × 25 columns)

| name | id | minutes | contributor_id | submitted | tags | n_steps | steps | description | ingredients | n_ingredients | user_id | recipe_id | date | rating | review | avg_rating | calories | total fat | sugar | sodium | protein | saturated fat | carbohydrates | healthy |
|:-----|:---|:--------|:--------------|:----------|:-----|:--------|:------|:------------|:------------|:--------------|:--------|:----------|:-----|:-------|:-------|:-----------|:---------|:----------|:------|:-------|:--------|:--------------|:--------------|:---------|
| 1 brownies in the world best ever | 333281 | 40 | 985201 | 2008-10-27 | ['60-minutes-or-less', 'time-to-make', ...] | 10 | ['heat the oven to 350f and arrange the rack in the middle', ...] | these are the most; chocolatey, moist, rich, dense, fudgy, ... | ['bittersweet chocolate', 'unsalted butter', ...] | 9 | 386585 | 333281 | 2008-11-19 | 4 | These were pretty good, but took forever to bake. I would... | 4 | 138.4 | 10 | 50 | 3 | 3 | 19 | 6 | False |
| 1 in canada chocolate chip cookies | 453467 | 45 | 1848091 | 2011-04-11 | ['60-minutes-or-less', 'time-to-make', ...] | 12 | ['pre-heat oven the 350 degrees f', ...] | this is the recipe that I use at my school cafeteria for... | ['white sugar', 'brown sugar', 'salt', ...] | 11 | 424680 | 453467 | 2012-01-26 | 5 | Originally I was gonna cut the recipe in half (just the 2 of us... | 5 | 595.1 | 46 | 211 | 22 | 13 | 51 | 26 | False |
| 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', ...] | 6 | ['preheat oven to 350 degrees', ...] | since there are already 411 recipes for broccoli casserole... | ['frozen broccoli cuts', 'cream of chicken soup', ...] | 9 | 29782 | 306168 | 2008-12-31 | 5 | This was one of the best broccoli casseroles that I have ever... | 5 | 194.8 | 20 | 6 | 32 | 22 | 36 | 3 | False |
| 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', ...] | 6 | ['preheat oven to 350 degrees', ...] | since there are already 411 recipes for broccoli casserole... | ['frozen broccoli cuts', 'cream of chicken soup', ...] | 9 | 1.19628e+06 | 306168 | 2009-04-13 | 5 | I made this for my son's first birthday party this weekend... | 5 | 194.8 | 20 | 6 | 32 | 22 | 36 | 3 | False |
| 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', ...] | 6 | ['preheat oven to 350 degrees', ...] | since there are already 411 recipes for broccoli casserole... | ['frozen broccoli cuts', 'cream of chicken soup', ...] | 9 | 768828 | 306168 | 2013-08-02 | 5 | Loved this. Be sure to completely thaw the broccoli. I didn't... | 5 | 194.8 | 20 | 6 | 32 | 22 | 36 | 3 | False |

The main columns I used in this project are: "minutes", "calories", "total fat", "sugar", "sodium", "protein", "saturated fat", "carbohydrates", and "healthy".
The schema for those variables is:

| minutes                | (int64) number of minutes to prepare |
| calories               | (int64) number of calories in recipe |
| total fat              | (int64) grams of total fat in recipe |
| sugar                  | (int64) grams of sugar in recipe     |
| sodium                 | (int64) grams of sodium in recipe    |
| protein                | (int64) grams of protein in recipe   |
| saturated fat          | (int64) grams of saturated fat in recipe |
| carbohydrates          | (int64) grams of carbohydrates in recipe |
| healthy               | (boolean) True if recipe is healthy, False if not |




### Univariate Analysis

<iframe
  src="assets/univar1.html"
  width="800"
  height="600"
  frameborder="0"></iframe> This histogram shows the distribution of calories in the entire recipe dataset. The plot shows that most of the recipes in my dataset are within 0-500 calories. Then, there is a sharp taper off, until I reach 1000 calories, at which point there are very little recipes that have over 1000 calories.

<iframe
  src="assets/univar2.html"
  width="800"
  height="600"
  frameborder="0"
  ></iframe>This histogram gives us the distribution of healthy and unhealthy recipes out of all the recipes in the dataset. The plot shows that a majority of the recipes are unhealthy.

### Bivariate Analysis

<iframe
  src="assets/bivar1.html"
  width="800"
  height="600"
  frameborder="0"
  ></iframe> This boxplot shows the distribution of calories out of all recipes in the dataset based on the recipe category (healthy or unhealthy). By comparing the median values, I can see that this plot shows that healthy recipes are lower in calories than unhealthy recipes.

<iframe
  src="assets/bivar2.html"
  width="800"
  height="600"
  frameborder="0"
  ></iframe> This scatterplot shows us the relationship betIen calories and total fat. The plot shows that calories and total fat have a strong, positive correlation. So, meals that are high in fats are also high in calories and vice versa.

### Interesting Aggregates
By grouping columns by the healthy column, I created this pivot table, which stores the mean value for each nutrition column based on the recipe
category (healthy or unhealthy). This pivot table more easily displays the differences in nutrition values for healthy and unhealthy foods.

| healthy   |   calories |   carbohydrates |   protein |   saturated fat |   sodium |   sugar |   total fat |
|:----------|-----------:|----------------:|----------:|----------------:|---------:|--------:|------------:|
| False     |    351.232 |         9.15961 |   32.0619 |         36.4012 |  22.4048 | 37.0908 |     28.7803 |
| True      |    257.48  |        11.7311  |   21.998  |         10.0385 |  17.3241 | 43.7574 |     10.9076 |

It shows that healthy foods are, on average, lower in calories, higher in carbs, lower in protein, lower in saturate fat, lower in sodium, higher in sugar, and
lower in total fat. The difference between total and saturate fat in healthy and unhealthy foods is stark, likely indicating that fat content has a high impact
on calories and whether or not the recipe is healthy.

## Assessment Of Missingness

|                |       |
|:---------------|------:|
| name           |     1 |
| id             |     0 |
| minutes        |     0 |
| contributor_id |     0 |
| submitted      |     0 |
| tags           |     0 |
| n_steps        |     0 |
| steps          |     0 |
| description    |   114 |
| ingredients    |     0 |
| n_ingredients  |     0 |
| user_id        |     1 |
| recipe_id      |     1 |
| date           |     1 |
| rating         | 15036 |
| review         |    58 |
| avg_rating     |  2777 |
| calories       |     0 |
| total fat      |     0 |
| sugar          |     0 |
| sodium         |     0 |
| protein        |     0 |
| saturated fat  |     0 |
| carbohydrates  |     0 |
| healthy        |     0 |
| missing_rating |     0 |

### NMAR Analysis
The "rating" column has the most missing values, at 15036. I believe that the "rating" column is NMAR (Not Missing At Random). This can be attributed to the fact that
people will usually only give recipes if they feel strongly about a recipe. They either really loved it and want to give it 5 stars, or they really hated it and want to
give it 1 star. A majority of the people who look up recipes on food.com likely just want to find something to cook. They probably won't come back after eating to give
their review of the recipe. I've done this many times, and I'm sure this is a common trend.

### Missingness Dependency
I will use permutation tests to analyze if the missingness of the "rating" column is dependent on "avg_rating" and "minutes".

For testing the relationship betIen "rating" and "avg_rating":

Null Hypothesis: The missingness of the "rating" column is not dependent on the "avg_rating" column.

Alternative Hypothesis: The missingness of the "rating" column is dependent on the "avg_rating" column.

Test Statistic: Difference in means

Results:

<iframe
  src="assets/missingdist1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> After running the permutation test, I got a p value of 0.0. Since this is below our significance level of 0.05, I will reject the null hypothesis. This shows that the missingness of "rating" is dependent on the "avg_rating" column.

Second Permutation Test:

Null Hypothesis: The missingness of the "rating" column is not dependent on the "minutes" column.

Alternative Hypothesis: The missingness of the "rating" column is dependent on the "minutes" column.

Test Statistic: Difference in means

Results:

<iframe
  src="assets/missingdist2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> After running the permutation test, I found a p value of 0.11. Since this p value is above our significance level of 0.05, I will fail to reject the null hypothesis. This shows that the missingness of "rating" is not dependent on the "minutes" column.

## Hypothesis Testing
I want to run a hypothesis test to answer the following question: Do healthy recipes have a lower average calorie count than unhealthy recipes? So, I ran a 
permutation test to answer the question.

Null Hypothesis: There is no difference in the mean calorie count between healthy and unhealthy recipes.

Alternative Hypothesis: Healthy recipes have a lower average calorie count than unhealthy recipes.

Test Statistic: Difference in Means

Results:

My resulting p value was 0.0, which means that I can reject the null hypothesis that there is no difference in the mean calorie count between healthy and unhealthy
recipes. There is statistical significance to show that healthy and unhealthy recipes have different mean calorie counts. This is relevant to the project question
because I want to ensure that there is statistical significance to show that the variation in calorie count between healthy and unhealthy food is not due to 
random chance. This implies (although it doesn't prove or otherwise say conclusively) that healthy foods are lower in calories.

## Framing A Prediction Problem
The hypothesis test in the previous section showed us that there is a difference in the number of calories in a recipe depending on whether it is healthy or not.
So in the following sections, I want to be able to answer the following prediction problem: Can I predict the number of calories in a recipe based on other
information (like total fat, carbohydrates, n_ingredients, healthy, etc.)?

To start, I want to build a baseline model. I will begin by trying to predict how many calories a meal will have based on the amount of protein, carbohydrates,
and sugar.

## Baseline Model

To start, I want to build my baseline model off of the three features: protein, carbohydrates, and sugar. These three are all numerical quantitative variables.
In the pipeline, I used the StandardScaler on my three quantitative features to standardize them. Then, I use LinearRegression as my model on which I train my data.
The model scored an r^2 value of 0.761, indicating that the model is decently good at explaining the variability in my target feature, calories. Although my model is 
pretty good, it can definitely be improved upon. This is a good starting point for my model.

## Final Model

To build on my baseline model, I added two new features on top of the three original numerical quanitative features: total fat and saturated fat. These two features 
are also numerical quanitative features. So, my model now takes in the five numerical quantitative features: protein, carbohydrates, and sugar, total fat and saturated
fat. I thought that it would help the model to have the fat features because fat is a significant contributor to calorie count. Additionally, I engineered two features: 
carb_to_sugar_ratio and sat_fat_ratio. The carb_to_sugar_ratio feature represents the ratio of carbohydrates to sugar. The sat_fat_ratio feature represents the ratio of
saturated fats to total fats. These two engineered features are also numerical quantitative features. I thought that these features would improve the model because sugar
is a type of carbohydrate, so it would be helpful to differentiate how much of a recipe's carbs are coming from sugar and because saturated fats are typically associated
with more unhealthy, high-carbohydrate food, so it would be helpful for the model to differentiate how much of a recipe's total fat content is saturated fat. I use the 
StandardScaler on all of these quantitative features to standardize them. Next, I use the OneHotEncoder on the healthy feature to convert the feature so that the model 
can use the feature, since it is a categorical feature, so it has to be encoded. I thought it would be helpful for the model to be able to differentiate betIen if a recipe is
healthy or not, because as I saw earlier in the project, healthy meals typically (based on median) have lowwer calories than unhealthy meals. In the pipeline, I use the
above features, and for the model, I use a RandomForestRegressor, which is good for finding non-linear relationships between features. This is useful because of all of
the features I am passing into this model, whose relationships may not be best represented through linear relationships. Finally, I used GridSearchCV to tune
hyperparameters. I tuned the max_depth hyperparameter to have a max depth of 5 to give a preset limit to the tree depth. I tuned the n_estimators hyperparameter to 50 to ensure
that the model didn't run very long. Finally, I ran GridSearchCV with 3 CV folds to test the model while still prioritizing runtime. After running the final model,
I got a best CV r^2 value of 0.992 and a r^2 value of 0.993. This is a significant jump from the baseline model's performance. This model is able to almost perfectly capture
all of the variability in calories using the features it had passed in. These new features add much more information for the model to be able to use to predict calories.

## Fairness Analysis

To test the fairness of my final model, I want to test whether the model performs worse for healthy recipes versus unhealthy recipes. I chose to use RMSE at the
performance metric. To perform a permutation test, the null and alternative hypotheses are:

Null Hypothesis: The model’s RMSE is the same for healthy recipes as for unhealthy recipes. Any observed difference is due to random chance.

Alternative Hypothesis: The model’s RMSE is higher for healthy recipes than for unhealthy recipes. Any observed difference is not due to random chance.

After running the permutation test, I got a p value of 0.069. This is greater than my significance level cutoff of 0.05, so I fail to reject the null hypothesis.
This means that there is not enough evidence to say that the model performs better for healthy recipes than unhealthy recipes.
