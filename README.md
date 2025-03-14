# Recipe Deep-Dive: Analyzing and Predicting Calorie Counts
*By: Rushyendra Katabathuni*

## Introduction

This project is the final project for the course DSC 80 at the University of California, San Diego. This project analyzes a dataset of recipes which were scraped from food.com. This project seeks to first understand and answer the question: what types of recipes tend to be healthier and what nutritional values correlate with healthier foods? For example, do healthy foods have more protein than unhealthy foods? Do they have less carbohydrates than unhealthy foods? Then, this project will use models to predict the number of calories in recipes.

To begin, we had two datasets: recipes and ratings. The Recipes dataset contained information about recipes. These recipes were uploaded by users on food.com. The Ratings dataset contains information on ratings that users left for a specific recipe.

***Recipes Schema***

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

| Column Name            | Description                            |
|------------------------|---------------------------------------:|
| user_id                | (int64) unique user ID                 |
| recipe_id              | (int64) unique recipe ID               |
| date                   | (object) date rating/ review submitted |
| rating                 | (int64) rating from 1-5 submitted      |
| review                 | (object) review submitted              |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

We start the data cleaning process by left merging the recipes and ratings data. We want to left merge the ratings data onto recipes because we want to keep all recipes even
if they don't have a review for them. We merge the recipes data on the "id" column with the ratings data on the "user_id" column. Once we have our initial merged dataset, we 
want to ensure that values are properly represented. So first, we made sure to replace all occurences of "0" rating scores to "np.nan" rating scores. This is because a "0" 
rating doesn't make sense as a value, as it really just represents an absence of a rating. We wouldn't want to treat "0" ratings as valid ratings because they will skew the
average rating that we calculate for each recipe. Next, we want to calculate the average rating. We can do this by grouping the dataset by the "id", then extracting the
rating and calculating the mean for each group. Then, we want to merge this column with the main dataset. 

The next cleaning we want to do is converting the nutrition column from a list of values to individual values. The nutrition value for each row in a recipe contains
information about the nutritional value of that meal. This information is represented in the shape of a list, however it is actually stored as a string. So, we first 
convert each nutrition string in the column into a list. Then, we can put each value in the list into its own column, and convert that into a dataframe. 

The third cleaning step involves filtering out outliers. This dataset contains some significant outliers that skew the data and significantly impact visualizations. For ease,
we want to exclude these values from our data analysis. Logically, this makes sense, because a recipe with 45,000 calories is outside of regular consumption. The daily caloric
recommendation is 2000 calories, so I think it is valid to exclude these outliers for the sake of better visualizations and analyses. To filter out outliers, we defined a
method, "filter_outliers", which returns only those values within our lower and upper bounds. We then pass through every column in the nutrition dataframe, to remove outliers
for each column. Once we've filtered out the data to exclude outliers, we want to get the indices of the remaining data, and filter out our main dataset to get only the
non-outlier rows. Then, we can merge this filtered dataframe with the nutrition dataframe.

The final cleaning step is to create a healthy column, which contains True or False values depending on if the string "healthy" is in the tags for a recipe or not. 

After all of our data cleaning, our dataset will look like the table below:

| name | id | minutes | contributor_id | submitted | tags | n_steps | steps | description | ingredients | n_ingredients | user_id | recipe_id | date | rating | review | avg_rating | calories | total fat | sugar | sodium | protein | saturated fat | carbohydrates | healthy |
|:-----|:---|:--------|:--------------|:----------|:-----|:--------|:------|:------------|:------------|:--------------|:--------|:----------|:-----|:-------|:-------|:-----------|:---------|:----------|:------|:-------|:--------|:--------------|:--------------|:---------|
| 1 brownies in the world best ever | 333281 | 40 | 985201 | 2008-10-27 | ['60-minutes-or-less', 'time-to-make', ...] | 10 | ['heat the oven to 350f and arrange the rack in the middle', ...] | these are the most; chocolatey, moist, rich, dense, fudgy, ... | ['bittersweet chocolate', 'unsalted butter', ...] | 9 | 386585 | 333281 | 2008-11-19 | 4 | These were pretty good, but took forever to bake. I would... | 4 | 138.4 | 10 | 50 | 3 | 3 | 19 | 6 | False |
| 1 in canada chocolate chip cookies | 453467 | 45 | 1848091 | 2011-04-11 | ['60-minutes-or-less', 'time-to-make', ...] | 12 | ['pre-heat oven the 350 degrees f', ...] | this is the recipe that we use at my school cafeteria for... | ['white sugar', 'brown sugar', 'salt', ...] | 11 | 424680 | 453467 | 2012-01-26 | 5 | Originally I was gonna cut the recipe in half (just the 2 of us... | 5 | 595.1 | 46 | 211 | 22 | 13 | 51 | 26 | False |
| 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', ...] | 6 | ['preheat oven to 350 degrees', ...] | since there are already 411 recipes for broccoli casserole... | ['frozen broccoli cuts', 'cream of chicken soup', ...] | 9 | 29782 | 306168 | 2008-12-31 | 5 | This was one of the best broccoli casseroles that I have ever... | 5 | 194.8 | 20 | 6 | 32 | 22 | 36 | 3 | False |
| 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', ...] | 6 | ['preheat oven to 350 degrees', ...] | since there are already 411 recipes for broccoli casserole... | ['frozen broccoli cuts', 'cream of chicken soup', ...] | 9 | 1.19628e+06 | 306168 | 2009-04-13 | 5 | I made this for my son's first birthday party this weekend... | 5 | 194.8 | 20 | 6 | 32 | 22 | 36 | 3 | False |
| 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', ...] | 6 | ['preheat oven to 350 degrees', ...] | since there are already 411 recipes for broccoli casserole... | ['frozen broccoli cuts', 'cream of chicken soup', ...] | 9 | 768828 | 306168 | 2013-08-02 | 5 | Loved this. Be sure to completely thaw the broccoli. I didn't... | 5 | 194.8 | 20 | 6 | 32 | 22 | 36 | 3 | False |

### Univariate Analysis

<iframe
  src="assets/univar1.html"
  ></iframe>
This histogram shows the distribution of calories in the entire recipe dataset. The plot shows that most of the recipes in our dataset are within 0-500 calories.
Then, there is a sharp taper off, until we reach 1000 calories, at which point there are very little recipes that have over 1000 calories.

<iframe
  src="assets/univar2.html"
  ></iframe>
This histogram gives us the distribution of healthy and unhealthy recipes out of all the recipes in the dataset. The plot shows that a majority of the recipes are
unhealthy.

### Bivariate Analysis

<iframe
  src="assets/bivar1.html"
  ></iframe>
This boxplot shows the distribution of calories out of all recipes in the dataset based on the recipe category (healthy or unhealthy). By comparing the median values,
we can see that this plot shows that healthy recipes are lower in calories than unhealthy recipes.

<iframe
  src="assets/bivar2.html"
  ></iframe>
This scatterplot shows us the relationship between calories and total fat. The plot shows that calories and total fat have a strong, positive correlation. So, meals that
are high in fats are also high in calories and vice versa.

### Interesting Aggregates


## Assessment Of Missingness


## Hypothesis Testing

## Framing A Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis
