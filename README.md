[Link to Full Slide Deck (with comments)](https://docs.google.com/presentation/d/1zF-Ks9rmsYUv2dtZwkdY6UG-0l-4HsDpkpiLjPmbJOs/edit?usp=sharing)

# Introduction

In the context of this project, I am playing the role of an outside data consultant tasked with analyzing **Reddit** data in order to better understand customer feedback regarding **Alexa and Google Home devices**. 

Reddit data includes useful discussions that reveal customer opinions, however, since the data is text heavy, reading through each post is too time consuming of a task for a human.

#### Objective 

My objective is to use **NLP** to compare subreddit content, which will be used to **train a classifier on which subreddit a given post came from**. 

By assessing the data, we can **understand** **what users are saying** about each company’s devices and **provide better business recommendations**. 

# Data Description

To obtain my data, I utilized a python package called [PRAW (Python Reddit API Wrapper)](https://praw.readthedocs.io/en/latest/index.html) to scrape the Reddit API for roughly 1000 posts from each subreddit.

#### [Amazon Alexa Subreddit - r/alexa](https://www.reddit.com/r/alexa/) 

* 39.9k Members
* Created on Feb 19, 200
* 9998 Posts
* 4482 Comments

#### [Everything Google Home Subreddit - r/googlehome](https://www.reddit.com/r/googlehome/)

* 186K Members

* Created on May 18, 2016

* 968 Posts 

* 6233 Comments

  

Reference the following links for descriptions of the Data Dictionary:

* [Submissions](https://praw.readthedocs.io/en/latest/code_overview/models/submission.html)
* [Comments](https://praw.readthedocs.io/en/latest/code_overview/models/comment.html) 

# Exploratory Data Analysis

#### How are posts distributed over time? By Scores? By Comments?

As we'll see from the charts below, since the Google Home subreddit has roughly 140K more members, the amount of posts, scores, and comments tend to be higher than the Alexa subreddit. 

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/alexa_posts.jpeg"/> 

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/google_posts.jpeg"/>

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/alexa_scores.jpeg"/> <img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/google_scores.jpeg"/> 

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/alexa_comments_box.jpeg"/> 

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/google_comments_box.jpeg"/>  

#### What are the most frequent words in both datasets?

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/alexa_most_freq_1.jpeg"/>  

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/google_most_freq_1.jpeg"/>  

The word counts here reveal to us that some of the most common words that are **unique** to each invidivual dataset are `alexa`, `echo`, `dot`, `google`, `home`, `nest`, `assistant`, which make these words make good predictors for our model.

Alternatively, the most common words that are the **same** in each dataset are words like `app`, `device`, `devices`, `music`, `play`, `set`, `spotify`, `tv`. These words may reveal some information about the content of the posts, which discuss apps, using spotify, playing music, and how voice devices interact with tv. 

#### How positive, neutral, or negative is the sentiment of a post?

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/alexa_sentiment_compound.jpeg"/>  

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/google_sentiment_compound.jpeg"/>  

I classified the sentiment by classifying the compound score as follows:

* Negative =  `< -0.05`
* Neutral =  `>= -0.05` and `<= 0.05`
* Positive = `> 0.05` 

We can see that the posts are generally classified as positive, with the Alexa's positive percentage **(62.9%)** being slightly higher than **Google Home's (59.3%)**. Both subreddits have roughly a **26%** **negative** score. 

#### What can we learn from posts with the highest scores and most comments?

* Funny images and videos get the most upvotes and comments. 
* Feature requests, collective user frustrations, and product announcements also get more upvotes and comments.

**Examples:**

* “I created a skill that will play “Piano Man” by Billy Joel so it will play the line “it’s 9 o’clock on a Saturday” at exactly 9:00 on a Saturday.”
* “the alexa app is crap. it's slow, laggy, fully of bugs, and just a pain in the arse to use.”
* “Anyone Else Get Really Annoyed by Alexa’s “By the way…”
* “Dark mode has arrived, at long last!”
* “Do yourself a favor and ask google “what sound does a camel make?”	
* “Favorite Automation or Routine?”
* “Got my Alexa glasses today!“
* “Monthly Rants and Complaints Thread - August 2020”
* “Google Play Music will be discontinued starting September. New Zealand and South Africa you go first.”

# Modeling Process

Since I used a balanced dataset to train my model, my **baseline score to beat was 50%**. 

For my first models, I used a **Logistic Regression** model and a **Random Forest** model. I utilized `CountVectorizer` to things such as add stop words, in addition to `GridSearchCV` to tune my hyperparameters of the model. 

In this case of my predictions, `Google Home = 0` and `Alexa = 1`.

#### Logistic Regression Results

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/log_reg1_curve.jpeg"/> 

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/log_reg1_confusion_norm.jpeg"/> 

**Training Accuracy  - 99.6%**

**Testing Accuracy - 92.4%**

#### Random Forest Results

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/rf1_curve.jpeg"/> 

<img src="https://git.generalassemb.ly/Jmizraji/project_3/blob/master/imgs/rand_forest_confusion_norm.jpeg"/> 

**Training Accuracy  - 94.8%**

**Testing Accuracy - 91.6%**

##### Since my model performed so well, I decided to add the following to the stop words list to make predicting more challenging:  `alexa` , `echo`,  `dot`, `google`, `home`, `nest`, `assistant`

## New Model Results

| Model                                                    | Training Accuracy | Testing Accuracy |
| -------------------------------------------------------- | ----------------- | ---------------- |
| Logistic Regression                                      | 99.3%             | 79.6%            |
| Random Forest                                            | 71.9%             | 69.6%            |
| Multinomial Naive Bayes                                  | 92.4%             | 73.5%            |
| Random Forest  (Bagging + AdaBoost + Voting Classifiers) | 87.7%             | 75.9%            |

The **Logistic Regression** model was still my best performing model, depending on how much variance you want your model to handle.  

My model with the best bias/variance tradeoff, was the original **Random Forest** model. However, it has the worst accuracy out of all of my other models. 

# Recommendations 

* **Humor Gets Engagement** - Utilize funny posts in your social media and marketing. Showcase the innovative and humorous things users are creating with the devices.
  * **Idea:** Repost funny memes or ideas from users. Come up with your own. Encourage engagement within the community.
* **Consider Feature Requests** - Creative ideas can come from anywhere. Explore feature requests to find creative ideas for new features and skills. 
  * **Idea:** Use NLP to scrape Reddit top voted feature requests and sent out a weeky email with ideas to the team to generate useful skills.  
* **Listen To User Frustrations** - Users will switch to another product if their needs are not met. Consider pitfalls, engage with these users, and improve the overall experience.
  * **Idea:** Combine common compaints with other feedback sources to get a full picture of user frustrations. Directly engage with users through social media and share feedback with product and support teams so that they can fine tune their strategy.
* **Utilize NLP With Additional Outlets** - Apply similar analysis to App store reviews, product reviews, and social media comments.
  * **Idea:** Create a sentiment analysis with other outlets to measure customer opinions and find common themes. 

