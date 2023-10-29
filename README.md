# 🚀 Natural language processing NLP implementation using the BERT Sentiment Analysis App. 🚀

[![View Repositories](https://img.shields.io/badge/View-My_Repositories-blue?logo=GitHub)](https://github.com/justinjabo250?tab=repositories)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=orange)](https://www.linkedin.com/in/jabo-justin-2815341a2/) 
[![Medium Article](https://img.shields.io/badge/Medium-Article-purple)](https://medium.com/@jabojustin250/natural-language-processing-nlp-implementation-using-the-bert-sentiment-analysis-app-12b98fc8a300)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-yellow)](https://huggingface.co/spaces/Justin-J/Sentiment-Analysis-Application-Using-Streamlit)
[![Gradio App](https://img.shields.io/badge/Gradio-App-yellow)](https://huggingface.co/spaces/Justin-J/Sentiment-Analysis-App-using-Gradio)
[![Articles](https://img.shields.io/badge/My-Portfolio-darkblue?logo=Website)](https://justinjabo250.github.io/Developing-a-web-application-for-an-online-portfolio./)
[![View GitHub Profile](https://img.shields.io/badge/GitHub-Profile-darkgreen)](https://github.com/justinjabo250)


<img src="https://user-images.githubusercontent.com/115732734/271723332-6c824e95-5e2f-48ec-af1c-b66ac7db1d7a.jpeg" width="550">

# 🚀 Friendly Web Interface for Machine Leaning Project Project with Gradio and and Streamlit 🚀

There are many ways to make web interfaces to allow interaction with Machine Learning models and we will cover one of them.

In order to adjust the models to predict the feelings stated in a Tweet (e.g., neutral, positive, or negative), I fine-tuned pre-trained Deep Learning models from HuggingFace on a new dataset for this project.

The sentiment analysis will be analyzed with the aid of this BERT program. Understanding the mood underlying social media messages has become essential in today's fast-paced digital environment, when ideas and emotions are communicated in bite-sized bursts. Enter the world of sentiment analysis, where we use cutting-edge HuggingFace models and deep learning to uncover the mysteries held inside the huge sea of tweets and status updates.

<img src="https://user-images.githubusercontent.com/115732734/271723337-bc1ce6d1-8bed-4c75-86b3-fbb2d2eef919.png" width="550">

# Introduction
The COVID-19 pandemic has had a significant global impact, affecting economies, civilizations, and the lives of people all over the world. The development and distribution of COVID-19 vaccinations has been a significant initiative to solve this epidemic. These vaccines have proven to be remarkably successful at reducing fatalities and serious illnesses. However, false information and divisive discussions about their security, effectiveness, and fair distribution have also surfaced.



## Summary
| Code      | Name        | Published Article |  Deployed App |
|-----------|-------------|:-------------:|------:|
| Gradio and and Streamlit App | BERT Sentiment Analysis App based Natural Language Processing (NLP) implementation Project| [Read more here](https://medium.com/@jabojustin250/natural-language-processing-nlp-implementation-using-the-bert-sentiment-analysis-app-12b98fc8a300) | [![Streamlit App](https://img.shields.io/badge/Streamlit-App-yellow)](https://huggingface.co/spaces/Justin-J/Sentiment-Analysis-Application-Using-Streamlit) [![Gradio App](https://img.shields.io/badge/Gradio-App-yellow)](https://huggingface.co/spaces/Justin-J/Sentiment-Analysis-App-using-Gradio) |

<img src="https://user-images.githubusercontent.com/115732734/271723345-50f27ca9-94ee-4e7c-ad3b-2b10f27d31bb.jpeg" width="550">

## Project Description

I'll use pre-trained models from Huggingface to carry out a sentiment analysis of tweets about the COVID-19 vaccine. These models may be customized for different natural language processing (NLP) applications, including sentiment analysis, and have undergone training on huge volumes of text data. I'll use these pre-trained algorithms to classify tweets into favorable, negative, or neutral feelings by compiling a dataset of tweets about COVID-19 vaccinations.

# Dataset

Tweets have been classified as pro-vaccine (1), neutral (0) or anti-vaccine (-1). The tweets have had usernames and web addresses removed.


## Variable definition:


. **tweet_id:** A special tweet identification number.

. **safe_tweet:** The tweet's text content. Like usernames and urls, some sensitive information has been eliminated.

. **label:** Tweet sentiment (-1 for a negative tweet, 0 for a neutral tweet, 1 for a good tweet).

. **agreement:** Three persons assigned labels to the tweets. The percentage of the three reviewers who agreed with the given label is indicated by the word "agreement." Although you are allowed to use this column in your training, agreement information for the test set won't be shared.




## Files available for download are:

. **Train.csv** - Using labeled tweets to train your model.

. **Test.csv** - Twitter posts that need to be categorized using your trained model.

. **SampleSubmission.csv** - is an illustration of how your submission file should appear. The names of the IDs must be accurate, but the order of the rows is irrelevant. The 'label' column should contain values between -1 and 1.

. **NLP_Primer_twitter_challenge.ipynb** - is a starter notebook to assist you in submitting your initial entry for this challenge.



## Tools, software systems, and libraries needed for the project Setup

To run the app locally, follow these steps:

1. Clone the repository:

git clone git clone https://huggingface.co/eric2013/covid



2. Navigate to the project directory:

cd sentiment-analysis-app
  

  
3. Install the required dependencies:

pip install -r requirements.txt



4. Run the Streamlit app:

streamlit run sentimentapp.py


Open your web browser and visit http://127.0.0.1:7865 to access the app.


**Note:** 

. Run the notebook using Google Colab by forking this repository. Since the hugging face models are Deep Learning-based, training them will require a lot of GPU processing power. Use [Colab](https://colab.research.google.com/), another GPU cloud service, or a local workstation with an NVIDIA GPU to complete the task.

. Please be aware that Google Colab sessions have time limits and may end if inactive for a certain amount of time. You can, however, save your work in progress and reconnect to the GPU as necessary.

. Machine learning platforms and open-source software are offered by Hugging Face. Installing their package will give you access to some intriguing pre-built models that you can use right away or fine-tune (retrain on your dataset using the prior knowledge gained from the first training), after which you can host your trained models on the platform and use them later on other hardware and software.


![**Please use the platform's full feature set.**][go to the website and sign-in](https://huggingface.co/)

-[**Read more about Hugging Face's Text Classification.**](https://huggingface.co/tasks/text-classification)

# Evaluation

The **Root Mean Squared Error** is used as the assessment metric for this assignment.


## Screenshots

### Streamlit App

![ezgif com-optimize](https://github.com/ikoghoemmanuell/Sentiment-Analysis-with-Finetuned-Models/assets/102419217/de9740aa-dcc8-4215-bbf5-525f44db0050)

### Gradio App

![ezgif com-crop (1)](https://github.com/ikoghoemmanuell/Sentiment-Analysis-with-Finetuned-Models/assets/102419217/e6177a08-f3b0-4bda-9a83-7031c36235b0)

## Resources

1. [Quick intro to NLP](https://www.youtube.com/watch?v=CMrHM8a3hqw)
1. [Getting Started With Hugging Face in 15 Minutes](https://www.youtube.com/watch?v=QEaBAZQCtwE)
1. [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs)
1. [Fine-Tuning-DistilBert - Hugging Face Transformer for Poem Sentiment Prediction | NLP](https://www.youtube.com/watch?v=zcW2HouIIQg)
1. [Introduction to NLP: Playlist](https://www.youtube.com/playlist?list=PLM8wYQRetTxCCURc1zaoxo9pTsoov3ipY)
<!-- 1. [](https://www.youtube.com/)
1. [](https://www.youtube.com/) -->


## Contributing

Please feel free to submit a PR or report a problem 😃.

Oh, and just one more thing: while creating your PR, please remember to include a description 🙂.

## 👏 Support

If you found this article helpful, please give it a clap or a star on GitHub!👏 😃 🙂.


## Author
[Justin Jabo]
- [Linkedin Article](https://www.linkedin.com/pulse/bert-sentiment-analysis-app-based-natural-language-nlp-jabo-justin) 
- [Medium Article](https://medium.com/@jabojustin250/bert-sentiment-analysis-app-901447e80a2f)
- [Github Repository](https://github.com/justinjabo250)

## To View My Apps
- [![Streamlit App](https://img.shields.io/badge/Streamlit-App-yellow)](https://huggingface.co/spaces/Justin-J/Sentiment-Analysis-Application-Using-Streamlit)
- [![Gradio App](https://img.shields.io/badge/Gradio-App-yellow)](https://huggingface.co/spaces/Justin-J/Sentiment-Analysis-App-using-Gradio)
