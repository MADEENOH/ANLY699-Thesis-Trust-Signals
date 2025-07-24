How Trust Signals in Product Reviews Predict Recommendation Behavior: 
A Behavioral Study Using E-Commerce Data


This repository contains the code and resources for the ANLY 699 thesis paper, "How Trust Signals in Product Reviews Predict Recommendation Behavior: A Behavioral Study Using E-Commerce Data."

This research investigates how the complex language within online product reviews known as "trust signals" can predict a consumer's recommendation behavior. The study first establishes a baseline using traditional machine learning models on structured data, then demonstrates the superior performance of a fine-tuned Large Language Model (Mistral-7B) on the unstructured review text. The final model achieves over 92% accuracy in predicting recommendations based solely on the text of a review.
Key Features

    Exploratory Data Analysis (EDA): Initial analysis and visualization of the dataset to identify key trends and imbalances.

    Baseline Modeling: Implementation of Random Forest, Gradient Boosting, and XGBoost models on numerical data.

    LLM Fine-Tuning: Advanced text classification using a fine-tuned Mistral-7B model with the Unsloth library for high-efficiency training.

    Results Analysis: Detailed performance evaluation including accuracy, F1-scores, a confusion matrix, and N-gram analysis to identify key "trust signal" phrases.

Dataset

The study uses the publicly available "Women's Clothing E-Commerce Reviews" dataset from Kaggle.

    Link: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

Fine-Tuned Model

The final, fine-tuned sentiment analysis model is publicly available on the Hugging Face Hub.

    Model Link: https://huggingface.co/MadeEnoh/mistral-7b-clothing-sentiment-v1

How to Use This Code

This project was developed in a Google Colab environment. The primary file is LLM_Fine_Tuning_Analysis.ipynb.

    Environment Setup: To run the notebook, you will need a Python environment with the libraries listed in the notebook, including pandas, unsloth, transformers, torch, seaborn, and scikit-learn.

    Data: Download the dataset from the Kaggle link above and place it in the /content/data/ directory within the Colab environment.

    Execution: Run the cells in the notebook sequentially to replicate the data analysis, model training, and evaluation process.

Code Availability Statement

The complete code used for data analysis, model training, and fine-tuning in this study is contained within this repository.
