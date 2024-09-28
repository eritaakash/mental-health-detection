# Mental Health Sentiment Analysis

## Overview
This project aims to identify the sentiment of a person's mental health based on textual inputs. By leveraging Natural Language Processing (NLP) and Machine Learning (ML), the model can classify messages into one of the following categories:
- Normal
- Depressed
- Exhibiting signs of a mental disorder
- Suicidal

## Dataset
The model is trained on a dataset of approximately 50,000 labeled text messages. The classification is based on patterns in text that align with various mental health conditions. While this is the first version of the model, it achieves an accuracy of 77% and provides useful insights, though some misclassifications may occur.

## Technologies Used

### Backend
- **Pandas** and **Numpy**: For data manipulation and preparation.
- **NLTK**: For natural language processing tasks such as tokenization and text cleaning.
- **Scikit-learn**: For model building and evaluation.
- **XGBoost**: The primary classifier used for sentiment analysis.
- **Joblib/Pickle**: To save and load trained models.
- **FastAPI**: To create the backend API for serving the model.
- **Uvicorn**: ASGI server to run the FastAPI application.

### Frontend
- **React/NextJS**: For building the frontend interface.
- **FeatherIcons**: Lightweight and customizable icons used in the frontend.
- **Sass**: For styling and enhancing the user interface with custom CSS.

## Model Performance
- **Accuracy**: 77%
- The model uses the XGBoost classifier, known for its speed and accuracy in handling classification tasks.

## Model Limitations

This model is still in its first version and may misclassify messages. Further training and fine-tuning can help improve its accuracy.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.