# Cat and Dog Image Classification

![Default Homepage][index-screenshot]

## About The Project

This project demonstrates a simple web interface using Flask as the backend and Bootstrap as the frontend.  

The main goal is to experiment with CNN models in PyTorch. Specifically, a CNN model for binary classification of images of cats and dogs is developed.  
 
The minimalist web interface allows users to upload images, and the model provides corresponding predictions and accuracy.

### Built With

Major libraries, frameworks and languages used in this project:
* [![Python][Python.com]][Python-url]
* [![Pytorch][Pytorch.com]][Pytorch-url]
* [![Flask][Flask.com]][Flask-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]

## Table of Contents

1. [Getting Started](#getting-started)
2. [Usage](#usage)
3. [Model Overview](#model-overview)
5. [Acknowledgments](#acknowledgments)
6. [License](#license)


## Getting Started

_To get a local copy up and running, follow these steps._

### Prerequisites

Ensure you have the following dependencies installed:
* Flask
* Pillow
* torch
* torchmetrics
* torchvision
* Werkzeug

You can install these dependencies using pip:
  ```sh
  pip install Flask Pillow torch torchmetrics torchvision Werkzeug
  ```

### Initialize Environment

1. Clone the repository:
   ```sh
   git clone https://github.com/sjy-11/pytorch-cat-or-dog-project.git
   ```
2. Navigate to the project directory:
   ```sh
   cd pytorch-cat-or-dog-project
   ```
3. Run the application:
   ```sh
   python app.py
   ```
4. Access the web interface:  
   The local development server will be hosted on port 5000. After running python app.py, you can click on the URL shown in the terminal, or open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
### Usage

Once you have the application running, you can upload image files (jpg or jpeg) through the web interface. Depending on the prediction:
* An illustration of a cat will be shown if the model predicts a cat.
![Cat Prediction][cat-prediction-screenshot]

* An illustration of a dog will be shown if the model predicts a dog.
![Dog Prediction][dog-prediction-screenshot]

## Model Overview

The Convolutional Neural Network (CNN) model used in this project is designed for binary classification. Users can alter the model's architecture in the model.py file and train their models in the train_test.py file.

## Acknowledgments

* [Dogs vs Cats Kaggle Image Dataset](https://www.kaggle.com/datasets/moazeldsokyx/dogs-vs-cats)
* [Cat Image](https://pixabay.com/vectors/siamese-cat-siamese-cat-kitty-48032/)
* [Dog Image](https://pixabay.com/illustrations/dog-animal-art-pet-clip-cartoon-5119683/)

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
[index-screenshot]: images/index_screenshot.png
[cat-prediction-screenshot]: images/cat_screenshot.png
[dog-prediction-screenshot]: images/dog_screenshot.png
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[Pytorch.com]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white 
[Pytorch-url]: https://pytorch.org
[Flask.com]: https://img.shields.io/badge/Flask-000?logo=flask&logoColor=fff&style=for-the-badge
[Flask-url]: https://flask.palletsprojects.com/en/3.0.x
[Python.com]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=for-the-badge
[Python-url]: https://www.python.org