# SSBU Stage Selector

## Project Overview

The SSBU (Super Smash Bros. Ultimate) Stage Selector is a web application designed to help players select the optimal stage when facing an opponent in a competitive setting. The recommendations are powered by a neural network trained on competitive Smash Ultimate tournament data. 

## Methodology

### Data Preprocessing and Encoding
The data used for training is derived from competitive Smash Ultimate tournament data. The preprocessing involves encoding character and stage names into integers to facilitate neural network training.

The raw tournament data can be found here:
https://ultimategamedata.com/about

### Neural Network Architecture
The neural network comprises an embedding layer followed by two linear layers. The embedding layer helps in capturing the relationships between different characters and stages.

### Training the Neural Network
The neural network is trained using a subset of the data, with a portion reserved for validation. The training involves several epochs where, in each epoch, the model learns to minimize the error using the Adam optimizer and CrossEntropyLoss as the loss function.

### Flask API and Endpoints
The Flask API serves the web application and handles the recommendation logic. The main endpoint `/api/recommend-stage` takes in the names of two characters and returns the recommended stage based on the neural network's prediction.

## File Structure
- `models/stage_select_nn.py`: Contains the neural network definition and the training script.
- `public/app.py`: Defines the Flask API and the routes for the web application.
- `public/static/`: Holds static files such as images and CSS for the web application.
- `public/templates/`: Contains HTML templates for different pages of the web application.
- `data/`: Directory for storing the dataset and trained model files.

## Setup for Local Development (MacOS)
```
$ git clone https://github.com/justin-kleid/Smash_Bros_Stage_Selector.git
$ cd Smash_Bros_Stage_Selector
$ conda env create -f environment.yml
$ conda activate smash
$ cd public
$ python app.py
```
To change the model, the dataset can be downloaded and clean using the utilities/clean_csv.py file. Next, models/stage_select_nn.py will read the csv file stored in /data/cleaned_smash_tourney_data.csv.

## Usage
To use the web application:
1. Navigate to the home page and select your character.
2. Select your opponent's character.
3. Receive the recommended stage based on the neural network's prediction.

To interact with the API directly, make a POST request to `/api/recommend-stage` with a JSON body containing the character and opponent names.

---

Feel free to contribute and enhance the project's capabilities.
