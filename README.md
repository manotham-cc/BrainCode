# BrainCode
EEG Meditation Practice Game
This project is a meditation-focused game powered by EEG data and machine learning. Using EEG readings, the game dynamically adjusts based on the user's state of focus or distraction, aiming to enhance mental clarity and mindfulness.

Table of Contents
Overview
Features
Getting Started
Installation
Usage
Model Training
Game Logic
Requirements
Contributing
License
Overview
This project uses real-time EEG data to determine if a player is in a "focussed" or "unfocussed" state. A machine learning model trained on EEG data identifies these states, and the game's mechanics change accordingly, offering visual feedback to encourage the player to maintain focus.

Features
Real-time EEG state prediction: Predicts "focussed" or "unfocussed" states based on EEG readings.
Dynamic game adjustments: The ball moves up or down on the screen, providing immediate feedback on mental state.
Pre-trained XGBoost Model: Uses a trained XGBoost model for state prediction.
User-friendly Pygame interface: Simple and engaging interface with adjustable settings.
Getting Started
Prerequisites
Ensure you have the following tools and libraries installed:

Python 3.8+
Pygame
XGBoost
LightGBM
Scipy
MNE (for EEG data processing)
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/eeg-meditation-game.git
cd eeg-meditation-game
Install Required Packages:

bash
Copy code
pip install -r requirements.txt
Prepare the EEG Data:

Load and preprocess your EEG dataset in CSV format as outlined in average_df.csv (or use your custom dataset).
Train the Model (optional):

If you want to train your model, refer to Model Training below.
Usage
Run the Game:

bash
Copy code
python BrainGame.py
Gameplay:

The game begins with a ball centered on the screen.
Based on the EEG reading's predicted state, the ball moves:
Up when the player is focused.
Down when the player is unfocused.
The player receives visual feedback on their predicted mental state (displayed as "Focussed" or "Unfocussed").
Game Controls
Exit: Close the window or press the quit button in Pygame.
Model Training
Preprocess the Data:

Use your EEG dataset to create focussed and unfocussed state labels.
Apply necessary filtering using libraries such as MNE and Scipy.
Train the Model:

Train a classifier (e.g., XGBoost, LightGBM) to predict the focussed vs. unfocussed states based on EEG signals.
Save the Model:

After training, save the model using joblib for use in the game:
python
Copy code
import joblib
joblib.dump(your_model, "xgb_model.pkl")
Game Logic
EEG Data Input: Uses sequential EEG data to predict the current state.
Prediction Frequency: Predictions are made every 3 seconds during gameplay.
Feedback Mechanism: The game updates the ball’s position based on the predicted mental state, providing immediate visual feedback.
Requirements
All dependencies are listed in requirements.txt. Install them with:

bash
Copy code
pip install -r requirements.txt
Contributing
Fork the project.
Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.
License
This project is licensed under the MIT License. See LICENSE for more information.
