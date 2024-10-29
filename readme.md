# EEG Meditation Game Project

A brain-computer interface meditation game that uses EEG signals to control game elements based on user focussed or unfocussed . The project combines machine learning models (XGBoost and LightGBM) with real-time EEG signal processing to create an interactive meditation experience.

## Project Overview

The game uses trained machine learning models to predict use focussed or unfocussed from EEG signals. When users are focused, the in-game ball rises; when unfocused, it falls. This provides immediate visual feedback for meditation practice.

## Repository Structure

- `Braincode_Training_Pipeline.ipynb`: Jupyter notebook containing the ML pipeline
  - Implements XGBoost and LightGBM models
  - Uses Optuna for hyperparameter optimization
  - Includes data preprocessing and model training steps

- `BrainGame.py`: Main game implementation
  - Real-time EEG signal processing
  - Integration with XGBoost model for focus prediction
  - Interactive ball movement based on focus levels

- `BrainVisualizer.py`: EEG signal visualization tool
  - Real-time brain signal monitoring
  - Signal quality visualization
  - Frequency band analysis display

- `requirements.txt`: Project dependencies

- `Models/`
  - `lgbEEG.pkl`: Trained LightGBM model
  - `xgb_model.pkl`: Trained XGBoost model

- `test.ipynb`: Debug and testing notebook

- `average_df`: Sample dataset
  - Can be used for game simulation
  - Suitable for model training

## Installation

1. Clone the repository
```bash
git clone https://github.com/manotham-cc/BrainCode.git
cd BrainCode
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training Pipeline
```bash
jupyter notebook Braincode_Training_Pipeline.ipynb
```

### Running the Game
```bash
python BrainGame.py
```

### Visualizing EEG Signals
```bash
python BrainVisualizer.py
```

## Data Format

The EEG data should be structured with the following columns:
- Time-series EEG readings
- State,F7,F3,P7,O1,O2,P8,AF4

## Model Training

The training pipeline includes:
1. Data preprocessing
2. Feature extraction
3. Model selection (XGBoost/LightGBM)
4. Hyperparameter optimization using Optuna
5. Model evaluation and validation

## Dependencies

Key libraries required:
- XGBoost
- LightGBM
- Optuna
- NumPy
- Pandas
- Pygame (for game visualization)
- Matplotlib (for signal visualization)


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgements

This project was made possible with the support and resources provided by **Brain Code Camp**. We extend our heartfelt thanks to the Brain Code Camp team for their guidance, mentorship, and access to state-of-the-art tools that have been instrumental in developing this EEG Meditation Game Project.  
Their expertise in EEG processing and machine learning has significantly enhanced the quality and scope of this project, and we are grateful for the invaluable learning experience they have provided.

Thank you, Brain Code Camp!

