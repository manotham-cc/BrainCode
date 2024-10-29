import pygame
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the dataset
test_path = "./average_df.csv"
data = pd.read_csv(test_path)

# Load the pre-trained model
xgb_model = joblib.load('./xgb_model.pkl')

# Encode the 'state' column
label_encoder = LabelEncoder()
data['state'] = label_encoder.fit_transform(data['state'])

# Get model feature names, if available
model_features = xgb_model.feature_names if hasattr(xgb_model, 'feature_names') else list(data.drop('state', axis=1).columns)
print(model_features)
# Prepare the dataset to match model features
X = data.drop('state', axis=1)
X = X[model_features]  # Ensure order matches model training
y = data['state']      # Actual labels

# Set up Pygame
pygame.init()

# Game settings
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Meditation Practice Game")

# Ball settings
ball_radius = 20
ball_x = width // 2
ball_y = height // 2
ball_speed = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Font settings for text
font = pygame.font.SysFont(None, 36)

# Game loop settings
running = True
clock = pygame.time.Clock()

# Index for sequential input
data_index = 0

# Timer for prediction (every 3 seconds)
PREDICT_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(PREDICT_EVENT, 200)  # 3000 milliseconds = 3 seconds

# Initialize prediction variable
prediction = None  # Start with no prediction

# Function to update ball position based on prediction
def update_ball_position(prediction):
    global ball_y
    if prediction == 0:  # Focussed state -> move ball up
        ball_y -= ball_speed
    else:  # Unfocussed state -> move ball down
        ball_y += ball_speed

    # Ensure the ball stays within screen bounds
    if ball_y - ball_radius < 0:
        ball_y = ball_radius
    elif ball_y + ball_radius > height:
        ball_y = height - ball_radius

# Game loop
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Make a prediction every 3 seconds
        if event.type == PREDICT_EVENT:
            if data_index >= len(X):
                data_index = 0  # Loop back to the start if we reach the end of the data

            # Prepare the input sample for prediction
            X_sample = xgb.DMatrix([X.iloc[data_index].values], feature_names=model_features)
            prediction = xgb_model.predict(X_sample)  # Get the predicted class directly

            # Get the actual class for display
            actual_class = y.iloc[data_index]
            actual_class_text = "Focussed" if actual_class == 0 else "Unfocussed"  # Update to match new encoding
            
            # Update the ball's position based on the prediction
            update_ball_position(prediction)

            # Increment the index for the next prediction
            data_index += 1
    
    # Draw the ball
    pygame.draw.circle(screen, BLUE, (ball_x, ball_y), ball_radius)

    # Display the state and prediction on the screen
    if prediction is not None:  # Check if prediction has been made
        state_text = "Predicted: Focussed" if prediction == 0 else "Predicted: Unfocussed"
        actual_text = f"Actual: {actual_class_text}"
        prediction_text = font.render(state_text, True, BLACK)
        actual_text_display = font.render(actual_text, True, BLACK)
        
        screen.blit(prediction_text, (20, 20))
        screen.blit(actual_text_display, (20, 60))

    # Update display
    pygame.display.flip()

    # Frame rate
    clock.tick(60)

pygame.quit()
