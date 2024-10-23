import pygame
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
test_path = "./average_df.csv"
data = pd.read_csv(test_path)

# Load the pre-trained model
gbm_pickle = joblib.load('./lgbEEG.pkl')

# Encode the 'state' column
label_encoder = LabelEncoder()
data['state'] = label_encoder.fit_transform(data['state'])

# Prepare the data
X = data.drop('state', axis=1)  # Features
y = data['state']  # Labels

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

# Game loop
running = True
clock = pygame.time.Clock()

# Index for sequential input
data_index = 0

# Timer for prediction (every 3 seconds)
PREDICT_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(PREDICT_EVENT, 100)  # 100  milliseconds = 3 seconds

# Function to update ball position based on prediction
def update_ball_position(prediction):
    global ball_y
    
    # Use np.argmax to get the predicted class
    predicted_class = np.argmax(prediction)
    print(f"Predicted class: {predicted_class}")  # Debugging message

    if predicted_class == 1:  # Focused state -> move ball up
        ball_y -= ball_speed
    else:  # Unfocused state -> move ball down
        ball_y += ball_speed

    # Ensure the ball stays within screen bounds
    if ball_y - ball_radius < 0:
        ball_y = ball_radius
    elif ball_y + ball_radius > height:
        ball_y = height - ball_radius
    
    # Debugging message to check ball coordinates
    print(f"Ball position: {ball_x}, {ball_y}")

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

            X_sample = X.iloc[data_index].values.reshape(1, -1)  # Use current row as input
            X_sample = np.ascontiguousarray(X_sample)  # Fix LightGBM memory warning
            prediction = gbm_pickle.predict(X_sample)  # Get prediction probabilities

            # Update the ball's position based on the prediction
            update_ball_position(prediction)

            # Increment the index for the next prediction
            data_index += 1
    
    # Draw the ball
    pygame.draw.circle(screen, BLUE, (ball_x, ball_y), ball_radius)  # Drawing the ball

    # Display the state on the screen
    if 'prediction' in locals():
        predicted_class = np.argmax(prediction)
        state_text = "Focused" if predicted_class == 1 else "Unfocused"
        text = font.render(f"State: {state_text}", True, BLACK)
        screen.blit(text, (20, 20))

    # Update display
    pygame.display.flip()

    # Frame rate
    clock.tick(60)

pygame.quit()
