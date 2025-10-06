import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
import joblib
import numpy as np

# Define the root directory and subject range
root_dir = 'D:/Year 4/RP/ISA ne7las/Data'
subject_prefix = 'AB'
start_subject = 6
end_subject = 30
input_folder_name = 'imu'
output_folder_name = 'id'
treadmill_folder_name = 'treadmill'

# Function to check if directory name matches the subject pattern
def is_valid_subject_dir(dir_name):
    if dir_name.startswith(subject_prefix) and dir_name[2:].isdigit():
        subject_num = int(dir_name[2:])
        return start_subject <= subject_num <= end_subject
    return False

# Initialize lists to hold data
all_imu_data = []
all_id_data = []

# List directories in the root directory
subject_dirs = [d for d in os.listdir(root_dir) if is_valid_subject_dir(d)]

if not subject_dirs:
    print('No valid subject directories found')
else:
    for dir_name in subject_dirs:
        subject_dir = os.path.join(root_dir, dir_name)
        treadmill_dir = os.path.join(subject_dir, treadmill_folder_name)
        imu_dir = os.path.join(treadmill_dir, input_folder_name)
        id_dir = os.path.join(treadmill_dir, output_folder_name)

        # Check if the necessary directories exist
        if os.path.exists(imu_dir) and os.path.exists(id_dir):
            imu_files = [f for f in os.listdir(imu_dir) if f.endswith('.csv')]
            id_files = [f for f in os.listdir(id_dir) if f.endswith('.csv')]

            if imu_files and id_files:
                for imu_file, id_file in zip(imu_files, id_files):
                    imu_file_path = os.path.join(imu_dir, imu_file)
                    id_file_path = os.path.join(id_dir, id_file)
                    
                    # Read the IMU and ID CSV files
                    try:
                        imu_data = pd.read_csv(imu_file_path)
                        id_data = pd.read_csv(id_file_path)
                        
                        # Specify the relevant columns for IMU (input) and ID (output)
                        imu_columns = [
                            'foot_Accel_X', 'foot_Accel_Y', 'foot_Accel_Z',
                            'foot_Gyro_X', 'foot_Gyro_Y', 'foot_Gyro_Z',
                            'shank_Accel_X', 'shank_Accel_Y', 'shank_Accel_Z',
                            'shank_Gyro_X', 'shank_Gyro_Y', 'shank_Gyro_Z',
                            'thigh_Accel_X', 'thigh_Accel_Y', 'thigh_Accel_Z',
                            'thigh_Gyro_X', 'thigh_Gyro_Y', 'thigh_Gyro_Z',
                            'trunk_Accel_X', 'trunk_Accel_Y', 'trunk_Accel_Z',
                            'trunk_Gyro_X', 'trunk_Gyro_Y', 'trunk_Gyro_Z'
                        ]
                        id_columns = [
                            'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment',
                            'hip_flexion_l_moment', 'hip_adduction_l_moment', 'hip_rotation_l_moment',
                            'knee_angle_r_moment', 'knee_angle_l_moment',
                            'ankle_angle_r_moment', 'ankle_angle_l_moment'
                        ]
                        
                        imu_data = imu_data[imu_columns]
                        id_data = id_data[id_columns]
                        
                        # Append the data to the lists
                        all_imu_data.append(imu_data)
                        all_id_data.append(id_data)
                    except Exception as e:
                        print(f'Failed to read {imu_file_path} or {id_file_path}: {e}')
            else:
                print(f'No CSV files found in {imu_dir} or {id_dir}')
        else:
            print(f'IMU or ID path does not exist: {imu_dir} or {id_dir}')

# Combine all data
X = pd.concat(all_imu_data, ignore_index=True)
y = pd.concat(all_id_data, ignore_index=True)

# Ensure y has the same number of rows as X
if X.shape[0] != y.shape[0]:
    print(f'Row count mismatch after concatenation: X ({X.shape[0]}) and y ({y.shape[0]})')
else:
    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale the data
    scalerX, scalerY = MinMaxScaler(), MinMaxScaler()
    X_train = scalerX.fit_transform(X_train)
    X_test = scalerX.transform(X_test)
    X_val = scalerX.transform(X_val)
    y_train = scalerY.fit_transform(y_train)
    y_test = scalerY.transform(y_test)
    y_val = scalerY.transform(y_val)

    # Reshape data for CNN (assuming the temporal dimension is 1 for simplicity)
    X_train = X_train.reshape(X_train.shape[0], 6, 4, 1)  # Adjust dimensions if needed
    X_val = X_val.reshape(X_val.shape[0], 6, 4, 1)
    X_test = X_test.reshape(X_test.shape[0], 6, 4, 1)

    # Define the CNN model
    def create_cnn_model(input_shape):
        model = Sequential()
        model.add(Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))  # Linear activation for regression
        model.compile(loss='mse', optimizer=Adam(), metrics=['mse'])
        return model

    # Define the function to train and test the model
    def model_accuracy(X_train, y_train, X_val, y_val, X_test, y_test):
        model = create_cnn_model((6, 4, 1))
        
        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=10, min_lr=0.00001)

        class PrintLearningRate(Callback):
            def on_epoch_begin(self, epoch, logs=None):
                lr = tf.keras.backend.eval(self.model.optimizer.learning_rate)
                print(f'Learning rate for epoch {epoch+1}: {lr}')

        # Create an instance of the custom callback
        print_lr_callback = PrintLearningRate()

        # Train the model
        History = model.fit(X_train, y_train, epochs=1000, verbose=1, validation_data=(X_val, y_val), 
                            callbacks=[print_lr_callback, reduce_lr], batch_size=512)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R-squared (R2):", r2)

        # Plot epochs vs loss
        plt_loss = History.history['loss']
        plt_val_loss = History.history['val_loss']
        plt.plot(plt_loss, label='Training Loss')
        plt.plot(plt_val_loss, label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Access the MSE values from the history
        train_mse = History.history['mse']
        val_mse = History.history['val_mse']

        # Plot the MSE over epochs
        epochs = range(1, len(train_mse) + 1)
        plt.plot(epochs, train_mse, 'b', label='Training MSE')
        plt.plot(epochs, val_mse, 'r', label='Validation MSE')
        plt.title('Training and Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

        # Plot the predicted vs actual values for each joint moment
        for i, col in enumerate(id_columns):
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test[:, i], y_pred[:, i], edgecolors=(0, 0, 0))
            plt.plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'k--', lw=4)
            plt.xlabel(f'Actual {col}')
            plt.ylabel(f'Predicted {col}')
            plt.title(f'Actual vs Predicted for {col}')
            plt.show()

        return model

    # Train and test the model
    trained_model = model_accuracy(X_train, y_train, X_val, y_val, X_test, y_test)

    # Prompt the user to ask if they want to save the model
    save_model = input("Do you want to save the trained model? (yes/no): ").strip().lower()

    if save_model == 'yes':
        trained_model.save("trained_model.h5")
        # Save the scalers as well
        joblib.dump(scalerX, "scalerX.save")
        joblib.dump(scalerY, "scalerY.save")
        print("Model and scalers saved.")
    else:
        print("Model not saved.")
