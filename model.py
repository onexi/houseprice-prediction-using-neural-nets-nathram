import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
tf.config.set_visible_devices([], 'GPU')

# Load dataset
data = pd.read_csv("train.csv")

# Assuming the last column is the target, adjust if needed
target_column = data.columns[-1]
X = data.drop(columns=[target_column, 'Id'])
y = data[target_column]

print(X)

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Identify numerical columns (those not in categorical_columns)
numerical_columns = X.columns.difference(categorical_columns)

# Handle categorical columns with one hot encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Ignore unknown categories during testing
X_encoded = ohe.fit_transform(X[categorical_columns])
X_encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(categorical_columns))

# Scale numerical columns
scaler = StandardScaler()
X_numerical = X[numerical_columns]
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Convert scaled numerical data into a DataFrame with correct column names
X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_columns)

# Combine the scaled numerical columns with the one-hot encoded categorical columns
X_processed = pd.concat([X_encoded_df, X_numerical_scaled_df], axis=1)

print(X_processed)

X_processed = X_processed.fillna(1)

print(X_processed)
print(X)

# Apply cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_losses = []
cv_metrics = []

for train_index, val_index in kf.split(X_processed):
    X_train, X_val = X_processed[train_index], X_processed[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Build neural network model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1 if y.dtype != 'object' else len(set(y)), activation='linear' if y.dtype != 'object' else 'softmax')
    ])

    # Compile model with adjusted learning rate
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mean_squared_error' if y.dtype != 'object' else 'sparse_categorical_crossentropy', 
                  metrics=['mae' if y.dtype != 'object' else 'accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate model
    loss, metric = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold loss: {loss}")
    cv_losses.append(loss)
    cv_metrics.append(metric)

# Print cross-validation results
print(f"Average CV Loss: {np.mean(cv_losses)}")
print(f"Average CV Metric: {np.mean(cv_metrics)}")
