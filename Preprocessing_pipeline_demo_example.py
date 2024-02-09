import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Create a DataFrame from the generated data
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
data['target'] = y

# Save the data to a CSV file
data.to_csv('sample_data.csv', index=False)

class DataLoader:
    def __init__(self, data_source):
         """
        Initializes the DataLoader object with the provided data source.

        Parameters:
        - data_source: str
            The file path or URL from which data will be loaded.
        """
        self.data_source = data_source

    def load_data(self):
        """
        Loads data from the specified data source.

        Returns:
        - DataFrame
            A pandas DataFrame containing the loaded data.
        """
        return pd.read_csv(self.data_source)

class DataCleaner:
    def __init__(self):
        pass

    def clean_data(self, data):
        # For simplicity, let's assume we are dropping NaN values
        cleaned_data = data.dropna()
        return cleaned_data

class DataEncoder:
    def __init__(self):
        pass

    def encode_data(self, data):
        # No encoding needed for this demo
        return data

class FeatureScaler:
    def __init__(self):
        pass

    def scale_features(self, data):
        # No scaling needed for this demo
        return data

class FeatureSelector:
    def __init__(self):
        pass

    def select_features(self, data):
        # No feature selection needed for this demo
        return data

class DataSplitter:
    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def split_data(self, data):
        X = data.drop(columns=['target'])
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test

class PipelineController:
    def __init__(self, data_source):
        self.data_source = data_source
        self.data_loader = DataLoader(data_source)
        self.data_cleaner = DataCleaner()
        self.data_encoder = DataEncoder()
        self.feature_scaler = FeatureScaler()
        self.feature_selector = FeatureSelector()
        self.data_splitter = DataSplitter()

    def run_pipeline(self):
        # Load data
        data = self.data_loader.load_data()

        # Clean data
        cleaned_data = self.data_cleaner.clean_data(data)

        # Encode data
        encoded_data = self.data_encoder.encode_data(cleaned_data)

        # Scale features
        scaled_data = self.feature_scaler.scale_features(encoded_data)

        # Select features
        selected_features = self.feature_selector.select_features(scaled_data)

        # Split data
        X_train, X_test, y_train, y_test = self.data_splitter.split_data(selected_features)

        return X_train, X_test, y_train, y_test

# Execute the pipeline
pipeline = PipelineController(data_source='sample_data.csv')
X_train, X_test, y_train, y_test = pipeline.run_pipeline()

print("Data preprocessing pipeline executed successfully.")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
