import os
import pickle
import numpy as np


def extract_features(data):
    """Create histogram features from standardized data."""
    features = np.histogram(data, np.linspace(-4, 4, 101), density=True)[0]
    features = (features - features.min()) / (features.max() - features.min())
    return features.reshape(-1, 1)


def load_model(model_path):
    """Load trained CNN model and y_scaler from model directory."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(model_path, 'model.keras'))
    scalery = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))
    return model, scalery


def standardize_samples(data_list, filter_positive=False):
    """Standardize each sample to zero mean and unit variance.

    Args:
        data_list: list of arrays
        filter_positive: if True, keep only values > 0 before standardizing

    Returns (data_scaled, means, stds).
    """
    data_list = [np.array(i) for i in data_list]
    if filter_positive:
        data_list = [i[i > 0] for i in data_list]
    means = [i.mean() for i in data_list]
    stds = [i.std() for i in data_list]
    data_scaled = [(i - i.mean()) / i.std() for i in data_list]
    return data_scaled, means, stds


def predict_ris(model, scalery, data_scaled, ri_indices=(1, -3)):
    """Run model prediction and extract reference intervals.

    Args:
        model: trained keras model
        scalery: fitted y scaler
        data_scaled: list of standardized samples
        ri_indices: tuple of indices into predicted quantiles for lower/upper RI

    Returns (predictions, predicted_ris) where predictions is the full output
    and predicted_ris is the extracted RI pairs.
    """
    features = np.array([extract_features(i) for i in data_scaled])
    test_p = model.predict(features)
    test_p = scalery.inverse_transform(test_p)
    test_p_ris = [i[list(ri_indices)] for i in test_p]
    return test_p, test_p_ris
