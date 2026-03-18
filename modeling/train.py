"""
Train 1D CNN for reference interval estimation.

Usage:
    python train.py [--input_path ../data/simulated/ --output_path ./model/]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import RobustScaler

from utils import extract_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../data/simulated/')
    parser.add_argument('--output_path', type=str, default='./model/')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # load data
    x_train = pickle.load(open(os.path.join(args.input_path, 'x_train.pkl'), 'rb'))
    y_train = pickle.load(open(os.path.join(args.input_path, 'y_train.pkl'), 'rb'))
    sizes_train = pickle.load(open(os.path.join(args.input_path, 'sizes_train.pkl'), 'rb'))
    x_val = pickle.load(open(os.path.join(args.input_path, 'x_test.pkl'), 'rb'))
    y_val = pickle.load(open(os.path.join(args.input_path, 'y_test.pkl'), 'rb'))
    sizes_val = pickle.load(open(os.path.join(args.input_path, 'sizes_test.pkl'), 'rb'))

    # standardize
    x_train = [np.array(i) for i in x_train]
    x_val = [np.array(i) for i in x_val]
    y_train = [(i - j.mean()) / j.std() for i, j in zip(y_train, x_train)]
    y_val = [(i - j.mean()) / j.std() for i, j in zip(y_val, x_val)]
    x_train = [(i - i.mean()) / i.std() for i in x_train]
    x_val = [(i - i.mean()) / i.std() for i in x_val]

    # add size prediction target
    y_train = np.array(y_train)
    y_train = np.hstack([
        y_train,
        np.array([i[0] / sum(i) for i in sizes_train]).reshape(-1, 1)
    ])
    y_val = np.array(y_val)
    y_val = np.hstack([
        y_val,
        np.array([i[0] / sum(i) for i in sizes_val]).reshape(-1, 1)
    ])

    # extract features
    x_train = np.array([extract_features(i) for i in x_train])
    x_val = np.array([extract_features(i) for i in x_val])

    # standardize targets
    scalery = RobustScaler()
    y_train = scalery.fit_transform(y_train)
    y_val = scalery.transform(y_val)

    # define model
    inputs = tf.keras.layers.Input(shape=x_train.shape[1:])
    x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=7, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    o = tf.keras.layers.Dense(y_val.shape[1], activation='linear')(x)
    model = tf.keras.models.Model(inputs, o)

    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam()
    )
    model.summary()

    # train
    checkpoint_path = os.path.join(args.output_path, 'model.keras')
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint_callback],
        verbose=2
    )

    # save scaler and history
    with open(os.path.join(args.output_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scalery, f)
    with open(os.path.join(args.output_path, 'model_history.npy'), 'wb') as f:
        pickle.dump(history.history, f)

    print("Training complete.")


if __name__ == '__main__':
    main()
