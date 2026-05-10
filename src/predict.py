"""
Animal Image Classification Demo
Using VGG16 feature extraction + Random Forest classifier
"""

import numpy as np
import cv2
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg_preprocess
)


class AnimalClassifier:
    """Animal classifier using VGG16 + Random Forest."""

    CLASSES = [
        'butterfly',
        'cat',
        'chicken',
        'cow',
        'dog',
        'elephant',
        'horse',
        'sheep',
        'spider',
        'squirrel'
    ]

    def __init__(self, modules_path: str):

        self.modules_path = Path(modules_path)

        self.img_size = (224, 224)

        print("\nLoading models...")

        self._load_vgg16_model()
        self._load_scaler()
        self._load_pca()
        self._load_classifier()
        self._load_label_encoder()

        print("\n✓ Classifier initialized successfully!")

    def _load_vgg16_model(self):
        """Load VGG16 feature extractor."""

        base_model = VGG16(
            weights='imagenet',
            include_top=True
        )

        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc2').output
        )

        print("✓ VGG16 model loaded")

    def _load_scaler(self):
        """Load scaler."""

        scaler_path = (
            self.modules_path /
            'vgg16_scaler.pkl'
        )

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print("✓ Scaler loaded")

    def _load_pca(self):
        """Load PCA."""

        pca_path = (
            self.modules_path /
            'vgg16_pca.pkl'
        )

        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)

        print(
            f"✓ PCA loaded "
            f"({self.pca.n_components_} components)"
        )

    def _load_classifier(self):
        """Load Random Forest model."""

        model_path = (
            self.modules_path /
            'vgg16_random_forest_model.pkl'
        )

        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)

        print("✓ Random Forest classifier loaded")

    def _load_label_encoder(self):
        """Load label encoder."""

        encoder_path = (
            self.modules_path /
            'vgg16_label_encoder.pkl'
        )

        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        print("✓ Label encoder loaded")

    def preprocess_image(self, image_path: str):

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(
                f"Cannot read image: {image_path}"
            )

        # Resize
        img = cv2.resize(img, self.img_size)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # VGG16 preprocessing
        img = vgg_preprocess(img)

        return img

    def extract_features(self, image):

        features = self.feature_extractor.predict(
            image,
            verbose=0
        )

        return features

    def predict(self, image_path: str):

        # Preprocess
        image = self.preprocess_image(image_path)

        # Extract features
        features = self.extract_features(image)

        # Scale
        features_scaled = self.scaler.transform(
            features
        )

        # PCA
        features_pca = self.pca.transform(
            features_scaled
        )

        # Predict
        prediction_encoded = self.classifier.predict(
            features_pca
        )

        # Decode label
        prediction_label = (
            self.label_encoder.inverse_transform(
                prediction_encoded
            )[0]
        )

        result = {
            'image_path': image_path,
            'predicted_class': prediction_label,
            'encoded_label': int(
                prediction_encoded[0]
            ),
            'original_feature_shape': features.shape,
            'reduced_feature_shape': features_pca.shape
        }

        # Probability
        if hasattr(self.classifier, 'predict_proba'):

            probabilities = (
                self.classifier.predict_proba(
                    features_pca
                )[0]
            )

            result['confidence'] = float(
                np.max(probabilities)
            )

            result['probabilities'] = {
                self.label_encoder.inverse_transform(
                    [i]
                )[0]: float(p)
                for i, p in enumerate(probabilities)
            }

        return result

    def predict_batch(self, image_paths: list):

        results = []

        for path in image_paths:

            try:

                result = self.predict(path)

                results.append(result)

            except Exception as e:

                results.append({
                    'image_path': path,
                    'error': str(e)
                })

        return results


def display_prediction(result: dict):

    print("\n" + "=" * 50)

    if 'error' in result:

        print(f"Error: {result['error']}")

        return

    print(f"Image: {result['image_path']}")

    print(
        f"Predicted Class: "
        f"{result['predicted_class']}"
    )

    if 'confidence' in result:

        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )

    print(
        f"Feature Dimensions: "
        f"{result['original_feature_shape'][1]} → "
        f"{result['reduced_feature_shape'][1]}"
    )

    if 'probabilities' in result:

        print("\nTop Predictions:")

        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for cls, prob in sorted_probs:

            print(
                f"{cls:12s}: "
                f"{prob * 100:.2f}%"
            )


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description='Animal Image Classification Demo'
    )

    parser.add_argument(
        'image_path',
        type=str,
        nargs='?',
        help='Path to image'
    )

    parser.add_argument(
        '--modules',
        type=str,
        default='../modules',
        help='Path to modules folder'
    )

    args = parser.parse_args()

    modules_path = Path(args.modules)

    if not modules_path.is_absolute():

        modules_path = (
            Path(__file__).parent.parent /
            'modules'
        )

    classifier = AnimalClassifier(
        str(modules_path)
    )

    # Single prediction
    if args.image_path:

        result = classifier.predict(
            args.image_path
        )

        display_prediction(result)

    # Interactive mode
    else:

        print("\nAnimal Classification Demo")

        print(
            "Type image path "
            "or 'quit' to exit"
        )

        while True:

            try:

                image_path = input(
                    "\nImage path: "
                ).strip()

                if image_path.lower() in [
                    'quit',
                    'q',
                    'exit'
                ]:

                    print("Goodbye!")

                    break

                if not image_path:
                    continue

                result = classifier.predict(
                    image_path
                )

                display_prediction(result)

            except KeyboardInterrupt:

                print("\nGoodbye!")

                break

            except Exception as e:

                print(f"Error: {e}")


if __name__ == '__main__':
    main()