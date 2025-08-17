
#!/usr/bin/env python3
"""
Wine Quality ML Training Pipeline
CI/CD Ready implementation for training and deploying ML models
"""

import os
import sys
import json
import joblib
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
from pkgutil import zipimporter
# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class WineQualityMLPipeline:
    """
    Complete ML pipeline for wine quality prediction
    Designed for CI/CD deployment
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.metrics = {}
        self.feature_names = []

        # Create output directories
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.artifact_dir = Path(config.get('artifact_dir', 'artifacts'))
        self.report_dir = Path(config.get('report_dir', 'reports'))

        for directory in [self.model_dir, self.artifact_dir, self.report_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate the dataset"""
        try:
            file_path="winedata.csv"
            logger.info(f"Loading data from {file_path}")

            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError("Could not read file with any encoding")

            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            # Data validation
            if df.empty:
                raise ValueError("Dataset is empty")

            if df.isnull().sum().sum() > len(df) * 0.3:  # More than 30% missing
                logger.warning("Dataset has more than 30% missing values")

            # Log basic statistics
            logger.info(f"Missing values per column:\n{df.isnull().sum()}")
            logger.info(f"Data types:\n{df.dtypes}")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for training"""
        logger.info("Starting data preprocessing")

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))

        # Identify target column (assuming 'quality' is the target)
        target_columns = ['quality', 'target', 'label', 'class']
        target_col = None

        for col in target_columns:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            # If no explicit target, assume last column is target
            target_col = df.columns[-1]
            logger.warning(f"No standard target column found, using '{target_col}' as target")

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Store feature names
        self.feature_names = list(X.columns)
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Target column: {target_col}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        # Convert to classification problem if needed
        if y.dtype in ['float64', 'float32'] and len(y.unique()) > 10:
            # Convert continuous quality scores to categories
            y_binned = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
            y = y_binned
            logger.info("Converted continuous target to categorical (Low, Medium, High)")

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Preprocessing completed. Final shapes - X: {X_scaled.shape}, y: {y_encoded.shape}")
        return X_scaled, y_encoded

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train multiple models and select the best one"""
        logger.info("Starting model training")

        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=42,
                probability=True
            )
        }

        # Hyperparameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
            },
            'GradientBoosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear']
            }
        }

        best_score = 0
        best_model_name = None
        model_results = {}

        for name, model in models.items():
            logger.info(f"Training {name}")

            try:
                # Grid search with cross-validation
                if self.config.get('enable_hyperparameter_tuning', True):
                    grid_search = GridSearchCV(
                        model,
                        param_grids[name],
                        cv=5,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)

                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    score = grid_search.best_score_
                else:
                    # Simple training without hyperparameter tuning
                    model.fit(X_train, y_train)
                    score = cross_val_score(model, X_train, y_train, cv=5).mean()
                    best_model = model
                    best_params = model.get_params()

                model_results[name] = {
                    'model': best_model,
                    'score': score,
                    'params': best_params
                }

                logger.info(f"{name} - CV Score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model_name = name
                    self.model = best_model

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue

        if self.model is None:
            raise ValueError("No model could be trained successfully")

        logger.info(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
        return model_results

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model"""
        logger.info("Evaluating model performance")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }

        # ROC AUC for multiclass
        try:
            if len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            metrics['roc_auc'] = None

        self.metrics = metrics

        # Log metrics
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            if value is not None:
                logger.info(f"{metric.capitalize()}: {value:.4f}")

        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        # Save detailed report
        report_path = self.report_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=2)

        return metrics

    def generate_visualizations(self, X_test: np.ndarray, y_test: np.ndarray):
        """Generate model performance visualizations"""
        logger.info("Generating visualizations")

        try:
            # Predictions
            y_pred = self.model.predict(X_test)

            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(self.artifact_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Feature Importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                plt.figure(figsize=(10, 8))
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=True)

                plt.barh(importance_df['feature'], importance_df['importance'])
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(self.artifact_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()

            logger.info("Visualizations saved successfully")

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")

    def save_model(self) -> str:
        """Save the trained model and preprocessing components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"wine_quality_model_{timestamp}.joblib"
        model_path = self.model_dir / model_filename

        # Create model package
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'timestamp': timestamp
        }

        # Save model package
        joblib.dump(model_package, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        metadata = {
            'model_path': str(model_path),
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'classes': self.label_encoder.classes_.tolist(),
            'metrics': self.metrics,
            'training_timestamp': timestamp
        }

        metadata_path = self.artifact_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save latest model symlink for production
        latest_model_path = self.model_dir / "latest_model.joblib"
        if latest_model_path.exists():
            latest_model_path.unlink()
        latest_model_path.symlink_to(model_filename)

        return str(model_path)

    def validate_model_quality(self) -> bool:
        """Validate if model meets quality thresholds for deployment"""
        min_accuracy = self.config.get('min_accuracy_threshold', 0.6)
        min_f1 = self.config.get('min_f1_threshold', 0.5)

        accuracy_ok = self.metrics.get('accuracy', 0) >= min_accuracy
        f1_ok = self.metrics.get('f1_score', 0) >= min_f1

        quality_passed = accuracy_ok and f1_ok

        logger.info(f"Model Quality Validation:")
        logger.info(f"Accuracy: {self.metrics.get('accuracy', 0):.4f} >= {min_accuracy}: {accuracy_ok}")
        logger.info(f"F1 Score: {self.metrics.get('f1_score', 0):.4f} >= {min_f1}: {f1_ok}")
        logger.info(f"Quality Check: {'PASSED' if quality_passed else 'FAILED'}")

        return quality_passed


def main():
    """Main training pipeline function"""
    parser = argparse.ArgumentParser(description='Wine Quality ML Training Pipeline')
    parser.add_argument('--data-path', required=True, help='winedata.csv')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--no-hyperparameter-tuning', action='store_true', help='Disable hyperparameter tuning')

    args = parser.parse_args()

    # Load configuration
    config = {
        'model_dir': 'models',
        'artifact_dir': 'artifacts',
        'report_dir': 'reports',
        'min_accuracy_threshold': 0.6,
        'min_f1_threshold': 0.5,
        'enable_hyperparameter_tuning': not args.no_hyperparameter_tuning
    }

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))

    try:
        # Initialize pipeline
        logger.info("Starting Wine Quality ML Training Pipeline")
        pipeline = WineQualityMLPipeline(config)

        # Load and preprocess data
        df = pipeline.load_and_validate_data(args.data_path)
        X, y = pipeline.preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        # Train models
        model_results = pipeline.train_models(X_train, y_train)

        # Evaluate model
        metrics = pipeline.evaluate_model(X_test, y_test)

        # Generate visualizations
        pipeline.generate_visualizations(X_test, y_test)

        # Validate model quality
        quality_passed = pipeline.validate_model_quality()

        if quality_passed:
            # Save model
            model_path = pipeline.save_model()
            logger.info(f"Model training completed successfully. Model saved to: {model_path}")

            # Exit code for CI/CD
            sys.exit(0)
        else:
            logger.error("Model did not pass quality validation")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
