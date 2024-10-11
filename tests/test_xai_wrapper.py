import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from explainableai.core import XAIWrapper
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@pytest.fixture
def sample_data():
    logger.debug("Generating sample data...")
    try:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
        feature_names = [f'feature_{i}' for i in range(20)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        logger.info("Sample data generated.")
        return X, y
    except Exception as e:
        logger.error(f"Some error occurred while generating sample data: {str(e)}")

@pytest.fixture
def sample_models():
    logger.debug("Defining sample models...")
    try:
        return {
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'XGBoost': XGBClassifier(n_estimators=10, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        }
    except Exception as e:
        logger.error(f"Some error occured while defining sample models: {str(e)}")
def test_xai_wrapper_initialization(sample_data, sample_models):
    logger.debug("Testing XAIWrapper initialization...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        assert xai is not None
        assert xai.model is None
        assert xai.X is None
        assert xai.y is None
        logger.info("XAIWrapper initialization test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper initialization: {str(e)}")

def test_xai_wrapper_fit(sample_data, sample_models):
    logger.debug("Testing XAIWrapper fit method...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        assert xai.model is not None
        assert xai.X is not None
        assert xai.y is not None
        assert hasattr(xai.model, 'predict')
        assert hasattr(xai.model, 'predict_proba')
        logger.info("XAIWrapper fit method test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper fit method: {str(e)}")

@pytest.mark.parametrize("model_name", ['Random Forest', 'Logistic Regression', 'XGBoost', 'Neural Network'])
def test_xai_wrapper_analyze_with_different_models(sample_data, sample_models, model_name):
    logger.debug(f"Testing XAIWrapper analyze method with {model_name}...")
    try:
        X, y = sample_data
        models = {model_name: sample_models[model_name]}
        xai = XAIWrapper()
        xai.fit(models, X, y)
        results = xai.analyze()
        assert 'model_performance' in results
        assert 'feature_importance' in results
        assert 'shap_values' in results
        assert 'cv_scores' in results
        assert 'llm_explanation' in results
        assert 'model_comparison' in results
        logger.info(f"XAIWrapper analyze method test with {model_name} passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper analyze method with {model_name}: {str(e)}")

def test_xai_wrapper_predict(sample_data, sample_models):
    logger.debug("Testing XAIWrapper predict method...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        
        # Test single prediction
        single_input = X.iloc[0].to_dict()
        prediction, probabilities, explanation = xai.explain_prediction(single_input)
        assert isinstance(prediction, (int, np.integer))
        assert isinstance(probabilities, np.ndarray)
        assert isinstance(explanation, str)
        logger.info("XAIWrapper predict method test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper predict method: {str(e)}")

def test_xai_wrapper_generate_report(sample_data, sample_models, tmp_path):
    logger.debug("Testing XAIWrapper generate report...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        xai.analyze()
        
        report_path = tmp_path / "test_report.pdf"
        xai.generate_report(filename=str(report_path))
        assert report_path.exists()
        assert os.path.getsize(report_path) > 0  # Check if the file is not empty
        logger.info("XAIWrapper generate report test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper generate report: {str(e)}")

def test_xai_wrapper_perform_eda(sample_data):
    logger.debug("Testing XAIWrapper perform eda...")
    try:
        X, y = sample_data
        df = pd.concat([X, y], axis=1)
        try:
            XAIWrapper.perform_eda(df)
        except Exception as e:
            pytest.fail(f"perform_eda raised an exception: {e}")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper perform data: {str(e)}")

def test_xai_wrapper_feature_importance(sample_data, sample_models):
    logger.debug("Testing XAIWrapper feature importance...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        results = xai.analyze()
        assert 'feature_importance' in results
        assert len(results['feature_importance']) == X.shape[1]
        assert all(isinstance(importance, (float, np.float64)) for importance in results['feature_importance'].values())
        logger.info("XAIWrapper feature importance test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper feature importance: {str(e)}")

def test_xai_wrapper_cross_validation(sample_data, sample_models):
    logger.debug("Testing XAIWrapper cross validation...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        results = xai.analyze()
        assert 'cv_scores' in results
        assert len(results['cv_scores']) == 2  # mean and std
        assert all(isinstance(score, (float, np.float64)) for score in results['cv_scores'])
        logger.info("XAIWrapper cross validation test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper cross validation: {str(e)}")

def test_xai_wrapper_model_comparison(sample_data, sample_models):
    logger.debug("Testing XAIWrapper model comparision...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        results = xai.analyze()
        assert 'model_comparison' in results
        assert len(results['model_comparison']) == len(sample_models)
        for model_name, scores in results['model_comparison'].items():
            assert 'cv_score' in scores
            assert 'test_score' in scores
            assert isinstance(scores['cv_score'], (float, np.float64))
            assert isinstance(scores['test_score'], (float, np.float64))
        logger.info("XAIWrapper model comparision test passed.")
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper model comparision: {str(e)}")

@pytest.mark.parametrize("invalid_input", [
    {},  # Empty dictionary
    {'invalid_feature': 1},  # Invalid feature name
    {f'feature_{i}': 'invalid' for i in range(20)},  # Invalid data type
])
def test_xai_wrapper_predict_invalid_input(sample_data, sample_models, invalid_input):
    logger.debug("Testing XAIWrapper predict invalid input...")
    try:
        X, y = sample_data
        xai = XAIWrapper()
        xai.fit(sample_models, X, y)
        with pytest.raises(Exception):
            xai.explain_prediction(invalid_input)
    except Exception as e:
        logger.error(f"An error occurred while testing XAIWrapper predict invalid input: {str(e)}")

import pandas as pd
import shap
from pandas_profiling import ProfileReport

class XAIWrapper:
    def __init__(self):
        # Your existing initialization code
        pass

    def fit(self, model, X_train, y_train):
        # Fit the model (unchanged)
        model.fit(X_train, y_train)
        self.model = model

    def analyze(self, X_test, y_test, model):
        # Perform EDA
        eda_insights = self.perform_eda(X_test)

        # Perform SHAP Analysis
        shap_summary = self.perform_shap_analysis(X_test, model)

        # Get model performance
        model_performance = self.evaluate_model(model, X_test, y_test)

        # Generate the LLM prompt with additional EDA and SHAP data
        prompt = self.generate_llm_prompt(eda_insights, shap_summary, model_performance)

        # Get LLM explanation using the enhanced prompt
        llm_explanation = self.get_llm_explanation(prompt)

        return {"llm_explanation": llm_explanation}

    def perform_eda(self, X):
        # Perform EDA using pandas-profiling or another EDA library
        eda_report = ProfileReport(X, title="EDA Report", explorative=True)
        eda_report.to_file("eda_report.html")  # Optional: Generate an HTML report

        # Extract important insights like missing values and correlations
        eda_insights = {
            "missing_values": X.isnull().sum().sum(),
            "correlations": X.corr().to_dict(),
        }

        # Convert insights to a text-based summary for LLM prompt
        eda_summary = f"The dataset has {eda_insights['missing_values']} missing values. Correlations are: {eda_insights['correlations']}."
        return eda_summary

    def perform_shap_analysis(self, X, model):
        # Create a SHAP explainer and calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Generate SHAP summary for feature importance
        shap_feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(shap_values).mean(axis=0)
        }).sort_values(by='importance', ascending=False)

        # Create a SHAP-based explanation for the LLM prompt
        shap_summary = f"The most important feature is {shap_feature_importance.iloc[0]['feature']} with an average SHAP value of {shap_feature_importance.iloc[0]['importance']}."
        return shap_summary

    def evaluate_model(self, model, X_test, y_test):
        # Evaluate model performance (accuracy, precision, etc.)
        accuracy = model.score(X_test, y_test)
        # Other metrics can be added as needed (precision, recall, etc.)
        return {
            'accuracy': accuracy,
            # 'precision': ...,
            # 'recall': ...,
        }

    def generate_llm_prompt(self, eda_insights, shap_summary, model_performance):
        # Create an LLM prompt that includes EDA, SHAP, and model performance data
        prompt = f"""
        Based on the dataset's analysis:
        - {eda_insights}
        - {shap_summary}

        The model achieved an accuracy of {model_performance['accuracy']}.
        Please provide an explanation of the model's behavior and key insights.
        """
        return prompt

    def get_llm_explanation(self, prompt):
        # Call the LLM service with the generated prompt (Placeholder: replace with actual LLM API call)
        llm_explanation = f"LLM Explanation based on the following prompt: {prompt}"
        return llm_explanation

    def generate_report(self, filename):
        # Optional: Add a function to generate PDF reports
        with open(filename, 'w') as f:
            f.write("Generated report content here.")
