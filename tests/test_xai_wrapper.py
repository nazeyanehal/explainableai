import unittest
from explainableai import XAIWrapper
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class TestXAIWrapper(unittest.TestCase):
    def setUp(self):
        # Load dataset and initialize the XAIWrapper
        X, y = load_iris(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test = X_test
        self.y_test = y_test
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.xai = XAIWrapper()

    def test_llm_explanation_with_eda_and_shap(self):
        # Analyze the model and get LLM explanation
        results = self.xai.analyze(self.X_test, self.y_test, self.model)

        # Assert that the explanation contains EDA and SHAP data
        self.assertIn('missing values', results['llm_explanation'])
        self.assertIn('SHAP', results['llm_explanation'])

if __name__ == '__main__':
    unittest.main()
