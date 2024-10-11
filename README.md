## Enhanced LLM Explanations with EDA and SHAP Data

The latest version of ExplainableAI includes support for:
- **Exploratory Data Analysis (EDA)**: Automatically generate insights on missing values, correlations, and more.
- **SHAP Analysis**: Understand the feature importance driving model predictions.

Example usage:
```python
from explainableai import XAIWrapper

# Initialize XAIWrapper and analyze model
xai = XAIWrapper()
results = xai.analyze(X_test, y_test, model)

# Print enhanced LLM explanation
print(results['llm_explanation'])
