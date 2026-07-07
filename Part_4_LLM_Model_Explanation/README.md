# Part 4 — LLM-Powered Model Prediction Explanation

## Selected Feature Track

**Track C — Model Prediction Explanation Pipeline**

Part 4 extends the machine learning workflow developed in Parts 1–3 by adding an LLM-powered explanation layer to the best-performing classification model.

The serialized Random Forest pipeline from Part 3 was loaded using `joblib.load("best_model.pkl")`. For selected feature-vector inputs, the model generates a predicted class and predicted probability. These prediction results and the corresponding feature values are supplied to an LLM, which produces a structured JSON explanation.

The objective is to convert numerical machine learning predictions into structured and human-readable explanations while maintaining schema validation, reproducibility, and basic safety guardrails.

---

## LLM API Connection

The LLM API key is stored in an environment variable and is never hardcoded in the notebook.

The reusable `call_llm()` function constructs a JSON payload containing the model name, system and user messages, temperature, and maximum token limit. The request is sent using an HTTP POST request with JSON content headers and Bearer-token authentication.

The function checks the HTTP response status before parsing the returned JSON response.

A simple test prompt requesting the word `hello` was used to verify the API connection.

The returned response was:

```text
hello
```

This confirmed that the LLM API connection was configured successfully.

---

## Prompt Design

### Exact System Prompt

```text
You are a machine learning prediction explanation assistant.

Your task is to explain a house-price classification prediction using
the supplied feature values, predicted class, and predicted probability.

Return ONLY valid JSON. Do not include Markdown, code fences, or any
additional text outside the JSON object.

The JSON object must contain exactly these five scalar fields:

{
  "prediction_label": "string",
  "confidence_level": "low|medium|high",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}

Interpret predicted class 1 as a house with SalePrice greater than the
median SalePrice and predicted class 0 as a house with SalePrice less
than or equal to the median SalePrice.

Base the explanation only on the supplied information. Do not invent
feature values or personal information.

Confidence level must be:
- high when predicted probability is at least 0.80
- medium when predicted probability is at least 0.60 but below 0.80
- low when predicted probability is below 0.60
```

### Exact User Prompt Template

```text
Explain the following machine learning prediction.

Feature values:
{feature_values}

Predicted class:
{predicted_class}

Predicted probability:
{predicted_probability}

Return the required JSON explanation only.
```

The primary explanation pipeline uses `temperature=0.0`.

A temperature of zero was selected because structured JSON generation benefits from deterministic and reproducible output. Lower temperature makes the model consistently select high-probability tokens, reducing unnecessary variation in JSON structure and explanation wording.

---

## Structured JSON Output Schema

Every LLM explanation is required to contain exactly five scalar fields:

| Field | Type | Description |
|---|---|---|
| `prediction_label` | string | Description of the predicted house-price class |
| `confidence_level` | string | Low, medium, or high prediction confidence |
| `top_reason` | string | Primary reason for the prediction |
| `second_reason` | string | Secondary reason for the prediction |
| `next_step` | string | Suggested next action |

The schema requires all five fields and does not allow additional properties.

Each LLM response is stripped of surrounding whitespace and parsed using `json.loads()`. The parsed JSON object is then validated using `jsonschema.validate()`.

The validation pipeline explicitly handles `json.JSONDecodeError` and `jsonschema.ValidationError`.

If JSON parsing or schema validation fails, a fallback dictionary is returned with all five required fields set to `null`. The validation error is also printed.

This prevents malformed LLM output from breaking the complete prediction-explanation pipeline.

---

## Model Prediction Explanation Pipeline

The best-performing model from Part 3 was loaded using:

```python
best_model_part4 = joblib.load("best_model.pkl")
```

Three feature-vector inputs from the prepared classification test set were selected.

For every input, the pipeline performs the following sequence:

1. Convert the feature dictionary into a one-row DataFrame.
2. Align the input columns with the model training features.
3. Generate the predicted class using `.predict()`.
4. Generate the class probability using `.predict_proba()`.
5. Construct the structured LLM user prompt.
6. Apply the PII guardrail.
7. Call the LLM.
8. Parse the returned text using `json.loads()`.
9. Validate the parsed object using `jsonschema.validate()`.
10. Return the validated explanation or a fallback JSON object.

This creates a complete flow from tabular feature input to model prediction and finally to an LLM-generated structured explanation.

---

## PII Guardrail

A regular-expression-based PII guardrail is applied before every LLM API call.

The guardrail checks the user input for email addresses and phone numbers.

If PII is detected, the LLM API is not called. Instead, the system prints:

```text
Input blocked: PII detected.
```

and returns `None`.

### Guardrail Test Results

| Test Input | Result |
|---|---|
| Input containing an email address | Blocked |
| Clean house-price classification input | Allowed |

The email-containing test input was correctly blocked, while the clean input was allowed to proceed to the LLM API.

This demonstrates that the safety check is applied before transmitting user input to the external LLM service.

---

## Temperature A/B Comparison

Each of the three prediction inputs was evaluated using both `temperature=0.0` and `temperature=0.7`.

| Input | Temperature 0.0 | Temperature 0.7 | Key Difference |
|---|---|---|---|
| Input 1 | High-confidence class 1 explanation based on overall quality and living area | High-confidence class 1 explanation based on the same major features | Similar reasoning with slightly different wording |
| Input 2 | Low-confidence lower-price classification emphasizing quality and condition | Low-confidence class 0 explanation emphasizing overall and exterior quality | Same prediction interpretation with different explanation phrasing |
| Input 3 | High-confidence class 1 explanation emphasizing overall quality and lot area | High-confidence class 1 explanation emphasizing the same major features | Similar reasoning with minor wording variation |

At `temperature=0.0`, the LLM produced more deterministic and consistent structured explanations.

At `temperature=0.7`, the core prediction interpretation remained similar, but the wording and phrasing of the reasons and recommended next steps varied.

This occurs because `temperature=0` strongly favors the highest-probability next token, producing more deterministic outputs. A temperature of `0.7` samples from a broader token distribution and therefore introduces greater variability.

For a structured machine learning explanation pipeline, `temperature=0.0` is preferred because reproducibility and predictable JSON output are more important than linguistic creativity.

---

## Three-Row End-to-End Demonstration

The complete Track C pipeline was executed on three distinct feature-vector inputs.

| Input | LLM Output | Valid JSON | Pass/Block (Guardrail) |
|---|---|---|---|
| Input 1 | Structured house-price prediction explanation | PASS | Pass |
| Input 2 | Structured house-price prediction explanation | PASS | Pass |
| Input 3 | Structured house-price prediction explanation | PASS | Pass |

All three inputs successfully passed the PII guardrail. The LLM returned structured JSON explanations for all three model predictions.

Each response was successfully parsed and validated against the required JSON schema. Therefore, all three end-to-end test cases achieved **PASS** validation status.

---

## Structured Explanation Results

For a high-confidence prediction above the median SalePrice, the LLM identified high overall quality and large living area as major explanatory factors.

For the lower-price prediction, the explanation emphasized below-average overall quality and exterior quality.

For another high-confidence above-median prediction, the explanation highlighted high overall quality and large lot area.

These explanations are consistent with the feature-importance analysis from Part 3, where `Overall Qual`, `Gr Liv Area`, and related property-quality features were among the most influential Random Forest predictors.

---

## Part 4 Pipeline Summary

```text
Feature Vector
      ↓
Input Encoding and Feature Alignment
      ↓
Random Forest Prediction
      ↓
Predicted Class + Predicted Probability
      ↓
Structured LLM Prompt Construction
      ↓
PII Guardrail
      ↓
LLM API Call
      ↓
JSON Parsing
      ↓
JSON Schema Validation
      ↓
Validated Prediction Explanation
```

The pipeline successfully combines traditional machine learning with an LLM-based explanation layer.

The Random Forest model performs the classification, while the LLM translates the model output and supplied feature information into a structured explanation. JSON schema validation ensures that the explanation follows a predictable format, and the PII guardrail provides a basic safety check before external API calls.

---

## Part 4 Repository File

- `Part_4_LLM_Model_Explanation.ipynb` — LLM-powered Track C model prediction explanation pipeline with API integration, prompt engineering, structured JSON validation, temperature comparison, PII guardrails, and three-input end-to-end demonstration.

---

## Part 4 Conclusion

Part 4 successfully implemented an LLM-powered model prediction explanation pipeline using the best-performing Random Forest model from Part 3.

The system loads the serialized model, generates class predictions and probabilities, constructs structured prompts, applies a PII guardrail, calls an LLM API, parses the response, and validates the resulting JSON explanation against a predefined schema.

The temperature experiment demonstrated that `temperature=0.0` provides more deterministic and reproducible explanations, making it more suitable for structured machine learning workflows.

All three demonstration inputs produced valid JSON explanations and passed schema validation. The completed pipeline demonstrates how a conventional tabular machine learning model can be extended with an LLM layer to provide structured and interpretable prediction explanations.