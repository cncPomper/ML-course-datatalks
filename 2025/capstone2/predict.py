import numpy as np
import pandas as pd
import onnxruntime as rt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_preprocessor():
    """Create the same preprocessor used during training."""
    ordinal_cols = ["sleep_quality", "facility_rating", "exam_difficulty"]
    ordinal_categories = [
        ["poor", "average", "good"],      # sleep_quality
        ["low", "medium", "high"],        # facility_rating
        ["easy", "moderate", "hard"]      # exam_difficulty
    ]

    categorical_cols = ["gender", "course", "internet_access", "study_method"]
    numeric_cols = ["age", "study_hours", "class_attendance", "sleep_hours"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )
    
    return preprocessor


def fit_preprocessor_from_training_data(preprocessor):
    """Fit the preprocessor using the original training data."""
    df = pd.read_csv("data/Exam_Score_Prediction.csv")
    df = df.drop("student_id", axis=1)
    X = df.drop("exam_score", axis=1)
    preprocessor.fit(X)
    return preprocessor


def predict(model_path, input_data):
    """
    Make predictions using the ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
        input_data: Dictionary or pandas DataFrame with input features
        
    Returns:
        Predicted exam score
    """
    # Load ONNX model
    session = rt.InferenceSession(model_path)
    
    # Convert input to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Load and fit preprocessor
    preprocessor = load_preprocessor()
    preprocessor = fit_preprocessor_from_training_data(preprocessor)
    
    # Preprocess input
    X_processed = preprocessor.transform(input_df)
    X_processed = X_processed.astype(np.float32)
    
    # Make prediction
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: X_processed})[0]
    
    # Handle different output shapes and convert to scalar
    if isinstance(prediction, np.ndarray):
        prediction = prediction.item() if prediction.ndim == 0 else prediction.flatten()[0]
    return float(prediction)


if __name__ == "__main__":
    # Example usage
    sample_student = {
        "age": 20,
        "gender": "male",
        "course": "b.sc",
        "study_hours": 5.5,
        "class_attendance": 85.0,
        "internet_access": "yes",
        "sleep_hours": 7.0,
        "sleep_quality": "good",
        "study_method": "coaching",
        "facility_rating": "high",
        "exam_difficulty": "moderate"
    }
    
    predicted_score = predict("model.onnx", sample_student)
    print(f"Predicted exam score: {predicted_score:.2f}")
    
    # Example with multiple students
    print("\nPredicting for multiple students:")
    students = pd.DataFrame([
        {
            "age": 18, "gender": "female", "course": "diploma",
            "study_hours": 3.0, "class_attendance": 70.0,
            "internet_access": "yes", "sleep_hours": 6.0,
            "sleep_quality": "average", "study_method": "online videos",
            "facility_rating": "medium", "exam_difficulty": "easy"
        },
        {
            "age": 22, "gender": "male", "course": "bca",
            "study_hours": 8.0, "class_attendance": 95.0,
            "internet_access": "yes", "sleep_hours": 8.0,
            "sleep_quality": "good", "study_method": "coaching",
            "facility_rating": "high", "exam_difficulty": "hard"
        }
    ])
    
    for idx, student in students.iterrows():
        score = predict("model.onnx", student.to_dict())
        print(f"Student {idx + 1}: {score:.2f}")
