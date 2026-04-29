import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. DEFINE PYTORCH MODEL ---
class BreastCancerNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BreastCancerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out

# --- 2. LOAD & PREP DATA (CACHED) ---
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    base = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
    columns = ['id', 'diagnosis'] + [f'{n}1' for n in base] + [f'{n}2' for n in base] + [f'{n}3' for n in base]
    df = pd.read_csv(url, names=columns)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop(columns=['id', 'diagnosis']).values
    y = df['diagnosis'].values
    return X, y, df.columns[2:]

X, y, feature_names = load_data()

st.title("Breast Cancer Responder Classification")

# --- VARIABLES TABLE ---
base_features = [
    ("Radius",           "Mean distance from center to points on the perimeter"),
    ("Texture",          "Standard deviation of gray-scale values"),
    ("Perimeter",        "Cell nucleus perimeter"),
    ("Area",             "Cell nucleus area"),
    ("Smoothness",       "Local variation in radius lengths"),
    ("Compactness",      "Perimeter² / area - 1.0"),
    ("Concavity",        "Severity of concave portions of the contour"),
    ("Concave Points",   "Number of concave portions of the contour"),
    ("Symmetry",         "Symmetry of the cell nucleus"),
    ("Fractal Dimension","Coastline approximation - 1"),
]

base_col_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                  'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

rows = []
for stat_label, suffix in [("Mean", "1"), ("Standard Error", "2"), ("Worst", "3")]:
    for col_name, (name, desc) in zip(base_col_names, base_features):
        rows.append({
            "Column #":     len(rows) + 1,
            "CSV Column":   f"{col_name}{suffix}",
            "Feature Name": f"{name} ({stat_label})",
            "Type":         "Continuous",
            "Description":  desc,
        })

variables_df = pd.DataFrame(rows)

with st.expander("Dataset Variables Reference (30 attributes)", expanded=False):
    st.caption(
        "Ten nuclear characteristics are each measured three ways: "
        "mean, standard error, and worst (largest) value — yielding 30 features. "
        "Use the CSV column names shown below when uploading patient data."
    )
    st.dataframe(variables_df, use_container_width=True, hide_index=True)

# --- 3. SIDEBAR: HYPERPARAMETERS ---
st.sidebar.header("Model Hyperparameters")
epochs = st.sidebar.radio("Epochs", [50, 100, 200], index=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.1, 0.01, 0.001, 0.0001], index=1)
hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", min_value=4, max_value=64, value=16)

split_label = st.sidebar.radio("Train / Test Split", ["70 / 30", "80 / 20", "90 / 10"], index=1)
test_size = {"70 / 30": 0.30, "80 / 20": 0.20, "90 / 10": 0.10}[split_label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Initialize session state to store trained model and scaler
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# --- 4. TRAINING LOOP ---
if st.button("Train Model"):
    st.write(f"Training model with **{epochs} epochs** and **{learning_rate} LR**...")
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Convert to Tensors
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # Initialize Model, Loss (BCE), and Optimizer (Gradient Descent)
    model = BreastCancerNN(input_size=30, hidden_size=hidden_neurons)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # UI Elements for live tracking
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    loss_history = []

    # Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Update UI every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            chart_placeholder.line_chart(loss_history)
            
    st.success("Training Complete!")
    
    # Save model and scaler to session state for inference later
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.model_trained = True

st.markdown("---")

# --- 5. INFERENCE: INPUTTING A SAMPLE ---
st.header("Patient Inference")

def run_inference(values_2d):
    patient_scaled = st.session_state.scaler.transform(values_2d)
    patient_tensor = torch.tensor(patient_scaled, dtype=torch.float32)
    st.session_state.model.eval()
    with torch.no_grad():
        probs = st.session_state.model(patient_tensor).numpy().flatten()

    st.subheader("Prediction Result")
    labels     = ["Malignant" if p >= 0.5 else "Benign" for p in probs]
    confidences = [p if p >= 0.5 else 1 - p for p in probs]

    if len(probs) == 1:
        st.metric(label="Diagnosis", value=labels[0])
        st.write(f"Confidence Score: **{confidences[0]:.2%}**")
    else:
        results = pd.DataFrame({
            "Patient":          range(1, len(probs) + 1),
            "Diagnosis":        labels,
            "Confidence Score": [f"{c:.2%}" for c in confidences],
        })
        st.dataframe(results, use_container_width=True, hide_index=True)

if st.session_state.model_trained:
    X_df = pd.DataFrame(X, columns=feature_names)
    dummy_data = pd.DataFrame([X_test[0]], columns=feature_names)

    tab_csv, tab_manual = st.tabs(["CSV Upload", "Manual Input"])

    # --- TAB 1: CSV UPLOAD ---
    with tab_csv:
        st.write("Upload patient data (CSV with 30 features) to predict malignancy.")

        with st.expander("Expected CSV Format", expanded=True):
            st.caption(
                "Your CSV must have exactly 30 columns with the headers below. "
                "Each row represents one patient. Values should be raw (unscaled) measurements."
            )
            st.dataframe(dummy_data, use_container_width=True)
            st.caption(f"30 columns: {', '.join(feature_names)}")

        csv = dummy_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sample CSV Template",
            data=csv,
            file_name='sample_patient.csv',
            mime='text/csv',
        )

        uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])
        if uploaded_file is not None:
            patient_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:", patient_df)
            run_inference(patient_df.values)

    # --- TAB 2: MANUAL INPUT ---
    with tab_manual:
        st.write("Enter each measurement manually. Defaults are dataset averages.")
        input_values = {}

        for stat_label, suffix in [("Mean Values", "1"), ("Standard Error Values", "2"), ("Worst Values", "3")]:
            with st.expander(stat_label, expanded=(suffix == "1")):
                col_a, col_b = st.columns(2)
                for i, (col_name, (feat_name, desc)) in enumerate(zip(base_col_names, base_features)):
                    col_key = f"{col_name}{suffix}"
                    col_min  = float(X_df[col_key].min())
                    col_max  = float(X_df[col_key].max())
                    col_mean = float(X_df[col_key].mean())
                    with (col_a if i % 2 == 0 else col_b):
                        input_values[col_key] = st.number_input(
                            feat_name,
                            min_value=col_min,
                            max_value=col_max,
                            value=col_mean,
                            format="%.4f",
                            help=desc,
                            key=f"manual_{col_key}",
                        )

        if st.button("Predict"):
            patient_row = [[input_values[col] for col in feature_names]]
            run_inference(patient_row)

else:
    st.info("Please train the model using the controls above before uploading patient data.")