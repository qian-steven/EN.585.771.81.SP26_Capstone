import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# make a neural network with external parameters
class BreastCancerNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(BreastCancerNN, self).__init__()
        self.layer1  = nn.Linear(input_size, hidden_size)
        self.bn1     = nn.BatchNorm1d(hidden_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layer2  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out 

# load breast cancer data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    base = ['radius', 'texture', 'perimeter', 'area', 'smoothness','compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'] # 10 features with 3 measurements
    columns = ['id', 'diagnosis'] + [f'{n}1' for n in base] + [f'{n}2' for n in base] + [f'{n}3' for n in base]
    df = pd.read_csv(url, names=columns)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop(columns=['id', 'diagnosis']).values # X = the 30 features
    y = df['diagnosis'].values # y = diagnosis (benign or malignant), ID doesn't get used
    return X, y, df.columns[2:]

X, y, feature_names = load_data()

st.title("Breast Cancer Neural Network Model + Predictor")

# description of the dataset that's being trained in the model
st.markdown(
"""
This app trains a neural network to predict whether identified breast tumors are **malignant** or **benign** using the
[Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
from the UCI Machine Learning Repository. The dataset contains 569 patient samples, each described
by 30 numeric features derived from digitized fine needle aspirate (FNA) images of breast masses —
measuring characteristics of the cell nuclei such as radius, texture, smoothness, and symmetry.

The model is a fully connected neural network with one hidden layer. It applies batch normalization
after the hidden layer to stabilize training, dropout for regularization, and outputs a single value
representing the probability of malignancy. It is trained using binary cross-entropy with logits loss
and the Adam optimizer. Input features are standardized before training using a standard scaler fit
on the training set only.

---

#### Training Parameters

| Parameter | Description |
|---|---|
| **Epochs** | Number of full passes through the training data. More epochs allow more learning but risk overfitting. |
| **Learning Rate** | Controls how large each weight update step is. Lower values train more slowly but more stably. Increase Epochs when lowering the training rate. |
| **Hidden Layer Neurons** | Width of the hidden layer — more neurons increase model capacity. 8-32 is the recommended range for this model.|
| **Train / Test Split** | Fraction of data held out for evaluating generalization. |
| **Dropout Rate** | Fraction of neurons randomly disabled during each training step. Acts as regularization — prevents the model from memorizing training data. 0.0 disables dropout. |

---
""")

# description of all the features
# taken from: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
base_features = [
    ("Radius", "Mean distance from center to points on the perimeter"),
    ("Texture", "Standard deviation of gray-scale values"),
    ("Perimeter", "Cell nucleus perimeter"),
    ("Area", "Cell nucleus area"),
    ("Smoothness", "Local variation in radius lengths"),
    ("Compactness", "Perimeter² / area - 1.0"),
    ("Concavity", "Severity of concave portions of the contour"),
    ("Concave Points", "Number of concave portions of the contour"),
    ("Symmetry", "Symmetry of the cell nucleus"),
    ("Fractal Dimension", "Coastline approximation - 1"),
]

base_col_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness','compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

rows = []
for stat_label, suffix in [("Mean", "1"), ("Standard Error", "2"), ("Worst", "3")]:
    for col_name, (name, desc) in zip(base_col_names, base_features):
        rows.append({
            "Column #": len(rows) + 1,
            "CSV Column": f"{col_name}{suffix}",
            "Feature Name": f"{name} ({stat_label})",
            "Type": "Continuous",
            "Description": desc,
        })

variables_df = pd.DataFrame(rows)

with st.expander("Dataset Variables Reference (30 attributes)", expanded=False):
    st.caption(
        "Ten nuclear characteristics are each measured three ways: "
        "mean, standard error, and worst (largest) value — yielding 30 features. "
        "Use the CSV column names shown below when uploading patient data."
    )
    st.dataframe(variables_df, use_container_width=True, hide_index=True)

# make the adjustable parameters to train the model
st.sidebar.header("Model Parameters")
epochs = st.sidebar.slider(
    "Epochs", 
    min_value=50, 
    max_value=300, 
    value=100, 
    step=10,
    help="Number of full passes through the training data. More epochs allow more learning but risk overfitting."
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate", 
    [0.1, 0.01, 0.001, 0.0001], 
    index=1,
    help="Step size for each weight update. Lower values train more slowly but more stably."
)

hidden_neurons = st.sidebar.slider(
    "Hidden Layer Neurons", 
    min_value=4, 
    max_value=64, 
    value=16,
    help="Width of the hidden layer — controls model capacity. Too high of a value risks overfitting"
)

dropout_prob = st.sidebar.slider(
    "Dropout Rate", 
    min_value=0.0, 
    max_value=0.5, 
    value=0.2, 
    step=0.1,
    help="Fraction of neurons randomly disabled per training step. Prevents overfitting. 0.0 = no dropout"
)

train_pct = st.sidebar.slider(
    "Train / Test Split (% Train)", 
    min_value=60, 
    max_value=90, 
    value=80, 
    step=5,
    help="Percentage of data used for training. The remainder is held out for evaluation. 80% is standard."
)
test_size = (100 - train_pct) / 100
st.sidebar.caption(f"Train: **{train_pct}%** / Test: **{100 - train_pct}%**")
split_label = f"{train_pct} / {100 - train_pct}"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# flip UI to allow predictions once the model has been trained at least once
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_history' not in st.session_state: # store training data
    st.session_state.training_history = []

# train data
if st.button("Train Model"):
    st.write(f"Training model with **{epochs} epochs** and **{learning_rate} LR**...")
    
    # use StandardSCaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # tensor conversion
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # using Logits to calculate loss & Adam optimizer model
    model = BreastCancerNN(input_size=30, hidden_size=hidden_neurons, dropout_prob=dropout_prob)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    X_test_scaled = scaler.transform(X_test)
    X_val_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # UI elements
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    train_loss_history, val_loss_history = [], []

    # loop for each epoch
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
        val_loss_history.append(val_loss)

        # UI updates
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            loss_df = pd.DataFrame({
                "Training Loss":   train_loss_history,
                "Validation Loss": val_loss_history,
            })
            chart_placeholder.line_chart(loss_df, x_label="Epoch", y_label="Loss",color=["#1f77b4", "#d62728"]) # note: used AI to determine appropriate red/blue codes

    st.success("Training Complete!")
    col1, col2 = st.columns(2)

    # display training and validation loss
    col1.metric(
        "Final Training Loss", 
        f"{train_loss_history[-1]:.4f}"
        )
    col2.metric(
        "Final Validation Loss", 
        f"{val_loss_history[-1]:.4f}",
        delta=f"{val_loss_history[-1] - train_loss_history[-1]:+.4f}",
        delta_color="inverse"
        )
    
    # persist session data 
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.model_trained = True

    # store history of training attempts
    st.session_state.training_history.append({
        "Run": len(st.session_state.training_history) + 1,
        "Epochs": epochs,
        "Learning Rate": learning_rate,
        "Hidden Neurons": hidden_neurons,
        "Dropout Rate": dropout_prob,
        "Train/Test Split": split_label,
        "Final Train Loss": f"{train_loss_history[-1]:.4f}",
        "Final Val Loss": f"{val_loss_history[-1]:.4f}",
    })

# display table of training history in the UI
if st.session_state.training_history:
    st.markdown("#### Training Run History")
    history_df = pd.DataFrame(st.session_state.training_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    if st.button("Clear History"):
        st.session_state.training_history = []
        st.rerun()

st.markdown("---")

# prediction method
st.header("Patient Inference")
def run_inference(values_2d):
    patient_scaled = st.session_state.scaler.transform(values_2d)
    patient_tensor = torch.tensor(patient_scaled, dtype=torch.float32)
    st.session_state.model.eval()
    with torch.no_grad():
        logits = st.session_state.model(patient_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()  # convert logits to probabilities

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

# UI - allow user to either upload a CSV with designed features or manually input data
if st.session_state.model_trained:
    X_df = pd.DataFrame(X, columns=feature_names)
    dummy_data = pd.DataFrame([X_test[0]], columns=feature_names)

    tab_csv, tab_manual = st.tabs(["CSV Upload", "Manual Input"])

    # csv upload tab
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

    # manually input data - split 10 features into the mean/standard error/worst categories
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

# don't dispaly predictor if the model hasn't been trained yet
else:
    st.info("Please train the model using the controls above before uploading patient data.")
