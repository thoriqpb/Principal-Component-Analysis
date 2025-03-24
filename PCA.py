import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the data
file_path = "/home/uta/Coding/Machine Learning/DIA_trainingset_RDKit_descriptors.csv"
data = pd.read_csv(file_path)

# Preprocess the data
data = data.dropna(subset=["Label"])  # Drop rows where the target (Label) is missing
X = data[["BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v"]]  # Features
y = data["Label"]  # Target

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Apply PCA with 3 components
pca = PCA(n_components=3)  # Reduce to 3 dimensions
X_pca = pca.fit_transform(X_std)

# Visualize the PCA results in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# Scatter plot for Class 0 (DIA-negative drugs) and Class 1 (DIA-positive drugs)
scatter = ax.scatter(
    X_pca[y == 0, 0],  # Principal Component 1 for Class 0
    X_pca[y == 0, 1],  # Principal Component 2 for Class 0
    X_pca[y == 0, 2],  # Principal Component 3 for Class 0
    color="red",
    label="DIA-negative drugs (Class 0)",
    alpha=0.5,
)
scatter = ax.scatter(
    X_pca[y == 1, 0],  # Principal Component 1 for Class 1
    X_pca[y == 1, 1],  # Principal Component 2 for Class 1
    X_pca[y == 1, 2],  # Principal Component 3 for Class 1
    color="blue",
    label="DIA-positive drugs (Class 1)",
    alpha=0.5,
)

# Set labels and title
ax.set(
    title="First three PCA dimensions",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
    zlabel="3rd Principal Component",
)

# Add a legend
ax.legend(loc="upper right", title="Drug Classes")

# Show the plot
plt.show()

# Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)

# Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = LogisticRegression()  
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))