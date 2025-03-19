# %%
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os

# %%
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for CNN
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

# %%
# Load pre-trained ResNet18 for feature extraction
model = models.resnet101(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last classification layer
model.eval()  # Set model to evaluation mode

# %%
# Function to extract features
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        feature = model(image)
    return feature.squeeze().numpy().flatten()  # Flatten the feature vector

# %%
# Set your image folder path
image_folder = "unlabeled_data"

# %%
# Load all images and extract features
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

features = []
image_paths = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    feature_vector = extract_features(image_path)
    features.append(feature_vector)
    image_paths.append(image_path)

features = np.array(features)  # Convert list to NumPy array

# %%
from sklearn.cluster import KMeans

# Set number of clusters (choose based on your dataset)
# N_CLUSTERS = 5  

# Apply K-Means clustering
# kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans = KMeans(random_state=42, n_init=20)
labels = kmeans.fit_predict(features)

N_CLUSTERS = np.unique(labels).size

labels


# %%
import numpy as np

# Compute cluster centers
cluster_centers = kmeans.cluster_centers_

# Compute distances of each image from its assigned cluster center
distances = np.linalg.norm(features - cluster_centers[labels], axis=1)

# Identify outliers (e.g., top 5% farthest points)
threshold = np.percentile(distances, 95)  # Top 5% as outliers
outlier_indices = np.where(distances > threshold)[0]

print(f"Detected {len(outlier_indices)} outliers out of {len(features)} images.")


# %%
import matplotlib.pyplot as plt
from PIL import Image

# Select outlier images
outlier_images = [image_paths[i] for i in outlier_indices]

# Display outlier images
def show_outliers(outlier_images, max_cols=5):
    num_outliers = len(outlier_images)
    num_cols = min(num_outliers, max_cols)
    num_rows = (num_outliers + num_cols - 1) // num_cols  # Compute needed rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()

    for i, img_path in enumerate(outlier_images):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")

    # Hide any empty subplot slots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Detected Outliers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

show_outliers(outlier_images)


# %%
import shutil

# Create output directories
output_base = "labeled_data"
if os.path.exists(output_base):
    shutil.rmtree(output_base)

os.makedirs(output_base, exist_ok=True)

for cluster_id in range(N_CLUSTERS):
    os.makedirs(os.path.join(output_base, f"cluster_{cluster_id}"), exist_ok=True)

# Move images to respective cluster folders
for i, image_path in enumerate(image_paths):
    cluster_id = labels[i]
    new_path = os.path.join(output_base, f"cluster_{cluster_id}", os.path.basename(image_path))
    shutil.copyfile(image_path, new_path)

print(f"Images have been clustered into {N_CLUSTERS} groups and moved to '{output_base}'")


# %%
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output

# Define the number of clusters
N_CLUSTERS = 5  # Adjust based on your actual clusters

def move_image(image_path, destination_folder):
    """Move an image to the selected cluster folder."""
    os.makedirs(f'labeled_data/{destination_folder}', exist_ok=True)
    new_path = os.path.join(destination_folder, os.path.basename(image_path))
    shutil.move(image_path, new_path)
    print(f"Moved: {image_path} → {new_path}")

def delete_image(image_path):
    """Delete an image."""
    os.remove(image_path)
    print(f"Deleted: {image_path}")

def create_image_display(img_path):
    """Helper function to display an image with buttons below it."""
    img = Image.open(img_path)
    img.thumbnail((150, 150))  # Resize for uniform display
    
    # Display Image
    image_widget = widgets.Output()
    with image_widget:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    
    # Dropdown to select move destination
    move_dropdown = widgets.Dropdown(
        options=[(f"Cluster {i}", f"cluster_{i}") for i in range(N_CLUSTERS)],
        description="Move to:",
        layout=widgets.Layout(width='150px')
    )
    
    move_button = widgets.Button(description="Move", layout=widgets.Layout(width='60px'))
    delete_button = widgets.Button(description="Delete", layout=widgets.Layout(width='60px'))
    
    # Define button actions
    move_button.on_click(lambda btn: move_image(img_path, move_dropdown.value))
    delete_button.on_click(lambda btn: delete_image(img_path))
    
    # Layout buttons and dropdown
    button_box = widgets.HBox([move_dropdown, move_button, delete_button])
    return widgets.VBox([image_widget, button_box])

def visualize_clusters(image_paths, labels, max_columns=5):
    """Visualize clustered images with interactive move/delete buttons."""
    clusters = {}
    # Loop through each item in output_base
    for dir_name in os.listdir(output_base):
        dir_path = os.path.join(output_base, dir_name)  # Get full path
        
        # Check if it's a directory before listing files inside
        if os.path.isdir(dir_path):
            # Loop through each file in the directory
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                clusters.setdefault(dir_name, []).append(file_path)
    
    for cluster_id, img_list in clusters.items():
        print(f"Cluster {cluster_id}:")
        display(widgets.Label(value=f"Cluster {cluster_id}", style={'font_weight': 'bold'}))
        
        image_widgets = [create_image_display(img) for img in img_list]
        # Create a horizontal scrollable container
        scrollable_row = widgets.Box(
            children=image_widgets,
            layout=widgets.Layout(
                display="flex",
                flex_direction="row",
                overflow_x="auto",  # Enables horizontal scrolling
                width="100%",       
                height="600px",     # Adjust for visibility
                flex_wrap="nowrap"  # Prevents wrapping to new lines
            )
        )
        
        # Ensure images do not shrink too much
        for img_widget in image_widgets:
            img_widget.layout = widgets.Layout(min_width="300px", margin="5px")  # Adjust min_width for proper spacing
        
        display(scrollable_row)

# Example usage
visualize_clusters(output_base, labels)


