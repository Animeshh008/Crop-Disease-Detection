import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import numpy as np
import CNN  # Assuming you have defined this CNN model
import pandas as pd
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Set up the folder to save uploaded images
UPLOAD_FOLDER = os.path.join(os.getcwd(), r"C:/Users/prach/OneDrive/Desktop/Plant-Disease-Detection-main/Flask Deployed App/static/uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your disease info and supplement info from CSVs
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load your trained model
model = CNN.CNN(39)  # Assuming 39 classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=device))
model.eval()

# Function to preprocess image and predict disease
def predict_disease(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_data)
        prediction = np.argmax(output.cpu().numpy())
    return prediction

# Home page route
@app.route('/')
def home_page():
    return render_template('home.html')

# Image upload and disease prediction route
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('submit.html', title="Error", desc="No image uploaded.", prevent="", image_url="", sname="", simage="", buy_link="")

        image = request.files['image']
        filename = secure_filename(image.filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('submit.html', title="Error", desc="Invalid file type. Please upload a PNG or JPG image.", prevent="", image_url="", sname="", simage="", buy_link="")

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(file_path)

        # Get prediction from the model
        pred = predict_disease(file_path)

        # Get disease details from CSV based on prediction
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = filename
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]


        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

    return render_template('submit.html', title="Upload Image", desc="Upload a crop image for analysis.", prevent="", image_url="", sname="", simage="", buy_link="")

# Market page for viewing supplements
@app.route('/market')
def market():
    supplements = zip(supplement_info['supplement name'], supplement_info['supplement image'], supplement_info['buy link'])
    return render_template('market.html', supplements=supplements)

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
