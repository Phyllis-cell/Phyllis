from flask import Flask, render_template_string, request
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# Create Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model (for demo: using a mock-up of mask/no-mask)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: Mask, No Mask
model.eval()  # Set model to inference mode

# For demo: random weights (simulate behavior)
with torch.no_grad():
    model.fc.weight[:] = torch.randn_like(model.fc.weight)
    model.fc.bias[:] = torch.randn_like(model.fc.bias)

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# HTML template
html_template = '''
<!doctype html>
<title>Mask Detection</title>
<h2>😷 Mask vs No Mask Classifier</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=image>
  <input type=submit value='Classify Image'>
</form>

{% if filename %}
    <h3>📷 Uploaded Image:</h3>
    <img src="{{ url_for('static', filename='uploads/' + filename) }}" width="300"><br><br>
    <h3>🔍 Prediction: <span style="color: green;">{{ label }}</span></h3>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    label = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file).convert('RGB')
            filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(save_path)

            # Transform and predict
            img_tensor = transform(image).unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                label = 'Mask' if predicted.item() == 0 else 'No Mask'

    return render_template_string(html_template, filename=filename, label=label)

if __name__ == '__main__':
    app.run(debug=True)
