from flask import Flask, render_template, redirect, request, session, url_for, flash
import sqlite3
import pandas as pd
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
import os


app = Flask(__name__)


# Upload configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# It's crucial to set a strong secret key for production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# SQLite setup
DB_PATH = "micro.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn


def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
    print("Database initialized successfully: 'users' table ensured.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']


        if password != c_password:
            flash("Confirm password does not match!") # Using flash for messages
            return render_template('register.html')


        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            # Check if email already exists
            cur.execute("SELECT email FROM users WHERE UPPER(email) = ?", (email.upper(),))
            existing_user = cur.fetchone()


            if existing_user:
                flash("This email ID already exists!")
                return render_template('register.html')
            else:
                # Insert new user
                cur.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
                conn.commit()
                flash("Successfully Registered! Please go to login section")
                return redirect(url_for('login')) # Redirect to login page after successful registration
        except sqlite3.Error as e:
            conn.rollback()
            print(f"Database error during registration: {e}")
            flash("An error occurred during registration. Please try again.")
            return render_template('register.html')
        finally:
            conn.close()
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            # Retrieve user data based on email
            cur.execute("SELECT password FROM users WHERE email = ?", (email,))
            user_data = cur.fetchone() # Fetch one row


            if not user_data: # Email not found
                flash("This email ID does not exist!")
                return render_template('login.html')
            
            stored_password = user_data['password'] # Access by column name due to row_factory


            if password == stored_password: # In production, use hashed passwords and check with bcrypt/werkzeug.security
                session['logged_in'] = True
                session['user_email'] = email # Store email in session
                flash("Login successful!")
                return redirect(url_for("home"))
            else:
                flash("Invalid Password!")
                return render_template('login.html')
        except sqlite3.Error as e:
            print(f"Database error during login: {e}")
            flash("An error occurred during login. Please try again.")
            return render_template('login.html')
        finally:
            conn.close()
    return render_template('login.html')


@app.route('/home')
def home():
    if 'logged_in' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))
    return render_template('home.html', user_email=session.get('user_email'))


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_email', None)
    flash("You have been logged out.")
    return redirect(url_for('index'))


# Load ResNet Model for Direct Prediction
resnet_model_path = "resnet18_trained.pth"

class_names = [
    'Amoeba', 'Euglena', 'Hydra', 'Paramecium',
    'Rod Bacteria', 'Spherical Bacteria',
    'Spiral Bacteria', 'Yeast'
]

# Prepare device and transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

resnet_classifier = None
try:
    # Build ResNet with classification head
    resnet_classifier = models.resnet18(pretrained=False)
    resnet_classifier.fc = torch.nn.Linear(resnet_classifier.fc.in_features, len(class_names))
    # Load trained weights
    resnet_classifier.load_state_dict(torch.load(resnet_model_path, map_location=device))
    resnet_classifier = resnet_classifier.to(device)
    resnet_classifier.eval()
    print(f"ResNet classifier loaded from {resnet_model_path} for direct prediction.")
except Exception as e:
    print(f"Error loading ResNet model: {e}")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        flash("Please log in to upload images.")
        return redirect(url_for('login'))

    prediction = None
    prediction_accuracy = None
    filename = None
    error_message = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error_message = "No file part"
        else:
            file = request.files['image']
            if file.filename == '':
                error_message = "No selected file"
            elif file and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                if resnet_classifier is None:
                    error_message = "ResNet model failed to load. Cannot perform prediction."
                else:
                    try:
                        image = Image.open(filepath).convert("RGB")
                        input_tensor = transform(image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = resnet_classifier(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)
                            
                            predicted_class = class_names[predicted_idx.item()]
                            confidence_pct = round(confidence.item() * 100, 2)
                            
                            prediction = predicted_class
                            prediction_accuracy = confidence_pct

                    except Exception as e:
                        error_message = f"Prediction failed: {str(e)}. Please ensure the image is valid."
                        # Clean up the uploaded file if prediction fails
                        if os.path.exists(filepath):
                            os.remove(filepath)
            else:
                error_message = "Invalid file type. Allowed types are png, jpg, jpeg."

    return render_template('upload.html',
                           filename=filename,
                           prediction=prediction,
                           prediction_accuracy=prediction_accuracy,
                           error_message=error_message)


if __name__ == '__main__':
    init_db() # Initialize the database when the app starts
    app.run(debug=True)
