"""
🌾 Crop Classification Web API
==============================
Simple Flask app for crop prediction from coordinates.
Deploy to Railway.app for easy sharing!
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ee
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import os
import json

app = Flask(__name__)
CORS(app)

# ============================================================
# MODEL DEFINITIONS (same as inference_gee.py)
# ============================================================

class CNN1D(nn.Module):
    def __init__(self, n_features, n_timesteps, n_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNN1D_Enhanced(nn.Module):
    """Enhanced CNN1D with skip connections, attention, and more layers (matches training architecture)"""
    def __init__(self, n_features, n_timesteps, n_classes, dropout=0.4):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(n_features, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.conv2 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv_skip1 = nn.Conv1d(256, 512, kernel_size=1)
        
        self.conv4 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv_skip2 = nn.Conv1d(512, 1024, kernel_size=1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(1024, num_heads=8, dropout=dropout, batch_first=True)
        
        # Global pooling (Dual: Avg + Max)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, time)
        
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        identity1 = x
        
        # Block 2 with residual
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x + self.conv_skip1(identity1)
        identity2 = x
        
        # Block 3 with residual
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = x + self.conv_skip2(identity2)
        
        # Multi-head attention
        x_att = x.transpose(1, 2)
        x_att, _ = self.attention(x_att, x_att, x_att)
        x = x_att.transpose(1, 2)
        
        # Dual pooling
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x_max = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)
        
        # Classification head
        x = self.dropout(x)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        return self.fc4(x)


class BiLSTM(nn.Module):
    def __init__(self, n_features, n_timesteps, n_classes, hidden_size=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(self.dropout(context))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, n_timesteps, n_classes, d_model=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=n_timesteps + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.fc(self.dropout(x[:, 0]))


# ============================================================
# GEE DATA FETCHER
# ============================================================

class GEEDataFetcher:
    """Fetch Sentinel-2 time-series from Google Earth Engine"""
    
    MONTHS = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    FEATURES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'evi']
    
    def __init__(self):
        """Initialize GEE connection"""
        try:
            # Get project ID from environment or use default
            project_id = os.environ.get('GEE_PROJECT_ID', 'landprintml')
            
            # Check if we have credentials in environment variable
            earthengine_token = os.environ.get('EARTHENGINE_TOKEN')
            
            if earthengine_token:
                # Write credentials to temporary file for GEE
                import json
                import tempfile
                
                # Create credentials directory
                creds_dir = os.path.expanduser('~/.config/earthengine')
                os.makedirs(creds_dir, exist_ok=True)
                
                # Write credentials
                creds_path = os.path.join(creds_dir, 'credentials')
                with open(creds_path, 'w') as f:
                    f.write(earthengine_token)
                
                print(f"✅ GEE credentials written to {creds_path}")
                ee.Initialize(project=project_id)
                print(f"✅ GEE initialized with environment credentials (project: {project_id})")
            elif os.path.exists('gee-service-account.json'):
                # Try service account authentication
                credentials = ee.ServiceAccountCredentials(
                    os.environ.get('GEE_SERVICE_ACCOUNT'),
                    'gee-service-account.json'
                )
                ee.Initialize(credentials, project=project_id)
                print(f"✅ GEE initialized with service account (project: {project_id})")
            else:
                # Fallback to default authentication (for local testing)
                ee.Initialize(project=project_id)
                print(f"✅ GEE initialized with default credentials (project: {project_id})")
        except Exception as e:
            print(f"⚠️ GEE initialization error: {e}")
            raise
    
    def mask_s2_clouds(self, image):
        """Mask clouds using SCL band"""
        scl = image.select('SCL')
        clear_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return image.updateMask(clear_mask)
    
    def get_monthly_composite(self, lat, lon, year, month, month_name):
        """Get median composite for a single month"""
        point = ee.Geometry.Point([lon, lat])
        
        # Calculate date range
        if month >= 10:  # Oct, Nov, Dec from previous year
            start_date = f'{year-1}-{month:02d}-01'
            if month == 12:
                end_date = f'{year}-01-01'
            else:
                end_date = f'{year-1}-{month+1:02d}-01'
        else:  # Jan-Jul from current year
            start_date = f'{year}-{month:02d}-01'
            next_month = month + 1
            end_date = f'{year}-{next_month:02d}-01'
        
        # Get Sentinel-2 collection
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(point)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
              .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL']))
        
        count = s2.size().getInfo()
        if count == 0:
            return None, month_name
        
        # Apply cloud masking and get median
        s2_masked = s2.map(self.mask_s2_clouds)
        composite = s2_masked.median()
        
        # Sample at point
        sample = composite.sample(point, scale=10).first()
        
        if sample is None:
            return None, month_name
        
        try:
            values = sample.getInfo()['properties']
            
            blue = values.get('B2', 0)
            green = values.get('B3', 0)
            red = values.get('B4', 0)
            nir = values.get('B8', 0)
            swir1 = values.get('B11', 0)
            swir2 = values.get('B12', 0)
            
            # Calculate indices
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndwi = (green - nir) / (green + nir + 1e-10)
            evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
            
            return {
                'blue': blue, 'green': green, 'red': red,
                'nir': nir, 'swir1': swir1, 'swir2': swir2,
                'ndvi': ndvi, 'ndwi': ndwi, 'evi': evi
            }, month_name
            
        except Exception as e:
            return None, month_name
    
    def get_timeseries(self, lat, lon, year):
        """Get full 10-month time-series for a location"""
        months_info = [
            (10, 'Oct'), (11, 'Nov'), (12, 'Dec'),
            (1, 'Jan'), (2, 'Feb'), (3, 'Mar'),
            (4, 'Apr'), (5, 'May'), (6, 'Jun'), (7, 'Jul')
        ]
        
        result = {}
        missing_months = []
        
        for month_num, month_name in months_info:
            data, _ = self.get_monthly_composite(lat, lon, year, month_num, month_name)
            
            if data is None:
                missing_months.append(month_name)
                for feat in self.FEATURES:
                    result[f'{feat}_t{month_name}'] = np.nan
            else:
                for feat in self.FEATURES:
                    result[f'{feat}_t{month_name}'] = data[feat]
        
        if len(missing_months) > 2:
            return None, missing_months
        
        return result, missing_months


# ============================================================
# CROP CLASSIFIER
# ============================================================

class CropClassifier:
    """Crop Classification from coordinates"""
    
    MONTHS = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    FEATURES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'evi']
    
    def __init__(self, model_dir='.', model_file='crop_classifier_best.pt'):
        """Load model and initialize GEE"""
        self.model_dir = Path(model_dir)
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"📦 Loading model: {model_file}...")
        checkpoint = torch.load(self.model_dir / model_file, map_location=self.device)
        
        self.class_names = checkpoint.get('class_names', checkpoint.get('classes', []))
        self.n_classes = checkpoint.get('n_classes', len(self.class_names))
        self.n_features = checkpoint.get('n_features', 9)
        self.n_timesteps = checkpoint.get('n_timesteps', 10)
        self.model_name = checkpoint.get('model_name', checkpoint.get('model_type', 'Transformer'))
        
        # Create model
        model_name_lower = self.model_name.lower()
        if 'cnn1d_enhanced' in model_name_lower or 'enhanced' in model_name_lower:
            self.model = CNN1D_Enhanced(self.n_features, self.n_timesteps, self.n_classes)
        elif 'cnn' in model_name_lower:
            self.model = CNN1D(self.n_features, self.n_timesteps, self.n_classes)
        elif 'lstm' in model_name_lower:
            self.model = BiLSTM(self.n_features, self.n_timesteps, self.n_classes)
        else:
            self.model = TransformerClassifier(self.n_features, self.n_timesteps, self.n_classes)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler (use model-specific scaler if available, else default)
        scaler_file = 'scaler2.pkl' if 'best_2' in model_file else 'scaler.pkl'
        try:
            with open(self.model_dir / scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Loaded scaler: {scaler_file}")
        except FileNotFoundError:
            # Fallback to default scaler
            with open(self.model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"⚠️ Using default scaler (scaler2.pkl not found)")
        
        print(f"✅ Loaded {self.model_name} model")
        print(f"   Classes: {self.class_names}")
        
        # Initialize GEE
        self.gee = GEEDataFetcher()
    
    def preprocess(self, data_dict):
        """Convert feature dict to model input tensor"""
        X = np.zeros((1, self.n_timesteps, self.n_features))
        
        for t, month in enumerate(self.MONTHS):
            for f, feat in enumerate(self.FEATURES):
                X[0, t, f] = data_dict.get(f'{feat}_t{month}', np.nan)
        
        # Handle NaN with column means
        for f in range(self.n_features):
            col = X[0, :, f]
            if np.any(np.isnan(col)):
                mean_val = np.nanmean(col) if not np.all(np.isnan(col)) else 0
                col[np.isnan(col)] = mean_val
                X[0, :, f] = col
        
        # Normalize
        X_flat = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.transform(X_flat)
        X = X_scaled.reshape(1, self.n_timesteps, self.n_features)
        
        return torch.FloatTensor(X).to(self.device)
    
    @torch.no_grad()
    def predict_location(self, lat, lon, year):
        """Predict crop type at a location"""
        # Fetch data from GEE
        data, missing_months = self.gee.get_timeseries(lat, lon, year)
        
        if data is None:
            return {
                'success': False,
                'error': f'Too many missing months ({len(missing_months)}): {missing_months}'
            }
        
        # Preprocess
        X = self.preprocess(data)
        
        # Predict
        outputs = self.model(X)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = outputs.argmax(dim=1).item()
        
        predicted_class = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()
        
        # Get all probabilities
        all_probs = {name: float(probs[i].item()) for i, name in enumerate(self.class_names)}
        
        return {
            'success': True,
            'location': {'lat': lat, 'lon': lon},
            'year': year,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'missing_months': missing_months
        }


# ============================================================
# FLASK ROUTES
# ============================================================

# Global classifier instances (one per model)
classifiers = {}

def init_classifier(model_name='best'):
    """Initialize classifier on first request for a specific model"""
    global classifiers
    
    # Map model names to files
    model_files = {
        'best': 'crop_classifier_best.pt',
        'best_2': 'crop_classifier_best_2.pt'
    }
    
    if model_name not in model_files:
        model_name = 'best'  # Default fallback
    
    if model_name not in classifiers:
        model_file = model_files[model_name]
        print(f"🔄 Initializing model: {model_name} ({model_file})")
        classifiers[model_name] = CropClassifier('.', model_file=model_file)
    
    return classifiers[model_name]


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for crop prediction"""
    try:
        data = request.get_json()
        
        # Validate input
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        year = int(data.get('year'))
        model_name = data.get('model', 'best')  # Default to 'best'
        
        if not (-90 <= lat <= 90):
            return jsonify({'success': False, 'error': 'Latitude must be between -90 and 90'}), 400
        
        if not (-180 <= lon <= 180):
            return jsonify({'success': False, 'error': 'Longitude must be between -180 and 180'}), 400
        
        if not (2015 <= year <= 2024):
            return jsonify({'success': False, 'error': 'Year must be between 2015 and 2024'}), 400
        
        # Initialize classifier for selected model
        clf = init_classifier(model_name)
        
        # Make prediction
        result = clf.predict_location(lat, lon, year)
        
        # Add model info to result
        result['model_used'] = model_name
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/mapbiomas', methods=['POST'])
def get_mapbiomas():
    """API endpoint to fetch MapBiomas ground truth"""
    try:
        data = request.get_json()
        
        # Validate input
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        year = int(data.get('year'))
        
        # Initialize classifier if needed (to access GEE)
        clf = init_classifier()
        
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])
        
        # Load MapBiomas classification for the year
        mapbiomas = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2')
        classification = mapbiomas.select(f'classification_{year}')
        
        # Sample at point
        sample = classification.sample(point, 30).first()
        result = sample.getInfo()
        
        if result and result.get('properties'):
            class_code = result['properties'].get(f'classification_{year}')
            
            # MapBiomas class names
            class_names = {
                15: 'Pasture', 20: 'Sugar_Cane', 39: 'Soybean', 40: 'Rice',
                41: 'Other_Temporary_Crops', 46: 'Coffee', 47: 'Citrus',
                1: 'Forest', 3: 'Forest Formation', 24: 'Urban Area',
                26: 'Water', 33: 'River/Lake/Ocean'
            }
            
            class_name = class_names.get(class_code, f'Class {class_code}')
            
            return jsonify({
                'success': True,
                'class_code': class_code,
                'class_name': class_name,
                'lat': lat,
                'lon': lon,
                'year': year
            })
        else:
            return jsonify({'success': False, 'error': 'No data at this location'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mapbiomas-tiles/<int:year>')
def get_mapbiomas_tiles(year):
    """Get MapBiomas tile URL for the year"""
    try:
        # Initialize classifier if needed (to access GEE)
        clf = init_classifier()
        
        # Load MapBiomas classification
        mapbiomas = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2')
        classification = mapbiomas.select(f'classification_{year}')
        
        # Create palette for our 7 classes (gray for rest)
        # Classes: 15=Pasture, 20=Sugar_Cane, 39=Soybean, 40=Rice, 41=Other_Temp, 46=Coffee, 47=Citrus
        palette = ['cccccc'] * 63  # Gray for all
        palette[15] = 'FFD700'  # Pasture - Gold
        palette[20] = 'FF0000'  # Sugar Cane - Red
        palette[39] = '0066FF'  # Soybean - Blue
        palette[40] = '00CC00'  # Rice - Green
        palette[41] = 'FF6600'  # Other Temp - Orange
        palette[46] = '8B0000'  # Coffee - Dark Red
        palette[47] = '00FFFF'  # Citrus - Cyan
        
        # Visualize
        vis_params = {
            'min': 0,
            'max': 62,
            'palette': palette
        }
        
        # Get tile URL
        map_id = classification.visualize(**vis_params).getMapId()
        tile_url = map_id['tile_fetcher'].url_format
        
        return jsonify({
            'success': True,
            'tile_url': tile_url,
            'year': year
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'models_loaded': list(classifiers.keys()),
        'available_models': ['best', 'best_2']
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

