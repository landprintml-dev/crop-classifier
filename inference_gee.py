"""
🌾 Crop Classification from Coordinates
=======================================

Input: Latitude, Longitude, Year
Output: Predicted crop type

This script:
1. Connects to Google Earth Engine
2. Fetches Sentinel-2 time-series for the location
3. Computes spectral indices
4. Makes prediction using trained model

Run in Google Colab!
"""

import ee
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path

# ============================================================
# MODEL DEFINITIONS
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
    
    def __init__(self, project_id='landprintml'):
        """Initialize GEE connection"""
        try:
            ee.Initialize(project=project_id)
            print("✅ GEE already initialized")
        except:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            print("✅ GEE authenticated and initialized")
    
    def mask_s2_clouds(self, image):
        """Mask clouds using SCL band"""
        scl = image.select('SCL')
        clear_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return image.updateMask(clear_mask)
    
    def get_monthly_composite(self, lat, lon, year, month, month_name):
        """Get median composite for a single month"""
        # Create point geometry
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
                'blue': blue,
                'green': green,
                'red': red,
                'nir': nir,
                'swir1': swir1,
                'swir2': swir2,
                'ndvi': ndvi,
                'ndwi': ndwi,
                'evi': evi
            }, month_name
            
        except Exception as e:
            return None, month_name
    
    def get_timeseries(self, lat, lon, year):
        """
        Get full 10-month time-series for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            year: Year (e.g., 2023) - will fetch Oct(year-1) to Jul(year)
        
        Returns:
            dict with all feature values, or None if failed
        """
        print(f"\n🛰️ Fetching Sentinel-2 data for ({lat}, {lon}), Year {year}...")
        
        months_info = [
            (10, 'Oct'), (11, 'Nov'), (12, 'Dec'),  # Previous year
            (1, 'Jan'), (2, 'Feb'), (3, 'Mar'),     # Current year
            (4, 'Apr'), (5, 'May'), (6, 'Jun'), (7, 'Jul')
        ]
        
        result = {}
        missing_months = []
        
        for month_num, month_name in months_info:
            print(f"   {month_name}...", end=" ")
            data, _ = self.get_monthly_composite(lat, lon, year, month_num, month_name)
            
            if data is None:
                print("❌ (no data)")
                missing_months.append(month_name)
                # Fill with NaN (will be imputed)
                for feat in self.FEATURES:
                    result[f'{feat}_t{month_name}'] = np.nan
            else:
                print("✅")
                for feat in self.FEATURES:
                    result[f'{feat}_t{month_name}'] = data[feat]
        
        if len(missing_months) > 2:
            print(f"\n⚠️ Too many missing months ({len(missing_months)}): {missing_months}")
            return None
        
        if missing_months:
            print(f"\n⚠️ Missing months (will be imputed): {missing_months}")
        
        print(f"✅ Time-series complete!")
        return result


# ============================================================
# CROP CLASSIFIER WITH GEE
# ============================================================

class CropClassifierGEE:
    """
    Crop Classification from coordinates using GEE.
    
    Usage:
        classifier = CropClassifierGEE('/path/to/models/', project_id='landprintml')
        result = classifier.predict_location(lat=-12.5, lon=-55.3, year=2023)
    """
    
    MONTHS = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    FEATURES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'evi']
    
    def __init__(self, model_dir, project_id='landprintml'):
        """Load model and initialize GEE"""
        self.model_dir = Path(model_dir)
        self.project_id = project_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("📦 Loading model...")
        checkpoint = torch.load(self.model_dir / 'crop_classifier_best.pt', map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        self.n_classes = checkpoint['n_classes']
        self.n_features = checkpoint['n_features']
        self.n_timesteps = checkpoint['n_timesteps']
        self.model_name = checkpoint['model_name']
        
        print(f"   Model type: {self.model_name}")
        
        # Handle model name variations
        model_name_lower = self.model_name.lower()
        if model_name_lower == 'cnn1d' or 'cnn' in model_name_lower:
            self.model = CNN1D(self.n_features, self.n_timesteps, self.n_classes)
        elif model_name_lower == 'bilstm' or 'lstm' in model_name_lower:
            self.model = BiLSTM(self.n_features, self.n_timesteps, self.n_classes)
        elif model_name_lower == 'transformer' or 'trans' in model_name_lower:
            self.model = TransformerClassifier(self.n_features, self.n_timesteps, self.n_classes)
        else:
            raise ValueError(f"Unknown model type: '{self.model_name}'. Expected 'CNN1D', 'BiLSTM', or 'Transformer'")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        with open(self.model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Loaded {self.model_name} model")
        print(f"   Classes: {self.class_names}")
        
        # Initialize GEE with project ID
        self.gee = GEEDataFetcher(project_id=self.project_id)
    
    def preprocess(self, data_dict):
        """Convert feature dict to model input tensor"""
        # Create array in correct order
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
        """
        Predict crop type at a location.
        
        Args:
            lat: Latitude (e.g., -12.5 for Southern Brazil)
            lon: Longitude (e.g., -55.3 for Mato Grosso)
            year: Year (e.g., 2023)
        
        Returns:
            dict with prediction results
        """
        print(f"\n{'='*60}")
        print(f"🌾 CROP CLASSIFICATION")
        print(f"{'='*60}")
        print(f"   📍 Location: ({lat}, {lon})")
        print(f"   📅 Year: {year}")
        print(f"{'='*60}")
        
        # Fetch data from GEE
        data = self.gee.get_timeseries(lat, lon, year)
        
        if data is None:
            return {
                'success': False,
                'error': 'Could not fetch data (too many missing months)'
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
        all_probs = {name: probs[i].item() for i, name in enumerate(self.class_names)}
        
        # Print results
        print(f"\n{'='*60}")
        print(f"🎯 PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"   Predicted Class: {predicted_class}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"\n   All Probabilities:")
        for name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            bar = '█' * int(prob * 20)
            print(f"      {name:25s}: {prob:6.1%} {bar}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'location': {'lat': lat, 'lon': lon},
            'year': year,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'raw_data': data
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == '__main__':
    # Path to your trained model
    MODEL_DIR = "/content/drive/MyDrive/IDP2_exports/models"
    
    # Initialize classifier
    classifier = CropClassifierGEE(MODEL_DIR)
    
    # Example locations in Brazil
    examples = [
        # Mato Grosso - Soybean belt
        {"lat": -12.5, "lon": -55.7, "year": 2023, "expected": "Soybean"},
        
        # São Paulo - Sugar Cane region  
        {"lat": -21.5, "lon": -50.5, "year": 2023, "expected": "Sugar_Cane"},
        
        # Minas Gerais - Coffee region
        {"lat": -21.2, "lon": -45.0, "year": 2023, "expected": "Coffee"},
        
        # Rio Grande do Sul - Rice region
        {"lat": -30.0, "lon": -51.0, "year": 2023, "expected": "Rice"},
    ]
    
    print("\n" + "="*70)
    print("🌾 CROP CLASSIFICATION EXAMPLES")
    print("="*70)
    
    for ex in examples:
        result = classifier.predict_location(ex['lat'], ex['lon'], ex['year'])
        
        if result['success']:
            match = "✅" if result['predicted_class'] == ex['expected'] else "❓"
            print(f"\n{match} Expected: {ex['expected']}, Got: {result['predicted_class']}")

