"""
🌾 Model2 Crop Classification Inference
========================================
Inference script for CNN1D_Enhanced model trained with colab_train2.py

Input: Latitude, Longitude, Year (via GEE) OR pre-processed time-series
Output: Predicted crop type with confidence

Model: CNN1D_Enhanced (without SE blocks)
Features: blue, green, red, nir, swir1, swir2, ndvi, ndwi, evi (9)
Months: Oct, Nov, Dec, Jan, Feb, Mar, Apr, May, Jun, Jul (10)
Classes: Citrus, Coffee, Other_Temporary_Crops, Pasture, Rice, Soybean, Sugar_Cane
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path

# ============================================================
# MODEL DEFINITION - CNN1D_Enhanced
# ============================================================
class CNN1D_Enhanced(nn.Module):
    """
    CNN1D Enhanced architecture matching colab_train2.py
    Matches app.py implementation
    """
    def __init__(self, n_features, n_timesteps, n_classes, dropout=0.4):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(n_features, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Block 2
        self.conv2 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv_skip1 = nn.Conv1d(256, 512, kernel_size=1)
        
        # Block 3
        self.conv4 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv_skip2 = nn.Conv1d(512, 1024, kernel_size=1)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(1024, num_heads=8, dropout=dropout, batch_first=True)
        
        # Global pooling
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


# ============================================================
# GEE DATA FETCHER (Optional - for live inference)
# ============================================================
class GEEDataFetcher:
    """Fetch Sentinel-2 time-series from Google Earth Engine"""
    
    MONTHS = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    FEATURES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'evi']
    
    def __init__(self, project_id='landprintml'):
        """Initialize GEE connection"""
        try:
            import ee
            self.ee = ee
            try:
                ee.Initialize(project=project_id)
                print("✅ GEE already initialized")
            except:
                ee.Authenticate()
                ee.Initialize(project=project_id)
                print("✅ GEE authenticated and initialized")
        except ImportError:
            print("⚠️ Google Earth Engine not available. Use preprocess_data() with raw values instead.")
            self.ee = None
    
    def mask_s2_clouds(self, image):
        """Mask clouds using SCL band"""
        scl = image.select('SCL')
        clear_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return image.updateMask(clear_mask)
    
    def get_monthly_composite(self, lat, lon, year, month, month_name):
        """Get median composite for a single month"""
        if self.ee is None:
            return None, month_name
            
        ee = self.ee
        point = ee.Geometry.Point([lon, lat])
        
        if month >= 10:
            start_date = f'{year-1}-{month:02d}-01'
            if month == 12:
                end_date = f'{year}-01-01'
            else:
                end_date = f'{year-1}-{month+1:02d}-01'
        else:
            start_date = f'{year}-{month:02d}-01'
            next_month = month + 1
            end_date = f'{year}-{next_month:02d}-01'
        
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(point)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
              .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL']))
        
        count = s2.size().getInfo()
        if count == 0:
            return None, month_name
        
        s2_masked = s2.map(self.mask_s2_clouds)
        composite = s2_masked.median()
        
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
        """Get full 10-month time-series for a location."""
        if self.ee is None:
            print("❌ GEE not available")
            return None
            
        print(f"\n🛰️ Fetching Sentinel-2 data for ({lat}, {lon}), Year {year}...")
        
        months_info = [
            (10, 'Oct'), (11, 'Nov'), (12, 'Dec'),
            (1, 'Jan'), (2, 'Feb'), (3, 'Mar'),
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
# CROP CLASSIFIER - Model2
# ============================================================
class CropClassifierModel2:
    """
    Crop Classification using Model2 (CNN1D_Enhanced)
    
    Usage:
        classifier = CropClassifierModel2('/path/to/model2/')
        
        # Option 1: With GEE (requires ee library)
        result = classifier.predict_location(lat=-12.5, lon=-55.3, year=2023)
        
        # Option 2: With preprocessed data
        result = classifier.predict_from_data(data_dict)
    """
    
    MONTHS = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    FEATURES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'evi']
    
    def __init__(self, model_dir, project_id='landprintml', use_gee=True):
        """Load model and optionally initialize GEE"""
        self.model_dir = Path(model_dir)
        self.project_id = project_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("📦 Loading Model2 (CNN1D_Enhanced)...")
        
        # Load checkpoint
        checkpoint_path = self.model_dir / 'crop_classifier_best.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        self.n_classes = checkpoint['n_classes']
        self.n_features = checkpoint['n_features']
        self.n_timesteps = checkpoint['n_timesteps']
        self.model_name = checkpoint.get('model_name', 'CNN1D_Enhanced')
        dropout = checkpoint.get('dropout', 0.4)
        
        print(f"   Model: {self.model_name}")
        print(f"   Features: {self.n_features}")
        print(f"   Timesteps: {self.n_timesteps}")
        print(f"   Classes: {self.n_classes}")
        
        # Initialize model
        self.model = CNN1D_Enhanced(
            self.n_features, 
            self.n_timesteps, 
            self.n_classes, 
            dropout=dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("   ✓ Scaler loaded")
        else:
            self.scaler = None
            print("   ⚠️ No scaler found - using raw values")
        
        print(f"   ✓ Classes: {self.class_names}")
        print(f"✅ Model2 loaded successfully on {self.device}")
        
        # Initialize GEE if requested
        self.gee = None
        if use_gee:
            try:
                self.gee = GEEDataFetcher(project_id=project_id)
            except Exception as e:
                print(f"⚠️ GEE initialization failed: {e}")
    
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
        
        # Normalize using scaler
        if self.scaler is not None:
            X_flat = X.reshape(-1, self.n_features)
            X_scaled = self.scaler.transform(X_flat)
            X = X_scaled.reshape(1, self.n_timesteps, self.n_features)
        
        return torch.FloatTensor(X).to(self.device)
    
    @torch.no_grad()
    def predict_from_data(self, data_dict):
        """
        Predict crop type from preprocessed data dictionary.
        
        Args:
            data_dict: Dictionary with keys like 'blue_tOct', 'ndvi_tJan', etc.
        
        Returns:
            dict with prediction results
        """
        X = self.preprocess(data_dict)
        
        outputs = self.model(X)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = outputs.argmax(dim=1).item()
        
        predicted_class = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()
        all_probs = {name: probs[i].item() for i, name in enumerate(self.class_names)}
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs,
        }
    
    @torch.no_grad()
    def predict_location(self, lat, lon, year):
        """
        Predict crop type at a location using GEE.
        
        Args:
            lat: Latitude
            lon: Longitude  
            year: Year (e.g., 2023)
        
        Returns:
            dict with prediction results
        """
        if self.gee is None or self.gee.ee is None:
            return {
                'success': False,
                'error': 'GEE not available. Use predict_from_data() instead.'
            }
        
        print(f"\n{'='*60}")
        print(f"🌾 CROP CLASSIFICATION (Model2)")
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
        
        # Get prediction
        result = self.predict_from_data(data)
        
        # Add location info
        result['location'] = {'lat': lat, 'lon': lon}
        result['year'] = year
        result['raw_data'] = data
        
        # Print results
        print(f"\n{'='*60}")
        print(f"🎯 PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"   Predicted Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"\n   All Probabilities:")
        for name, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
            bar = '█' * int(prob * 20)
            print(f"      {name:25s}: {prob:6.1%} {bar}")
        print(f"{'='*60}\n")
        
        return result


# ============================================================
# STANDALONE FUNCTIONS (for API/backend use)
# ============================================================
def load_model2(model_dir=None, use_gee=False):
    """
    Load Model2 classifier for inference.
    
    Args:
        model_dir: Path to model2 directory (defaults to same directory as this script)
        use_gee: Whether to initialize GEE connection
    
    Returns:
        CropClassifierModel2 instance
    """
    if model_dir is None:
        model_dir = Path(__file__).parent
    
    return CropClassifierModel2(model_dir, use_gee=use_gee)


def predict_crop_model2(data_dict, model_dir=None):
    """
    Quick prediction function for single sample.
    
    Args:
        data_dict: Dictionary with time-series features
        model_dir: Path to model2 directory
    
    Returns:
        dict with predicted_class, confidence, all_probabilities
    """
    classifier = load_model2(model_dir, use_gee=False)
    return classifier.predict_from_data(data_dict)


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == '__main__':
    # Get the directory containing this script
    SCRIPT_DIR = Path(__file__).parent
    
    print("\n" + "="*70)
    print("🌾 MODEL2 CROP CLASSIFIER")
    print("="*70)
    
    # Initialize classifier (without GEE for testing)
    classifier = CropClassifierModel2(SCRIPT_DIR, use_gee=False)
    
    # Example: Create mock data for testing
    print("\n📝 Testing with mock data...")
    
    # Create sample data (you would replace with real values)
    sample_data = {}
    for month in classifier.MONTHS:
        sample_data[f'blue_t{month}'] = 500
        sample_data[f'green_t{month}'] = 800
        sample_data[f'red_t{month}'] = 600
        sample_data[f'nir_t{month}'] = 3000
        sample_data[f'swir1_t{month}'] = 2000
        sample_data[f'swir2_t{month}'] = 1000
        sample_data[f'ndvi_t{month}'] = 0.7
        sample_data[f'ndwi_t{month}'] = -0.5
        sample_data[f'evi_t{month}'] = 0.5
    
    # Make prediction
    result = classifier.predict_from_data(sample_data)
    
    print(f"\n{'='*60}")
    print(f"🎯 PREDICTION RESULT (Mock Data)")
    print(f"{'='*60}")
    print(f"   Predicted Class: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"\n   All Probabilities:")
    for name, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 20)
        print(f"      {name:25s}: {prob:6.1%} {bar}")
    print(f"{'='*60}")
    
    print("\n✅ Model2 inference working correctly!")
    print("\n📖 Usage:")
    print("   from inference_gee2 import CropClassifierModel2")
    print("   classifier = CropClassifierModel2('/path/to/model2/')")
    print("   result = classifier.predict_from_data(data_dict)")
