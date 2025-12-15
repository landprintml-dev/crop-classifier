# 🌾 Crop Classification Web App

A simple web interface for predicting crop types in Brazil using satellite imagery from Google Earth Engine.

![Crop Classifier Demo](https://img.shields.io/badge/Status-Ready%20to%20Deploy-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine
```bash
earthengine authenticate
```

### 3. Run the App
```bash
python app.py
```

### 4. Open Browser
Visit: **http://localhost:5000**

---

## 🌐 Deploy to Cloud

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed instructions on deploying to:
- ✅ **Railway** (Recommended - no timeout limits)
- ⚠️ **Vercel** (Alternative - may timeout on slow queries)

---

## 📁 Files

```
models/
├── app.py                          # Flask web app
├── templates/
│   └── index.html                  # UI interface
├── crop_classifier_best.pt         # Trained model weights
├── scaler.pkl                      # Feature scaler
├── label_encoder.pkl               # Label encoder
├── inference_gee.py                # Inference utilities
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway deployment config
├── vercel.json                     # Vercel deployment config
├── runtime.txt                     # Python version
└── DEPLOYMENT.md                   # Deployment guide
```

---

## 🎯 Usage

### Web Interface
1. Enter **latitude** and **longitude** (or click example locations)
2. Select **year** (2015-2024)
3. Click **"Predict Crop Type"**
4. Wait 30-60 seconds for satellite data
5. See prediction results! 🎉

### API Endpoint

**POST** `/api/predict`

```json
{
  "lat": -12.5,
  "lon": -55.7,
  "year": 2023
}
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "Soybean",
  "confidence": 0.89,
  "all_probabilities": {
    "Soybean": 0.89,
    "Pasture": 0.06,
    "Other_Temp_Crops": 0.03,
    ...
  },
  "location": {"lat": -12.5, "lon": -55.7},
  "year": 2023,
  "missing_months": []
}
```

---

## 🧪 Test Locations

| Crop | Location | Coordinates | Year |
|------|----------|-------------|------|
| 🌱 Soybean | Mato Grosso | `-12.5, -55.7` | 2023 |
| 🎋 Sugar Cane | São Paulo | `-21.5, -50.5` | 2023 |
| ☕ Coffee | Minas Gerais | `-21.2, -45.0` | 2023 |
| 🍚 Rice | Rio Grande do Sul | `-30.0, -51.0` | 2023 |

---

## 🏗️ Architecture

```
User Input (lat, lon, year)
    ↓
Flask Backend (app.py)
    ↓
Google Earth Engine API
    ↓
Sentinel-2 Imagery (10 months)
    ↓
Deep Learning Model (Transformer/CNN/LSTM)
    ↓
Crop Type Prediction
```

---

## 📊 Supported Crop Classes

- 🌱 **Soybean** - Major export crop
- 🎋 **Sugar Cane** - Biofuel & sugar production
- 🌾 **Pasture** - Cattle grazing land
- ☕ **Coffee** - Arabica coffee regions
- 🍊 **Citrus** - Orange groves
- 🍚 **Rice** - Irrigated rice paddies
- 🌾 **Other Temp Crops** - Various seasonal crops

---

## 🔧 Configuration

### Change Model
Replace `crop_classifier_best.pt` with another model:
- `best_transformer.pt` (default)
- `best_cnn1d.pt`
- `best_bilstm.pt`

### Adjust Timeout
Edit `Procfile`:
```
web: gunicorn app:app --timeout 180 --workers 1
```

### Customize UI
Edit `templates/index.html` - styles are inline in `<style>` section

---

## 🐛 Troubleshooting

### "GEE Authentication Failed"
```bash
# Re-authenticate
earthengine authenticate
```

### "Model Loading Error"
- Check that `crop_classifier_best.pt` exists
- Verify `scaler.pkl` is present
- Make sure files are in same directory as `app.py`

### "Request Timeout"
- GEE queries can take 30-60 seconds
- Use Railway instead of Vercel
- Try locations with better satellite coverage

### "Module Not Found"
```bash
pip install -r requirements.txt
```

---

## 📈 Performance

- **Response Time**: 30-60 seconds (depends on GEE API)
- **Accuracy**: ~85-90% on test set
- **Coverage**: All of Brazil (2015-2024)
- **Resolution**: 10m (Sentinel-2)

---

## 🔐 Security Notes

- **GEE Credentials**: Never commit credentials to git
- **Rate Limiting**: Consider adding rate limits for production
- **CORS**: Currently allows all origins (adjust in `app.py` if needed)

---

## 📝 License

This project uses:
- Google Earth Engine (requires authentication)
- Sentinel-2 imagery (Copernicus, free and open)
- MapBiomas data (public dataset)

---

## 🙏 Acknowledgments

- **Google Earth Engine** - Satellite imagery platform
- **MapBiomas** - Brazil land cover dataset
- **Sentinel-2** - ESA Earth observation mission

---

## 🚀 Ready to Deploy?

1. ✅ Test locally first (`python app.py`)
2. ✅ Authenticate GEE (`earthengine authenticate`)
3. ✅ Read [DEPLOYMENT.md](DEPLOYMENT.md)
4. ✅ Deploy to Railway or Vercel
5. ✅ Share your link! 🎉

---

**Questions?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides!

