# ⚡ Quick Start - Deploy in 5 Minutes

## 🎯 What You Got

A complete **web app** for crop classification that you can deploy to **Railway** or **Vercel**!

### Files Created:
- ✅ `app.py` - Flask backend with API
- ✅ `templates/index.html` - Beautiful UI
- ✅ `requirements.txt` - Dependencies
- ✅ `Procfile` - Railway config
- ✅ `vercel.json` - Vercel config
- ✅ `test_local.py` - Pre-deployment tests
- ✅ `DEPLOYMENT.md` - Detailed guide
- ✅ `README.md` - Documentation

---

## 🚀 Deploy NOW (3 Steps)

### Option 1: Railway (Recommended - No Timeouts)

```bash
# 1. Test locally first
cd models/
python test_local.py

# 2. Run locally
python app.py
# Visit http://localhost:5000

# 3. Deploy to Railway
# Go to https://railway.app
# Click "New Project" → "Deploy from GitHub"
# Connect your repo → Done! 🎉
```

### Option 2: Vercel (Quick but has 60s timeout)

```bash
# 1. Test locally
cd models/
python test_local.py

# 2. Install Vercel CLI
npm install -g vercel

# 3. Deploy
vercel login
vercel
```

---

## 🧪 Test Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate GEE (one-time)
earthengine authenticate

# Run tests
python test_local.py

# Start server
python app.py

# Open browser → http://localhost:5000
```

---

## 📍 Try These Locations

Once deployed, test with:

| Crop | Lat | Lon | Year |
|------|-----|-----|------|
| Soybean | -12.5 | -55.7 | 2023 |
| Sugar Cane | -21.5 | -50.5 | 2023 |
| Coffee | -21.2 | -45.0 | 2023 |
| Rice | -30.0 | -51.0 | 2023 |

---

## 🎨 Features

✅ Simple one-page interface  
✅ Real-time satellite data from GEE  
✅ Prediction confidence scores  
✅ All class probabilities  
✅ Beautiful gradient UI  
✅ Mobile responsive  
✅ Example locations built-in  
✅ RESTful API endpoint  

---

## 🔑 API Example

```bash
curl -X POST https://your-app.railway.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"lat": -12.5, "lon": -55.7, "year": 2023}'
```

Response:
```json
{
  "success": true,
  "predicted_class": "Soybean",
  "confidence": 0.89,
  "all_probabilities": {...}
}
```

---

## 🐛 Issues?

### GEE Not Authenticated
```bash
earthengine authenticate
```

### Model Not Found
Make sure these files are in `models/`:
- `crop_classifier_best.pt`
- `scaler.pkl`
- `app.py`

### Import Errors
```bash
pip install -r requirements.txt
```

### Timeout on Vercel
Use Railway instead - it has no timeout limits.

---

## 📚 Documentation

- **DEPLOYMENT.md** - Full deployment guide
- **README.md** - Complete documentation
- **test_local.py** - Pre-flight checks

---

## ✨ What It Looks Like

**Homepage:**
```
🌾 Crop Classifier
Identify crop types in Brazil using satellite imagery

┌─────────────────────────────────────┐
│ Latitude:    [-20.0502          ]   │
│ Longitude:   [-48.3965          ]   │
│ Year:        [2023              ]   │
└─────────────────────────────────────┘

         [🚀 Predict Crop Type]

📍 Try these locations:
[Soybean Region] [Sugar Cane] [Coffee] [Rice]
```

**Results:**
```
🎯 PREDICTION RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Soybean
Confidence: 89.2%

All Probabilities:
Soybean        89.2% ████████████████████
Pasture         6.1% ████
Other Crops     3.0% ██
...
```

---

## 🎉 You're Ready!

1. ✅ Test locally: `python app.py`
2. ✅ Deploy to Railway/Vercel
3. ✅ Share your link!

**Railway URL:** `https://your-app.railway.app`  
**Vercel URL:** `https://your-app.vercel.app`

---

**Need help?** Read [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions!

