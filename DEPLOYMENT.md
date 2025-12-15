# 🚀 Deployment Guide - Crop Classifier Web App

This guide shows how to deploy your crop classifier to **Railway** (recommended) or **Vercel**.

---

## 📋 Prerequisites

1. Your model files in this directory:
   - `crop_classifier_best.pt` ✅
   - `scaler.pkl` ✅
   - `label_encoder.pkl` ✅
   - `app.py` ✅
   - `templates/index.html` ✅

2. Google Earth Engine authentication set up

---

## 🚂 Option 1: Railway (RECOMMENDED)

Railway is **better for this app** because:
- ✅ No strict timeout limits (GEE queries take 30-60 seconds)
- ✅ Persistent instances (faster after first load)
- ✅ Easy GEE authentication
- ✅ Free tier available

### Step-by-Step:

#### 1. Install Railway CLI (optional)
```bash
npm install -g @railway/cli
```

#### 2. Authenticate Google Earth Engine locally
```bash
# Install earthengine-api if needed
pip install earthengine-api

# Authenticate (opens browser)
earthengine authenticate

# This creates credentials at ~/.config/earthengine/credentials
```

#### 3. Deploy to Railway

**Option A: Using Railway CLI**
```bash
cd models/
railway login
railway init
railway up
```

**Option B: Using GitHub (easier)**
1. Push your `models/` folder to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Flask app ✅

#### 4. Set Environment Variables (Railway Dashboard)

After deployment, go to your project settings and add:

```bash
# Your GEE project ID (if using service account)
GEE_PROJECT_ID=your-project-id
```

#### 5. Copy EarthEngine Credentials

**Important:** You need to add your GEE credentials to Railway:

1. Find your local credentials:
   ```bash
   cat ~/.config/earthengine/credentials
   ```

2. In Railway Dashboard:
   - Go to your project → Variables
   - Add a new variable: `EARTHENGINE_TOKEN`
   - Paste the JSON content from credentials file

3. Update `app.py` to use this token (already handled in code)

#### 6. Done! 🎉

Your app will be live at: `https://your-app-name.railway.app`

---

## ▲ Option 2: Vercel (Alternative)

⚠️ **Warning:** Vercel has limitations:
- ❌ 60-second timeout (may fail for slow GEE queries)
- ❌ Serverless (cold starts)
- ❌ More complex GEE authentication

### Step-by-Step:

#### 1. Install Vercel CLI
```bash
npm install -g vercel
```

#### 2. Authenticate GEE and get token
```bash
earthengine authenticate

# Copy credentials
cat ~/.config/earthengine/credentials
```

#### 3. Deploy to Vercel
```bash
cd models/
vercel login
vercel
```

Follow the prompts:
- Project name: `crop-classifier`
- Framework: `Other`

#### 4. Add Environment Variables
```bash
# Add EarthEngine token
vercel env add EARTHENGINE_TOKEN
# Paste your credentials JSON when prompted

# Add GEE project (if using service account)
vercel env add GEE_PROJECT_ID
```

#### 5. Redeploy with variables
```bash
vercel --prod
```

#### 6. Done! 🎉

Your app will be live at: `https://your-app-name.vercel.app`

---

## 🧪 Test Locally First

Before deploying, test the app locally:

```bash
cd models/

# Install dependencies
pip install -r requirements.txt

# Authenticate GEE
earthengine authenticate

# Run Flask app
python app.py
```

Visit: `http://localhost:5000`

Try these test coordinates:
- **Soybean**: `-12.5, -55.7, 2023`
- **Sugar Cane**: `-21.5, -50.5, 2023`
- **Coffee**: `-21.2, -45.0, 2023`
- **Rice**: `-30.0, -51.0, 2023`

---

## 🐛 Troubleshooting

### "GEE authentication failed"
- Make sure you ran `earthengine authenticate`
- Check that credentials exist: `ls ~/.config/earthengine/`
- For Railway/Vercel: Add `EARTHENGINE_TOKEN` environment variable

### "Model not found"
- Ensure all model files are in the same directory as `app.py`
- Check file names match exactly: `crop_classifier_best.pt`, `scaler.pkl`

### "Request timeout"
- This happens when GEE query takes >60 seconds
- **Solution**: Use Railway instead of Vercel
- Or try a different location with better satellite coverage

### "Module not found"
- Make sure `requirements.txt` is in the same directory
- Railway/Vercel should auto-install dependencies

---

## 📊 Example API Usage

Once deployed, you can use the API:

### cURL
```bash
curl -X POST https://your-app.railway.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"lat": -12.5, "lon": -55.7, "year": 2023}'
```

### Python
```python
import requests

response = requests.post(
    'https://your-app.railway.app/api/predict',
    json={'lat': -12.5, 'lon': -55.7, 'year': 2023}
)
print(response.json())
```

### JavaScript
```javascript
fetch('https://your-app.railway.app/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({lat: -12.5, lon: -55.7, year: 2023})
})
.then(r => r.json())
.then(data => console.log(data));
```

---

## 🎨 Customization

### Change the UI theme
Edit `templates/index.html` - look for the `<style>` section

### Add more example locations
In `index.html`, add more example buttons:
```html
<span class="example-btn" onclick="fillExample(LAT, LON, YEAR)">Location Name</span>
```

### Change model
Replace `crop_classifier_best.pt` with another model file, and update `app.py` if needed

---

## 📱 Share Your App

After deployment, share the link:
- Railway: `https://your-app-name.railway.app`
- Vercel: `https://your-app-name.vercel.app`

Users can:
1. Click example locations
2. Or enter custom coordinates
3. Get instant crop predictions! 🌾

---

## 🚀 Next Steps

1. **Custom Domain**: Add your own domain in Railway/Vercel dashboard
2. **Analytics**: Add Google Analytics to track usage
3. **Rate Limiting**: Add rate limiting for production use
4. **Caching**: Cache GEE results to speed up repeat queries
5. **Map Integration**: Add interactive map for point selection

---

## 💡 Tips

- **Railway is recommended** for this use case (long-running requests)
- Test locally before deploying
- Monitor usage in Railway/Vercel dashboard
- GEE has rate limits - consider caching popular locations
- Keep model files under 50MB for faster deployments

---

## 📞 Support

If you run into issues:
1. Check Railway/Vercel logs
2. Test locally first
3. Verify GEE authentication
4. Check that all model files are uploaded

Happy deploying! 🎉

