# 🔐 GitHub Account Setup for Deployment

## ⚡ QUICK: Local-Only Setup (Recommended)

Use a **different GitHub account ONLY for this repo** without affecting your global settings:

```bash
cd /Users/ibrahimahmed/Desktop/IDP2/models/

# Set username/email ONLY for this repository (not global)
git config user.name "YourDeploymentUsername"
git config user.email "deployment.email@gmail.com"

# Verify (should show your new username)
git config user.name

# Initialize and push
git init
git add .
git commit -m "Crop Classifier Web App"

# Use Personal Access Token for authentication (more secure)
# 1. Create token at: https://github.com/settings/tokens
# 2. When pushing, use token as password:
git remote add origin https://github.com/YOUR_USERNAME/crop-classifier.git
git push -u origin main
# Username: YourDeploymentUsername
# Password: <paste your GitHub token>
```

This way:
- ✅ Only affects this `models/` folder
- ✅ Your global Git config stays unchanged
- ✅ Other repos use your main account
- ✅ This repo uses your deployment account

---

## 📝 Steps to Deploy from Different GitHub Account (Detailed)

### 1. Check Current GitHub User
```bash
git config --global user.name
git config --global user.email
```

### 2. Switch to Your Deployment GitHub Account

**⭐ Option A: Use Per-Repo Config (RECOMMENDED - Only affects this repo)**
```bash
cd /Users/ibrahimahmed/Desktop/IDP2/models/

# Set username and email ONLY for this repository
git config user.name "YourNewUsername"
git config user.email "your.new.email@gmail.com"

# Verify it's set locally
git config user.name
git config user.email

# Check where it's configured (should say "local")
git config --show-origin user.name
```

**Option B: Change Global Git Config (affects ALL repos)**
```bash
git config --global user.name "YourNewUsername"
git config --global user.email "your.new.email@gmail.com"
```

### 3. Logout from GitHub CLI (if installed)
```bash
# Check if logged in
gh auth status

# Logout
gh auth logout

# Login with new account
gh auth login
# Choose: GitHub.com → HTTPS → Yes → Login with browser
```

### 4. Clear GitHub Credentials (macOS)

**⚠️ IMPORTANT:** On macOS, credentials are stored globally in Keychain. When you clear them, it affects all repos. However, you can use **per-repository credentials** instead.

**Method A: Clear All GitHub Credentials (affects all repos)**
```bash
# Remove stored credentials from Keychain
git credential-osxkeychain erase
# Then press Enter, type:
host=github.com
protocol=https
# Press Enter twice
```

**Method B: Use Personal Access Token (per-repo)**

This is better for multiple accounts - you'll set different tokens per repo:

1. Generate token on GitHub:
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Give it a name: "crop-classifier-deployment"
   - Check scopes: `repo`, `workflow`
   - Generate & copy the token (starts with `ghp_`)

2. When pushing, use token as password:
   ```bash
   git push
   # Username: YourNewUsername
   # Password: <paste your token here>
   ```

3. Or set remote URL with token (only for this repo):
   ```bash
   git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/crop-classifier.git
   ```

**Method C: Use Keychain Access:**
- Open "Keychain Access" app
- Search for "github.com"
- Delete all GitHub entries
- Next push will ask for new credentials

### 5. Create New Repo on GitHub

**Option A: Using GitHub Website**
1. Go to https://github.com/new
2. Repository name: `crop-classifier` (or any name)
3. Keep it **Public** (for free Railway/Vercel)
4. **Don't** initialize with README
5. Click "Create repository"

**Option B: Using GitHub CLI**
```bash
cd models/
gh repo create crop-classifier --public --source=. --remote=origin
```

### 6. Initialize Git in Models Folder
```bash
cd models/

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Crop Classifier Web App"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/crop-classifier.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 7. Verify on GitHub
Go to: `https://github.com/YOUR_USERNAME/crop-classifier`

You should see all your files! ✅

---

## 🚂 Now Deploy to Railway

### Method 1: GitHub Integration (Recommended)

1. Go to https://railway.app
2. Click "Login" → "Login with GitHub"
3. Make sure you're logged into the **correct GitHub account**
4. Click "New Project"
5. Click "Deploy from GitHub repo"
6. Select `crop-classifier` repo
7. Railway auto-detects Flask and deploys! 🎉

### Method 2: Railway CLI

```bash
cd models/

# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to new project
railway init

# Deploy
railway up
```

### Add Environment Variables in Railway:

1. Go to your Railway project
2. Click "Variables" tab
3. Add these:

```bash
# Your EarthEngine credentials (from ~/.config/earthengine/credentials)
EARTHENGINE_TOKEN=<paste your credentials JSON here>

# Optional: Your GEE project ID
GEE_PROJECT_ID=your-project-id
```

**Get your credentials:**
```bash
cat ~/.config/earthengine/credentials
```

Copy the entire JSON output and paste as `EARTHENGINE_TOKEN` value.

---

## ▲ Deploy to Vercel

### Method 1: GitHub Integration

1. Go to https://vercel.com
2. Click "Login" → "Continue with GitHub"
3. Use the **same GitHub account** as Railway
4. Click "Add New" → "Project"
5. Import `crop-classifier` repo
6. Click "Deploy"

### Method 2: Vercel CLI

```bash
cd models/

# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? (your account)
# - Link to existing project? No
# - Project name? crop-classifier
# - Directory? ./
# - Override settings? No
```

### Add Environment Variables in Vercel:

```bash
# Add EarthEngine token
vercel env add EARTHENGINE_TOKEN
# Paste your credentials JSON when prompted

# Redeploy with env vars
vercel --prod
```

---

## 🔐 SSH Keys (Alternative to HTTPS)

If you prefer SSH authentication:

### 1. Generate SSH Key
```bash
ssh-keygen -t ed25519 -C "your.email@gmail.com"
# Press Enter for default location
# Press Enter for no passphrase (or set one)
```

### 2. Add SSH Key to GitHub
```bash
# Copy public key to clipboard
pbcopy < ~/.ssh/id_ed25519.pub

# Or display it
cat ~/.ssh/id_ed25519.pub
```

Then:
1. Go to https://github.com/settings/keys
2. Click "New SSH key"
3. Paste your key
4. Click "Add SSH key"

### 3. Use SSH Remote
```bash
cd models/
git remote set-url origin git@github.com:YOUR_USERNAME/crop-classifier.git
git push
```

---

## 🧪 Verify Everything Works

### Test Git Push
```bash
cd models/
echo "# Test" >> README.md
git add README.md
git commit -m "Test commit"
git push
```

If it prompts for credentials, you're using HTTPS authentication.  
If it pushes without prompting, you're using SSH.

### Check Railway Deployment
- Railway URL: https://your-project.railway.app
- Check logs in Railway dashboard

### Check Vercel Deployment
- Vercel URL: https://crop-classifier.vercel.app
- Check logs in Vercel dashboard

---

## ⚠️ Common Issues

### "Authentication failed" when pushing
**Solution:**
```bash
# Clear credentials
git credential-osxkeychain erase
host=github.com
protocol=https
<press Enter twice>

# Or use SSH instead
git remote set-url origin git@github.com:YOUR_USERNAME/crop-classifier.git
```

### "Permission denied" on Railway/Vercel
**Solution:**
- Make sure repo is **Public** (not Private)
- Or upgrade to paid plan for private repos

### "GitHub account mismatch"
**Solution:**
- Logout from GitHub on browser
- Login with correct account
- Reconnect Railway/Vercel to GitHub

---

## 📋 Quick Checklist

- [ ] Switch GitHub account locally
- [ ] Clear old GitHub credentials
- [ ] Create new repo on GitHub
- [ ] Push models folder to repo
- [ ] Login to Railway with same GitHub account
- [ ] Deploy from GitHub on Railway
- [ ] Add EARTHENGINE_TOKEN to Railway
- [ ] Login to Vercel with same GitHub account
- [ ] Deploy from GitHub on Vercel
- [ ] Add EARTHENGINE_TOKEN to Vercel
- [ ] Test both deployments!

---

## 🎉 Done!

Your app should now be live on:
- **Railway:** https://your-app.railway.app
- **Vercel:** https://your-app.vercel.app

Share the link and start predicting crops! 🌾

