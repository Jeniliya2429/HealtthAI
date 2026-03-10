# 🚀 FREE DEPLOYMENT GUIDE
## Deploy HealthAI to the Internet — 100% Free

---

## 🏆 Best Free Option: Streamlit Community Cloud

**Why it's perfect:**
- ✅ 100% Free — no credit card needed
- ✅ Your app gets a public URL (e.g. `healthai-yourname.streamlit.app`)
- ✅ Runs 24/7 automatically
- ✅ Easy 3-step deploy from GitHub
- ✅ Auto-redeploys when you update code

---

## 📋 BEFORE YOU START — Checklist

- [ ] Project files downloaded and working locally
- [ ] `python train_models.py` has been run (models exist in `/models` folder)
- [ ] GitHub account (free at github.com)
- [ ] Streamlit account (free at streamlit.io)

---

## STEP 1 — Create a GitHub Account (if you don't have one)

1. Go to **https://github.com**
2. Click **"Sign Up"** (top right)
3. Enter your email, create a username and password
4. Verify your email
5. You're done! ✅

---

## STEP 2 — Install Git on Windows

1. Go to **https://git-scm.com/download/win**
2. Download and run the installer
3. Keep clicking **"Next"** with default settings
4. Open **Command Prompt** and type:
   ```
   git --version
   ```
   You should see something like `git version 2.44.0`

---

## STEP 3 — Create a New GitHub Repository

1. Log in to **github.com**
2. Click the **"+"** button (top right) → **"New repository"**
3. Fill in:
   - Repository name: `healthai-app`
   - Description: `AI Multi-Disease Risk Assessment`
   - Visibility: **Public** ← important for free hosting
   - ❌ Do NOT check "Add README" (we have our own)
4. Click **"Create repository"**
5. You'll see a page with setup instructions — **keep this open**

---

## STEP 4 — Upload Your Project to GitHub

Open **Command Prompt** and run these commands one by one:

```bash
# 1. Go to your project folder
cd Desktop\HealthAI\health_app

# 2. Initialize git
git init

# 3. Add all files
git add .

# 4. Make your first commit
git commit -m "Initial commit: HealthAI Multi-Disease App"

# 5. Connect to your GitHub repo
#    REPLACE 'YourUsername' with your actual GitHub username:
git remote add origin https://github.com/YourUsername/healthai-app.git

# 6. Push (upload) your code
git branch -M main
git push -u origin main
```

When prompted, enter your **GitHub username and password**.

> ⚠️ If GitHub asks for a token instead of password:
> Go to GitHub → Settings → Developer Settings → Personal Access Tokens →
> Tokens (classic) → Generate new token → Select "repo" scope → Copy token →
> Use that token as your password.

✅ Refresh your GitHub page — you should see all your files!

---

## STEP 5 — Create a Streamlit Account

1. Go to **https://streamlit.io/cloud**
2. Click **"Sign up"**
3. Click **"Continue with GitHub"** ← easiest option
4. Authorize Streamlit to access your GitHub
5. You're in! ✅

---

## STEP 6 — Deploy Your App

1. On Streamlit Cloud, click **"New app"** (blue button)
2. Fill in the form:
   - **Repository**: Select `YourUsername/healthai-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click **"Deploy!"**

The app will start building. You'll see logs scrolling.
**First deployment takes 3–5 minutes** (it's installing packages).

✅ When it says "Your app is live!", you'll get a URL like:
`https://healthai-app-yourname.streamlit.app`

---

## STEP 7 — Share Your App!

Copy your URL and share it with anyone.
They can register, log in, and use the health analyzer from anywhere! 🌍

---

## ⚠️ IMPORTANT: Handling the User Database on Streamlit Cloud

Streamlit Cloud resets the filesystem when the app redeploys.
This means `users.db` (user accounts) will be lost on redeploy.

**Solutions (pick one):**

### Option A — Accept it (for demo/college projects)
Users register fresh after each redeploy. Fine for demos.

### Option B — Commit users.db to GitHub (simple workaround)
After creating some test accounts locally:
```bash
git add users.db
git commit -m "Add initial user database"
git push
```
Note: Stored passwords are hashed — safe to commit.

### Option C — Use Free Cloud Database (production-grade)
Use **Supabase** (free PostgreSQL) or **MongoDB Atlas** (free).
This requires changing `auth.py` to use the cloud DB instead of SQLite.
Ask for help if you need this upgrade!

---

## 🔄 How to Update Your App After Changes

Every time you update your code:

```bash
cd Desktop\HealthAI\health_app
git add .
git commit -m "Update: describe what you changed"
git push
```

Streamlit Cloud will automatically redeploy within 30–60 seconds! ✅

---

## 🆓 Other Free Hosting Options

| Platform | Free Tier | Difficulty | Notes |
|----------|-----------|-----------|-------|
| **Streamlit Cloud** ⭐ | Unlimited | Easy | Best for Streamlit apps |
| **Render.com** | 750 hrs/month | Medium | Sleeps after 15 min idle |
| **Railway.app** | $5 credit/month | Medium | Faster cold starts |
| **Hugging Face Spaces** | Unlimited | Easy | Good for ML apps |
| **Replit** | Limited RAM | Easy | Good for testing |

---

## 🐛 Common Deployment Problems & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| App keeps restarting | Reduce RAM usage — check for memory leaks |
| Models not found | Make sure `models/` folder is in GitHub |
| Login not working | `users.db` may have reset — re-register |
| Slow first load | Normal — Streamlit takes 30–60s to wake up |
| `git push` rejected | Make sure you're on `main` branch |

---

## 📁 Files That MUST Be in GitHub

```
health_app/
├── app.py              ← REQUIRED
├── auth.py             ← REQUIRED (new!)
├── train_models.py     ← REQUIRED
├── requirements.txt    ← REQUIRED
├── utils/
│   ├── __init__.py     ← REQUIRED
│   ├── risk_assessment.py ← REQUIRED
│   └── preprocessing.py   ← REQUIRED
├── models/             ← REQUIRED (all 10 .pkl files)
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── ... (all 10)
└── .streamlit/
    └── config.toml     ← RECOMMENDED
```

---

## ✅ Final Checklist Before Deploying

- [ ] All files committed to GitHub
- [ ] `models/` folder with all 10 `.pkl` files included
- [ ] `requirements.txt` is complete
- [ ] App runs locally with `streamlit run app.py`
- [ ] Registration and login tested locally
- [ ] Streamlit account connected to GitHub

---

*You're all set! 🎉 Your app will be live on the internet for free.*
