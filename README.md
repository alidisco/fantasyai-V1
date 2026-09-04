# ⚡ FPL AI Pro — Autonomous Fantasy Premier League Manager

An intelligent, full-stack autonomous agent and dashboard designed to optimize, manage, and execute Fantasy Premier League (FPL) decisions 24/7 using Machine Learning and automated browser workflows.

---

## 🚀 Key Features

- 🧠 **ML-Driven Point Predictions**: Uses **XGBoost** and **Scikit-Learn** trained on historical fixture data, player form, ICT index, and underlying metrics to project player points.
- 🤖 **Autonomous AI Manager (Autopilot)**:
  - Automated deadline monitoring and team management.
  - Smart chip planning (Wildcard, Free Hit, Triple Captain, Bench Boost).
  - Configurable risk thresholds and hit allowances (e.g., max -4 hit limits).
- 🔄 **Optimal Squad & Transfer Solver**: Generates math-optimized 15-man squads and transfer suggestions within strict budget constraints.
- 🌐 **Browser-Based Execution**: Uses **Playwright** for automated, secure authentication and squad confirmation on the official FPL website.
- 📊 **Modern Cyberpunk UI**: Sleek, glassmorphic dark-mode dashboard with pitch visualization, live league tracking, and decision logs.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask
- **Data & ML**: Pandas, NumPy, Scikit-Learn, XGBoost
- **Automation**: Playwright (Headless Browser)
- **Frontend**: Vanilla HTML5, CSS3 Glassmorphism, Space Grotesk & Outfit typography

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/alidisco/fantasyai-V1.git
cd fantasyai-V1
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / Mac / Termux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
playwright install --with-deps chromium
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your details:
```env
FPL_EMAIL=your_fpl_email@example.com
FPL_PASSWORD=your_fpl_password
FPL_TEAM_ID=your_team_id
```

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 📱 Mobile / 24/7 Hosting (Termux)

This project can be run 24/7 on Android devices via **Termux (PRoot Ubuntu)**:

```bash
# Inside Termux Ubuntu
proot-distro login ubuntu
cd <project-folder>
source .venv/bin/activate
python app.py
```

---

## ⚠️ Disclaimer

This project is not officially affiliated with the Premier League or Fantasy Premier League. Use autopilot features at your own discretion.
