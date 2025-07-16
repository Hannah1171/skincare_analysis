# Peaks & Pies and Beiersdorf: Analyzing skincare trends in Genz from TikTok Data


This is a capstone project focused on skincare analysis using data science, NLP, and a Streamlit dashboard for visualization. TikTok Data is used and the data is scrapped and stored in Google Big Query

## 📁 Project Structure
```
ENV_CAPSTONE/
├── .streamlit/         # Streamlit configuration files
├── .venv/              # Python 
├── backend/            # Backend logic
├── dashboard/          # Streamlit app frontend
├── data/               # Raw and processed datasets
├── include/            
├── lib/              
├── notebooks/          # Jupyter notebooks for exploration
├── skincare/           # Skincare-specific modules or logic
├── .gitignore
├── .python-version
├── log.txt            
├── poetry.lock         # Poetry dependency lockfile
└── pyproject.toml    

 ```




## Getting Started

### Prerequisites

- Python (recommended via [`pyenv`](https://github.com/pyenv/pyenv))
- [Poetry](https://python-poetry.org/) for dependency management
- Streamlit

### Installation

```bash
# Clone the repository
git clone https://github.com/Hannah1171/skincare_analysis


# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Running the files
In order to get weekly updates, run the following command to pull, model and save the data files. This would take approximately an hour.
```bash
poetry run python -m backend.main
```

### Running the dashboard
After all the files are available, execute the following command to open the dashboard in web browser.
```bash
streamlit run dashboard/app.py
```
