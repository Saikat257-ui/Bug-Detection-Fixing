# Code Bug Detection and Fix Recommendation System

This project uses Machine Learning to automatically identify bugs in code and suggest fixes using the Gemini API.

## Project Structure

- `data/`: Contains training and testing datasets
- `models/`: Saved model checkpoints
- `src/`: Source code for the project
  - `data_preparation.py`: Scripts to collect and prepare code data
  - `model.py`: ML model architecture for bug detection
  - `train.py`: Model training code
  - `evaluate.py`: Evaluation scripts and metrics calculation
  - `gemini_api.py`: Gemini API integration for fix recommendations
  - `utils.py`: Helper functions
- `main.py`: Entry point for the application
- `config.py`: Configuration settings
- `requirements.txt`: Required dependencies(compatible with python 3.11 64-bit)

## Setup

1. Clone this repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```
   # Create a .env file with your Gemini API key
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run this main script(preferably) to analyze all the code samples loaded in the 'samples' folder at a time:
```
python main.py
```

Run this main script to analyze all the code samples loaded in the 'samples' folder:
```
python main.py --analyze-file "path to that code sample"
```


## Features

- Automatic bug detection using ML models
- Bug classification by type
- Fix recommendations using Gemini API
- Performance metrics for model evaluation

## Team Members

1. Saikat Patra
   - Led the technical-development of the project, including designing, coding, training, and testing the machine learning and deep learning models. Took the lead in implementation, troubleshooting, and fine-tuning, ensuring the core engine of the Bug Detection and Fix Recommendation System functioned effectively and met the desired objectives.

2. Aishika Majumdar
   - Played a key role in coordinating team efforts and ensuring smooth communication and collaboration throughout the project. Took full responsibility for creating a comprehensive project report, aligning all documentation with technical developments, and maintaining consistent progress tracking. Also helped in quality-checking and aligning research with report requirements.


3. Pritam Chakrabortty
   - Contributed heavily to the foundational stage of the project by sourcing relevant datasets, researching similar work, and ensuring high-quality inputs for model training. Also assisted in filtering data, performing exploratory data analysis, and identifying useful resources that shaped the model development.

4. Soumavo Acharjee
   - Handled critical research and background study related to existing systems and models. Worked closely with the report and tech teams to validate technical content, support write-ups with solid references, and ensure timely delivery of tasks. Assisted in polishing the final documentation and reviewing every phase to maintain quality and deadlines.
