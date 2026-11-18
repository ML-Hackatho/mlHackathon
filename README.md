# PathSnap: GPS Map Matching and Route Reconstruction

## Team Contributions

### Person 1 – Data Processing and Feature Engineering
- Dataset loading and preprocessing
- Exploratory Data Analysis (EDA)
- GPS-to-road candidate generation
- Feature engineering and label alignment
- Logistic Regression and Random Forest baseline models

### Person 2 – Machine Learning Models and API
- LightGBM and XGBoost model development
- Hyperparameter tuning and model evaluation
- Model comparison and performance analysis
- Prediction API development
- Integration of model predictions with the HMM/Viterbi decoder

### Person 3 – HMM/Viterbi Route Decoder
- Hidden Markov Model (HMM) implementation
- Viterbi algorithm for optimal route reconstruction
- Observation and transition probability handling
- Integration with machine learning model predictions
- Final route reconstruction and evaluation

## Hackathon Deliverables

### Final Project Report
- [Team 16 Final Report (PDF)](Team16_Report.pdf)

### Technical Documentation

**Person 1 – Data Processing and Feature Engineering**
- [Person 1 Results](PERSON1_RESULTS.md)

**Person 2 – Machine Learning Models and API**
- [Person 2 Results and API Instructions](PERSON2_RESULTS.md)
- [Person 2 Integration Guide](PERSON2_INTEGRATION.md)

**Person 3 – HMM/Viterbi Route Decoder**
- HMM implementation and Viterbi decoder documentation
- Route reconstruction and evaluation details

**Shared Team Documentation**
- [Team Interface Specification](INTERFACE.md)

### Supporting Evidence
- `reports/technical_report_assets/` – Route maps and screenshots used as demonstration evidence.
- `reports/` – Report-generation analyses and comparison plots.

### API and UI Demonstration
The API and evaluator UI were demonstrated together on two laptops:
- **API:** Port `8000`
- **Evaluator UI:** Port `5500`

Refer to [Person 2 Integration Guide](PERSON2_INTEGRATION.md) for setup commands, CORS configuration, and network instructions.