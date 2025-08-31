# COVID Chest Classifier

An end-to-end machine learning project that detects COVID-19 from chest X-ray images using deep learning and MLOps best practices.

## What it does
Classifies chest X-rays into three categories: COVID-19, Pneumonia, or Normal using a custom CNN model.

## Project Structure
```
covid_chest_classifier/
├── src/ccclassifier/     # Main ML code
├── artifacts/            # Saved models and results  
├── logs/                 # Training logs
├── notebooks/            # Jupyter notebooks
└── configs/              # Configuration files
```

## Quick Start

1. **Install**
   ```bash
   git clone https://github.com/Yusuf-Abol/endtoend-mlops.git
   cd endtoend-mlops/projects/covid_chest_classifier
   pip install -e .
   ```

2. **Train the model**
   ```bash
   python src/ccclassifier/pipeline/training_pipeline.py
   ```

3. **Make predictions**
   ```python
   from ccclassifier.pipeline.prediction import predict
   result = predict("your_xray.jpg")
   ```

## Key Features
- Automated data processing pipeline
- Custom CNN architecture for medical imaging
- Experiment tracking and logging
- Production-ready code structure
- CI/CD integration ready

## Tech Stack
- **ML Framework**: TensorFlow/PyTorch
- **Pipeline**: Custom modular design
- **Tracking**: Built-in logging system
- **Deployment**: Ready for cloud deployment

## Author
**Yusuf Abolarinwa** - [GitHub](https://github.com/Yusuf-Abol) | [Email](mailto:yusufabolarinwa@gmail.com)

## License
MIT License