# NeuroHeaven Backend

Deep learning-based EEG signal processing and analysis API.

## Features

- **EEG Preprocessing**: Advanced signal processing pipeline with artifact removal
- **Deep Learning Models**: Autoencoders for EEG analysis
- **FastAPI Backend**: High-performance REST API
- **Production Ready**: Docker support, logging

## Project Structure

```
backend/
├── src/
│   ├── api/                    # API routes and endpoints
│   │   ├── routes/             # API route definitions
│   │   └── dependencies.py     # Dependency injection
│   ├── models/                 # Neural network models
│   │   ├── autoencoder.py
│   │   ├── variational_auto_encoder.py
│   │   └── EEGNet.py
│   ├── preprocessing/          # EEG preprocessing pipelines
│   │   └── eeg_preprocessing.py
│   ├── training/               # Model training scripts
│   │   └── train.py
│   ├── services/               # Business logic layer
│   ├── helper/                 # Utility functions
│   │   └── logger.py
│   └── main.py                 # Application entry point
├── data/                       # Data directory
│   ├── raw/                    # Raw EEG data
│   └── preprocessed/           # Preprocessed data
├── models/
│   └── checkpoints/            # Model checkpoints
├── notebooks/                  # Jupyter notebooks
├── test/                       # Test files
├── logs/                       # Application logs
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for faster training)
- Docker (optional, for containerized deployment)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the API at `http://localhost:8000`

## API Documentation

Once the server is running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

Configure the application using environment variables in `.env` file:

- **API_HOST**: Host address (default: 0.0.0.0)
- **API_PORT**: Port number (default: 8000)
- **LOG_LEVEL**: Logging level (DEBUG, INFO, WARNING, ERROR)
- **DEVICE**: Computing device (cuda/cpu)

See `.env.example` for all available options.

## Development

### Running Tests

```bash
pytest test/ -v
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

## Model Training

Train models using the training scripts:

```bash
python -m src.training.train --config configs/train_config.yaml
```

## Data Preprocessing

Preprocess EEG data:

```bash
python -m src.preprocessing.eeg_preprocessing --input data/raw --output data/preprocessed
```

## Deployment

### Production Deployment

1. Set `ENVIRONMENT=production` in `.env`
2. Update `SECRET_KEY` and other security settings
3. Use production-grade ASGI server:

```bash
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Health Checks

- **Health**: GET `/health`
- **Readiness**: GET `/ready`

## Monitoring

- Application logs: `logs/`
- Access logs: Configured in uvicorn
- Metrics endpoint: `/metrics` (if enabled)

## Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions, please open a GitHub issue.
