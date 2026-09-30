# Image Authenticity Classifier (Deployment Build)

A Flask web application that classifies uploaded images as **real** or **fake** using a custom Convolutional Neural Network trained with TensorFlow/Keras. This is the deployment-oriented version of the classifier: the application lives under `api/` for Vercel, the model is loaded from a repository-relative path, dependencies are pinned, and gunicorn is included as the WSGI server. A research proposal documenting the project is included as a PDF.

## What It Does

- Serves a single-page upload form styled with Tailwind CSS (loaded from CDN).
- Accepts an image via `POST /predict`, converts it to RGB, resizes it to 150x150, and scales pixel values to the 0-1 range.
- Runs the custom CNN (`Custom_CNN_best_model.h5`) on the preprocessed image.
- Reports `fake` when the model output is below 0.5, otherwise `real`, together with a rounded confidence value.
- Loads the Keras checkpoint with a custom `F1Score` metric implementation (precision/recall based).
- Serves `favicon.ico` from `api/static/`.

## Trained Models

The repository ships three Keras checkpoints:

| File                          | Role                                            |
|-------------------------------|-------------------------------------------------|
| `Custom_CNN_best_model.h5`    | Model loaded by the application at startup      |
| `InceptionV3_best_model.h5`   | Additional trained checkpoint (not loaded by the app) |
| `MobileNetV2_best_model.h5`   | Additional trained checkpoint (not loaded by the app) |

## Tech Stack

- Python 3.10 (pinned in `runtime.txt`), Flask 3.1.1, gunicorn 21.2.0
- TensorFlow 2.15.0 / Keras (inference, custom F1Score metric)
- NumPy 1.26.4, Pillow 10.3.0 (image preprocessing)
- Tailwind CSS via CDN (UI styling)
- Vercel (`@vercel/python` build, `vercel.json` routing)

## Project Structure

```
dep2/
├── api/
│   ├── app.py                    # Flask app: routes, preprocessing, UI template, model loading
│   └── static/
│       └── favicon.ico
├── Custom_CNN_best_model.h5     # Custom CNN checkpoint (loaded by the app)
├── InceptionV3_best_model.h5    # InceptionV3 checkpoint
├── MobileNetV2_best_model.h5    # MobileNetV2 checkpoint
├── Research Proposal.pdf         # Project research proposal
├── requirements.txt              # Pinned dependencies (flask, gunicorn, tensorflow, pillow, numpy)
├── runtime.txt                   # python-3.10.12
└── vercel.json                   # Routes all traffic to api/app.py
```

## Getting Started

### Prerequisites

- Python 3.10
- pip

### Install

```bash
pip install -r requirements.txt
```

### Run

The Flask application object is `app` in `api/app.py`. Serve it with the pinned WSGI server:

```bash
gunicorn api.app:app
```

The built-in `app.run(...)` development block is commented out in the source. Open the served URL, upload an image, and press **Classify Image**.

The model is loaded from the repository root (`Custom_CNN_best_model.h5`) using a relative path, so start the server from the project root directory.

## HTTP Interface

| Method | Path           | Description                                       |
|--------|----------------|---------------------------------------------------|
| GET    | `/`            | Upload form (HTML page)                           |
| POST   | `/predict`     | Multipart form field `image`; returns result page |
| GET    | `/favicon.ico` | Favicon from `api/static/`                        |

## Deployment

`vercel.json` builds `api/app.py` with `@vercel/python` and routes all requests (`/(.*)`) to it, so pushing the repository to a Vercel-connected project deploys the classifier directly. All dependencies are version-pinned for reproducible serverless builds.
