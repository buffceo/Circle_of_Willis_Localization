# Circle of Willis Localization – MLOps Pipeline

## Overview

This project detects the Circle of Willis region from CTA/MRA/MRI scans using a slice classification model.

## Architecture

* Training: GPU machine
* Tracking: `config/config.json`
* Deployment: Docker + Flask API

## How to Run

### 1. Build Docker image

docker build -t cow-app .

### 2. Run container

docker run -p 5000:5000 cow-app

### 3. Open UI

http://localhost:5000

## Config

Training writes runtime metadata and metrics to:
`config/config.json`

Inference reads the saved model name, threshold, and preprocessing settings from the same file.

## Features

* Multi-modality support (CTA, MRA, MRI)
* JSON-based metric/config tracking
* Dynamic model loading
* Dockerized deployment
