# Contributing to Body Sizer AI

Thank you for your interest in contributing! This document explains how to contribute in a way that helps maintain a clean and collaborative repository.

## How to contribute

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make changes in a clear and focused manner.
4. Write or update documentation when appropriate.
5. Submit a pull request describing the change.

## Development workflow

- Install dependencies with `pip install -r requirements.txt`
- Run the app locally using `python api/main.py`
- Run unit tests with `pytest`
- Run linting with `flake8 --max-line-length=100`

## Issues and pull requests

- Open an issue for bugs, feature requests, or improvements.
- Use descriptive titles and include relevant details.
- Reference issues in pull requests when applicable.

## Notes

- Model artifacts are kept out of Git by `.gitignore`.
- The `model/` directory should contain trained artifacts before running inference locally.
- If adding or changing API behavior, update the README and endpoint docs.
