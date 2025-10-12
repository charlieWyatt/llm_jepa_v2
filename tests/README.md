# Tests


## Running Tests

```bash
poetry run pytest
```

### Run tests for a specific module

```bash
poetry run pytest tests/maskers/
```

### Run a specific test file

```bash
poetry run pytest tests/maskers/test_random_masker.py
```

### Run a specific test

```bash
poetry run pytest tests/maskers/test_random_masker.py::TestRandomMasker::test_create_mask_returns_boolean_array
```

### Run with verbose output

```bash
poetry run pytest -v
```

### Run with coverage

```bash
poetry run pytest --cov=src --cov-report=html
```

## Test Philosophy

- **Mirror structure**: Tests mirror the `src` directory structure for easy navigation
- **Comprehensive coverage**: Each public method should have multiple test cases
- **Edge cases**: Tests include empty inputs, boundary conditions, and error cases
- **Reproducibility**: Tests use fixed seeds to ensure deterministic behavior
- **Clear naming**: Test names describe what is being tested
