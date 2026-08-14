# Testing Guide — Asmeranda AI

## Quick Start

### Installation
```bash
# Install all dependencies
make install

# Or manually:
pip install -r requirements.txt -r requirements-backend.txt -r requirements-dev.txt
```

### Setup Development Environment
```bash
make dev  # Installs pre-commit hooks
```

## Running Tests

### Run All Tests
```bash
make test
```

### Run Specific Test Types

**Unit Tests Only**
```bash
make test-unit
pytest backend/tests/unit -v
```

**Integration Tests Only**
```bash
make test-integration
pytest backend/tests/integration -v
```

**Specific Test File**
```bash
pytest backend/tests/unit/test_core_state.py -v
```

**Specific Test Class**
```bash
pytest backend/tests/unit/test_core_state.py::TestCoreState -v
```

**Specific Test Method**
```bash
pytest backend/tests/unit/test_core_state.py::TestCoreState::test_get_default_state -v
```

### Run Tests by Marker
```bash
# Run only fast tests (exclude slow)
pytest -m "not slow" -v

# Run security tests only
pytest -m security -v
```

## Coverage Reports

### Generate Coverage Report
```bash
make coverage
```

This will:
- Run all tests with coverage tracking
- Generate HTML report in `htmlcov/`
- Display coverage summary in terminal
- Create XML report for CI/CD

### View Coverage Report
```bash
# Open in browser
open htmlcov/index.html        # macOS
xdg-open htmlcov/index.html   # Linux
start htmlcov/index.html      # Windows PowerShell
```

### Coverage Targets
| Component | Target |
|-----------|--------|
| Backend | 80%+ |
| Core | 85%+ |
| Critical Paths | 100% |

## Code Quality

### Linting & Formatting

**Run All Quality Checks**
```bash
make lint
```

**Auto-Format Code**
```bash
make format
```

**Check Code Style (Black)**
```bash
make black
```

**Sort Imports (isort)**
```bash
make isort
```

**Lint with flake8**
```bash
make flake8
```

**Type Checking (mypy)**
```bash
make type-check
```

## Security Testing

### Run Security Checks
```bash
make security
```

### Security Scan (bandit)
```bash
make bandit
```

### Dependency Vulnerability Check
```bash
make safety
```

## Pre-Commit Hooks

### Install Hooks
```bash
make dev
```

### Run Hooks Manually
```bash
pre-commit run --all-files
```

### Skip Hooks (not recommended)
```bash
git commit --no-verify
```

## Test Structure

```
backend/tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_core_state.py
│   ├── test_workflow_validator.py
│   ├── test_error_handler.py
│   └── test_data_utils.py
├── integration/             # Integration tests (medium speed)
│   └── test_api_endpoints.py
└── security/                # Security tests
```

## Test Fixtures

Common fixtures available in `conftest.py`:

```python
def test_with_sample_data(sample_df_classification):
    """Use sample classification dataset."""
    assert len(sample_df_classification) == 12

def test_with_client(client):
    """Use FastAPI TestClient."""
    response = client.get("/health")
    assert response.status_code == 200

def test_with_temp_dir(temp_data_dir):
    """Use temporary directory."""
    csv_path = temp_data_dir / "test.csv"
    # ... work with file
```

Available fixtures:
- `sample_df_classification` - Sample classification dataset
- `sample_df_regression` - Sample regression dataset
- `sample_df_with_nulls` - Dataset with missing values
- `sample_df_categorical` - Categorical features dataset
- `sample_df_timeseries` - Time series dataset
- `temp_data_dir` - Temporary directory
- `sample_csv_file` - Sample CSV file
- `sample_parquet_file` - Sample Parquet file
- `client` - FastAPI TestClient
- `app` - FastAPI application
- `clean_state` - Clean core.state before test

## CI/CD Pipeline

### GitHub Actions Workflows

**Backend CI** (`.github/workflows/backend-ci.yml`)
- Runs on: push to main/develop, pull requests
- Tests on: Python 3.9, 3.10, 3.11
- Checks: linting, type checking, unit tests, integration tests
- Uploads: coverage to Codecov

**Frontend CI** (`.github/workflows/frontend-ci.yml`)
- Runs on: push to main/develop, pull requests (frontend changes)
- Checks: linting, formatting, tests, build
- Uploads: coverage to Codecov

**Security** (`.github/workflows/security.yml`)
- Runs: daily + on push/PR
- Checks: Bandit, Safety, pip-audit, npm audit
- Generates: detailed security reports

### Running Locally
```bash
# Simulate GitHub Actions locally (requires act)
act push -j test
```

## Common Issues & Solutions

### Issue: Tests Timeout
```bash
# Increase timeout (in seconds)
pytest --timeout=600
```

### Issue: Import Errors
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/asmeranda-modular"
pytest backend/tests
```

### Issue: Database/State Issues
```bash
# Clean state before testing
pytest backend/tests --clean-state
```

### Issue: Slow Tests
```bash
# Run tests in parallel
pytest -n auto backend/tests
```

## Writing New Tests

### Test Template
```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    """Test description."""
    
    def test_happy_path(self):
        """Test description."""
        # Arrange
        data = prepare_test_data()
        
        # Act
        result = function_under_test(data)
        
        # Assert
        assert result is not None
        assert result == expected_value
    
    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_under_test(invalid_data)
```

### Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Fixtures: descriptive names

### Best Practices
1. One assertion per test (when possible)
2. Clear test names describing behavior
3. Use fixtures for common setup
4. Mark tests with appropriate markers
5. Include docstrings
6. Keep tests isolated and independent

## Performance Testing

### Load Testing (Optional)
```bash
# Install Locust
pip install locust

# Run load tests
locust -f load_tests/locustfile.py -u 100 -r 10 --run-time 5m
```

## Debugging Tests

### Run with Detailed Output
```bash
pytest -vv --tb=long
```

### Debug Single Test
```bash
pytest --pdb backend/tests/unit/test_core_state.py::TestCoreState::test_get_default_state
```

### Print Debug Info
```python
def test_debug():
    print("Debug info here")
    pytest.set_trace()  # Breakpoint
```

### Run with Logging
```bash
pytest --log-cli-level=DEBUG
```

## Continuous Integration

### Before Pushing Code
```bash
# Run all checks locally
make all

# Or manually:
make lint
make type-check
make test
```

### Pre-Commit Checks
```bash
# Automatically run before commit
pre-commit run --all-files

# Show what will be checked
pre-commit run --all-files --dry-run
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Coverage Documentation](https://coverage.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/actions)

## Getting Help

```bash
# Show pytest help
pytest --help

# Show specific marker info
pytest --markers

# List available fixtures
pytest --fixtures
```

---

**Last Updated**: 2026-08-15
**Maintained By**: QA Team
