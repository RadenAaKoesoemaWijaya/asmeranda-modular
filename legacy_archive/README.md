# Legacy Archive

This directory contains obsolete files from the original Streamlit application that have been replaced by the new modular Next.js/FastAPI architecture.

## Archived Components

### Streamlit Application Files
- `app.py` - Original Streamlit application entry point
- `pages/` - Streamlit page components
- `utils.py` - Utility functions
- `requirements.txt` - Original Python dependencies

### Utility and Helper Files
- `advanced_ml.py` - Advanced ML utilities
- `anomaly_detection_utils.py` - Anomaly detection utilities
- `auth_db.py` - Authentication database functions
- `captcha_utils.py` - CAPTCHA utilities
- `data_type_detector.py` - Data type detection
- `db_pool.py` - Database connection pooling
- `error_handler.py` - Error handling
- `forecasting_utils.py` - Forecasting utilities
- `param_presets.py` - Parameter presets
- `priority2_functions.py` - Priority 2 functions
- `priority3_functions.py` - Priority 3 functions
- `run_app.py` - Application runner
- `session_manager.py` - Session management
- `task_manager.py` - Task management
- `workflow_validator.py` - Workflow validation

### Legacy Modules
- `modules/` - Legacy module structure
- `ml_engine/` - Legacy ML engine

### Test Files
- Various test files that have been superseded by the new test structure

## Migration Status

All functionality from these legacy files has been migrated to the new modular architecture:

- **Backend**: `backend/` directory with FastAPI
- **Frontend**: `frontend/` directory with Next.js
- **Core**: `core/` directory with shared utilities
- **Services**: `backend/services/` with modular service architecture

## Deletion Timeline

These files can be safely deleted after:
1. Final verification that all features work in the new architecture
2. User acceptance testing
3. Full deployment to production

## References

For the new architecture, see:
- Backend: `backend/` directory
- Frontend: `frontend/` directory
- API Documentation: Available at `/docs` endpoint when backend is running