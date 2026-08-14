import json
import os
from typing import Dict, Any, List

# Preset konfigurasi parameter untuk berbagai model
PARAMETER_PRESETS = {
    # Classification Models
    "RandomForestClassifier": {
        "Default": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        "Fast Training": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"]
        },
        "High Accuracy": {
            "n_estimators": [200, 300, 500],
            "max_depth": [5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        "Small Dataset": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 7],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6],
            "max_features": ["sqrt", "log2"]
        }
    },
    
    "LogisticRegression": {
        "Default": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000]
        },
        "Fast Training": {
            "C": [0.1, 1.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [500]
        },
        "High Accuracy": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["saga"],
            "max_iter": [2000]
        },
        "Balanced Dataset": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "class_weight": ["balanced"]
        }
    },
    
    "SVC": {
        "Default": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        },
        "Fast Training": {
            "C": [0.1, 1.0],
            "kernel": ["linear"],
            "gamma": ["scale"]
        },
        "High Accuracy": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
        },
        "Linear Kernel": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear"],
            "gamma": ["scale"]
        }
    },
    
    "KNeighborsClassifier": {
        "Default": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        },
        "Fast Training": {
            "n_neighbors": [3, 5],
            "weights": ["uniform"],
            "metric": ["euclidean"]
        },
        "High Accuracy": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"]
        },
        "Small Dataset": {
            "n_neighbors": [1, 3, 5],
            "weights": ["uniform"],
            "metric": ["euclidean"]
        }
    },
    
    # Regression Models
    "RandomForestRegressor": {
        "Default": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        "Fast Training": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"]
        },
        "High Accuracy": {
            "n_estimators": [200, 300, 500],
            "max_depth": [5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        "Small Dataset": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 7],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6],
            "max_features": ["sqrt", "log2"]
        }
    },
    
    "SVR": {
        "Default": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "epsilon": [0.1, 0.5, 1.0]
        },
        "Fast Training": {
            "C": [0.1, 1.0],
            "kernel": ["linear"],
            "gamma": ["scale"],
            "epsilon": [0.1, 0.5]
        },
        "High Accuracy": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "epsilon": [0.01, 0.1, 0.5, 1.0, 2.0]
        },
        "Linear Kernel": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear"],
            "gamma": ["scale"],
            "epsilon": [0.1, 0.5, 1.0]
        }
    },
    
    "MLPRegressor": {
        "Default": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "max_iter": [200, 500, 1000]
        },
        "Fast Training": {
            "hidden_layer_sizes": [(50,), (100,)],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.01, 0.1],
            "max_iter": [200, 500]
        },
        "High Accuracy": {
            "hidden_layer_sizes": [(100,), (100, 50), (100, 100), (150, 100, 50)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "sgd", "lbfgs"],
            "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
            "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
            "max_iter": [500, 1000, 2000]
        },
        "Small Dataset": {
            "hidden_layer_sizes": [(50,), (100,)],
            "activation": ["relu", "tanh"],
            "solver": ["lbfgs", "adam"],
            "alpha": [0.001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.01],
            "max_iter": [500, 1000]
        }
    }
}

def get_available_presets(model_type: str) -> List[str]:
    """Mendapatkan daftar preset yang tersedia untuk model tertentu"""
    if model_type in PARAMETER_PRESETS:
        return list(PARAMETER_PRESETS[model_type].keys())
    return []

def get_preset_params(model_type: str, preset_name: str) -> Dict[str, Any]:
    """Mendapatkan parameter preset untuk model dan nama preset tertentu"""
    if model_type in PARAMETER_PRESETS and preset_name in PARAMETER_PRESETS[model_type]:
        return PARAMETER_PRESETS[model_type][preset_name].copy()
    return {}

def save_custom_preset(model_type: str, preset_name: str, params: Dict[str, Any], 
                        filepath: str = "custom_presets.json") -> bool:
    """Menyimpan preset kustom ke file"""
    try:
        # Load existing custom presets
        custom_presets = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                custom_presets = json.load(f)
        
        # Add new preset
        if model_type not in custom_presets:
            custom_presets[model_type] = {}
        
        custom_presets[model_type][preset_name] = params
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(custom_presets, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving custom preset: {e}")
        return False

def load_custom_presets(filepath: str = "custom_presets.json") -> Dict[str, Any]:
    """Memuat preset kustom dari file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading custom presets: {e}")
        return {}

def get_all_presets(model_type: str, custom_presets_file: str = "custom_presets.json") -> Dict[str, Dict[str, Any]]:
    """Mendapatkan semua preset (built-in dan custom) untuk model tertentu"""
    all_presets = {}
    
    # Add built-in presets
    if model_type in PARAMETER_PRESETS:
        all_presets.update(PARAMETER_PRESETS[model_type])
    
    # Add custom presets
    custom_presets = load_custom_presets(custom_presets_file)
    if model_type in custom_presets:
        all_presets.update(custom_presets[model_type])
    
    return all_presets

def export_preset_to_json(model_type: str, preset_name: str, params: Dict[str, Any], 
                           filename: str = None) -> str:
    """Mengekspor preset ke file JSON"""
    if filename is None:
        filename = f"{model_type}_{preset_name}_preset.json"
    
    export_data = {
        "model_type": model_type,
        "preset_name": preset_name,
        "parameters": params,
        "description": f"Parameter preset untuk {model_type} - {preset_name}"
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        return filename
    except Exception as e:
        print(f"Error exporting preset: {e}")
        return ""

def import_preset_from_json(filename: str) -> Dict[str, Any]:
    """Mengimpor preset dari file JSON"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        required_fields = ["model_type", "preset_name", "parameters"]
        if all(field in data for field in required_fields):
            return data
        else:
            print(f"Invalid preset file format. Required fields: {required_fields}")
            return {}
    except Exception as e:
        print(f"Error importing preset: {e}")
        return {}

def create_preset_summary(model_type: str, preset_name: str, params: Dict[str, Any]) -> str:
    """Membuat ringkasan deskriptif untuk preset"""
    summary = f"**{model_type} - {preset_name}**\n\n"
    summary += "Parameter ranges:\n"
    
    for param, values in params.items():
        if isinstance(values, list):
            if len(values) <= 5:
                summary += f"- {param}: {values}\n"
            else:
                summary += f"- {param}: [{values[0]}, {values[1]}, ..., {values[-1]}] ({len(values)} values)\n"
        else:
            summary += f"- {param}: {values}\n"
    
    return summary