from dotenv import load_dotenv
import os

load_dotenv()

def get_env(varname: str):
    value = os.getenv(varname)
    if not value:
        raise ValueError(f"ERROR: No se encontró la variable {varname} en .env")
    return value

def get_old_python():
    return get_env("OLD_PYTHON_PATH")

def get_medspaner_script():
    return get_env("MEDSPANER_SCRIPT")

def get_medspaner_config():
    return get_env("MEDSPANER_CONFIG")

def get_data_dir():
    return get_env("DATA_DIR")
def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key
