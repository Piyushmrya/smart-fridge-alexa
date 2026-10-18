# Load the project's .env (if python-dotenv is installed) so settings work on any OS.
try:
    from pathlib import Path as _P
    from dotenv import load_dotenv as _ld
    _ld(_P(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass
