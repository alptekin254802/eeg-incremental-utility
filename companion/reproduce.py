"""Run verified analysis and display commands into a fresh, separate directory."""
import sys
sys.dont_write_bytecode = True
from tools.portable_pipeline import main

if __name__ == '__main__':
    main()
