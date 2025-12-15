import sys
import os
import datetime
import logging
from pathlib import Path

class Tee(object):
    """
    Duplicate stdout/stderr to a file with timestamp and log level.
    """
    def __init__(self, filename, level="INFO"):
        self.terminal = sys.stdout if level == "INFO" else sys.stderr
        self.log_file = open(filename, "w", encoding='utf-8')
        self.level = level
        self.buffer = ""

    def write(self, message):
        # Write to terminal as-is
        self.terminal.write(message)
        
        # Buffer for log file to ensure we timestamp lines correctly
        self.buffer += message
        
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line: 
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Format: [Timestamp] [LEVEL] Message
                log_entry = f"[{timestamp}] [{self.level}] {line}\n"
                self.log_file.write(log_entry)
        
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.buffer:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{self.level}] {self.buffer}\n"
            self.log_file.write(log_entry)
            self.buffer = ""
        self.log_file.flush()

def setup_logging(script_name="pipeline"):
    """
    Sets up logging to logs/script_name.log using Stdout Hijacking.
    """
    # ... legacy implementation wrapper or keep original code ...
    # Re-implementing simplified path finding for legacy support
    CURRENT_DIR = Path(__file__).parent.absolute()
    LOG_DIR = CURRENT_DIR.parent / 'logs'
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    log_path = LOG_DIR / f"{script_name}.log"
    print(f"--- Logging started to {log_path} ---")
    sys.stdout = Tee(str(log_path), "INFO")
    sys.stderr = Tee(str(log_path), "ERROR")
    return log_path

def setup_logger(name, log_file, level=logging.INFO):
    """
    Standard Python Logging setup.
    Returns a logger object.
    """
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    handler = logging.FileHandler(log_file, mode='w') # Overwrite mode
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = [] # Clear existing
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger
