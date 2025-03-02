import logging

def configure_logging(file_name):
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a new file handler to store logs
    file_handler = logging.FileHandler(file_name, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Define the format of logs
    formatter = logging.Formatter(
        80 * "-"
        + "\n"
        + "%(asctime)s - %(levelname)s:\n"
        + "%(message)s\n"
        + 80 * "-"
        + "\n"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler) 