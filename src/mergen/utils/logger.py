import logging
import sys


def get_logger(name: str = "mergen", level: str = "INFO") -> logging.Logger:
    """Mergen-SNN için standart konsol logger'ı döndürür.

    Args:
        name: Logger ismi (modül veya bileşen adı).
        level: Log seviyesi ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").

    Returns:
        Yapılandırılmış logging.Logger nesnesi.
    """
    logger = logging.getLogger(name)

    # Log seviyesini ayarla
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Mükerrer handler eklenmesini engelle
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    logger.propagate = False
    return logger
