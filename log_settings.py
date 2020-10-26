import logging

FORMAT = "[%(lineno)s -%(funcName) 10s()] -> %(message)s"

logging.basicConfig(format=FORMAT, level=logging.DEBUG)

lg = logging.getLogger()
