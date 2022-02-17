import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)-50s - %(asctime)s - %(levelname)-10s - %(message)s')
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)
