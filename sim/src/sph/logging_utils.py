from timeit import default_timer as timer

def sph_stage_logger(logger):
    def decorator(method):
        def wrapper(*args, **kwargs):
            logger.info(f"calling {method.__name__} function")
            s = timer()
            result = method(*args, **kwargs)
            logger.info(f"finished {method.__name__} call in {timer() - s:.4f} seconds")
        return wrapper
    return decorator