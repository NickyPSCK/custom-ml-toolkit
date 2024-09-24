from pyspark.sql.functions import udf


def create_udf(output_type):
    def decorator(func):
        def function_wrapper(*cols):
            return udf(func, output_type)(*cols)
        return function_wrapper
    return decorator