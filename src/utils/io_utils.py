import io

class DevNull(io.StringIO):
    """
    A dummy stream to suppress output (acts like /dev/null).
    """
    def write(self, *args, **kwargs):
        return 0
