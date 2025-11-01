import logging

class ExcludeFileFilter(logging.Filter):
    def __init__(self, library_names):
        super().__init__()
        self.library_names = library_names

    def filter(self, record):
        # Exclude records if their logger name starts with any of the specified library names
        for lib_name in self.library_names:
            if record.name.startswith(lib_name):
                return False
        return True