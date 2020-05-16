import sisl as si
from io import StringIO

def get_io_sile(sile_cls):
    class IOSile(sile_cls):
        def __init__(self, filename, mode="r", comment=None, ioobj=None, **kwargs):
            super().__init__(filename, mode=mode, comment=comment, **kwargs)
            self.fh = self.set_io(ioobj)
            
        def set_io(self, io=None):
            if io is None:
                io = StringIO()
            #io._close = io.close
            #io.close = lambda: None
            self._io = io
            return io
        
        def _open(self):
            if self.fh is not self._io:
                # If %include in fdf
                super()._open()
            elif "w" in self._mode:
                self.fh = self.set_io()
            else:
                self.fh.seek(0)
    IOSile.__name__ = "IO" + sile_cls.__name__
    return IOSile