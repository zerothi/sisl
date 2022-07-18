from pathlib import Path

from .file_source import FileData
import sisl

class FileDataSIESTA(FileData):
    """ Data source for reading data from a SIESTA calculation """    
    def get_sile(self, path=None, fdf=None, cls=None, **kwargs):
        """ Wrapper around FileData.get_sile that infers files from the root fdf

        Parameters
        ----------
        path : str or Path, optional
            the path to the file to be read.
        cls : sisl.io.SileSiesta, optional
            if `path` is not provided, we try to infer it from the root fdf file,
            looking for files that fullfill this class' rules.

        Returns
        ---------
        Sile:
            The sile object.
        """
        if fdf is not None and isinstance(fdf, (str, Path)):
            fdf = self.get_sile(path=fdf)

        if path is None:
            if cls is None:
                raise ValueError(f"Either a path or a class must be provided to {self.__class__.__name__}.get_sile")
            if fdf is None:
                raise ValueError(f"We can not look for files of a sile type without a root fdf file.")
            
            for rule in sisl.get_sile_rules(cls=cls):
                filename = fdf.get('SystemLabel', default='siesta') + f'.{rule.suffix}'
                try:
                    path = fdf.dir_file(filename)
                    return self.get_sile(path=path, **kwargs)
                except:
                    pass
            else:
                raise FileNotFoundError(f"Tried to find a {cls} from the root fdf ({fdf.file}), "
                f"but didn't find any.")

        return super().get_sile(path=path, **kwargs)