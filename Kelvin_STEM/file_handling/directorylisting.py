import os, fnmatch
"""
creates simple tools for file handling using existing repositories
"""

def file_list(path, searchterm = "*"):
    """
    Parameters
    ----------
    path: str
        A valid path for your system, best ended with "/"
    searchterm: str
        A text string you want to find in the filenames to be listed, which will need 
        to include wildcards if you are listing multiple files (e.g. *STEM*.h5)

    """
    files = []
    paths = os.listdir(path)
    if isinstance(searchterm, str):
        for file in paths:
            if fnmatch.fnmatch(file, searchterm): 
                #taking only files in data set that are using hdf5
                files += [file]
        return sorted(files)
    else:
        print('Error: use a text string as search term!')
        pass