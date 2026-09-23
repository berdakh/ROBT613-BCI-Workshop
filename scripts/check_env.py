import sys,importlib.metadata,shutil
print('Python:',sys.version.split()[0])
for package in ['mne','moabb','numpy','scipy','scikit-learn','matplotlib','nbformat','nbclient']:
    try: print(package,importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError: print(package,'MISSING')
print('Free disk (GB):',round(shutil.disk_usage('.').free/1e9,1))
print('Use a CPU runtime; notebook 13 additionally requires PyTorch.')
