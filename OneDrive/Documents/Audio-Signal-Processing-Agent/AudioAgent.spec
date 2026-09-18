# AudioAgent.spec
# PyInstaller spec file for the Audio Signal Processing Agent desktop app

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all data files needed by librosa and flask
datas = []
datas += collect_data_files('librosa')
datas += collect_data_files('flask')
datas += collect_data_files('scipy')

# Include sample_audio folder if it exists
if os.path.isdir('sample_audio'):
    datas += [('sample_audio', 'sample_audio')]

# Hidden imports that PyInstaller misses
hiddenimports = (
    collect_submodules('librosa') +
    collect_submodules('scipy') +
    collect_submodules('sklearn') +
    collect_submodules('numba') +
    [
        'flask',
        'flask.templating',
        'jinja2',
        'werkzeug',
        'werkzeug.serving',
        'werkzeug.debug',
        'soundfile',
        'numpy',
        'scipy.signal',
        'scipy.fft',
        'scipy.io.wavfile',
        'audioread',
        'decorator',
        'packaging',
        'pooch',
        'lazy_loader',
        'msgpack',
        'resampy',
        'soxr',
        'app',
        'feature_extractor',
        'classifier',
        'generate_test_tones',
    ]
)

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PIL', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AudioSignalAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # no black console window — GUI only
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
