# Audio Splicer

A simple tool for selecting, splicing, and saving audio clips with a visual interface.

## Features

- Load audio files from a folder
- Waveform visualization with time markers (ms)
- Selection tool for splicing
- 100ms padding option
- Save with tracking of used files

## Requirements

- Python 3.7+
- PyQt6
- numpy
- scipy
- soundfile
- sounddevice
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MuazAhmad7/audio-splicer.git
cd audio-splicer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python audio_splicer.py
```

1. Load audio files from a folder
2. Select an audio file from the list
3. Use the waveform to select the portion you want to splice
4. Optionally enable 100ms padding
5. Save the spliced audio clip

## License

MIT

