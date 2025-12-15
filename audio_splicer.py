"""
Audio Splicer - A tool for selecting, splicing, and saving audio clips.

Features:
- Load audio files from a folder
- Waveform visualization with time markers (ms)
- Selection tool for splicing
- 100ms padding option
- Save with tracking of used files
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QSlider, QCheckBox, QSplitter, QFrame, QMessageBox, QGroupBox,
    QSpinBox, QStatusBar, QProgressBar, QScrollArea, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QColor, QFont

import soundfile as sf
import sounddevice as sd
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector


class WaveformCanvas(FigureCanvas):
    """Canvas for displaying audio waveform with selection capability."""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 4), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        self.setParent(parent)
        self.audio_data = None
        self.sample_rate = None
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None
        self.playback_line = None
        self.on_selection_change = None
        
        self._setup_axes()
        self._setup_selection()
        
    def _setup_axes(self):
        """Setup the axes styling."""
        self.ax.set_facecolor('#16213e')
        self.ax.tick_params(colors='#e0e0e0', labelsize=9)
        self.ax.spines['bottom'].set_color('#e0e0e0')
        self.ax.spines['left'].set_color('#e0e0e0')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_xlabel('Time (ms)', color='#e0e0e0', fontsize=10)
        self.ax.set_ylabel('Amplitude', color='#e0e0e0', fontsize=10)
        self.fig.tight_layout(pad=2)
        
    def _setup_selection(self):
        """Setup the span selector for audio selection."""
        # Disconnect old span selector if it exists
        if hasattr(self, 'span_selector') and self.span_selector is not None:
            self.span_selector.disconnect_events()
            
        self.span_selector = SpanSelector(
            self.ax, self._on_select, 'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='#00d9ff'),
            interactive=True,
            drag_from_anywhere=True
        )
        
    def _on_select(self, xmin, xmax):
        """Handle selection change."""
        if self.sample_rate is None:
            return
            
        # Convert ms to samples
        self.selection_start = max(0, int(xmin * self.sample_rate / 1000))
        self.selection_end = min(len(self.audio_data), int(xmax * self.sample_rate / 1000))
        
        # Update selection rectangle
        self._draw_selection()
        
        if self.on_selection_change:
            duration_ms = (self.selection_end - self.selection_start) / self.sample_rate * 1000
            self.on_selection_change(xmin, xmax, duration_ms)
            
    def _draw_selection(self):
        """Draw selection rectangle."""
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            
        if self.selection_start is not None and self.selection_end is not None:
            start_ms = self.selection_start / self.sample_rate * 1000
            end_ms = self.selection_end / self.sample_rate * 1000
            
            ymin, ymax = self.ax.get_ylim()
            self.selection_rect = Rectangle(
                (start_ms, ymin), end_ms - start_ms, ymax - ymin,
                alpha=0.25, facecolor='#00d9ff', edgecolor='#00d9ff', linewidth=2
            )
            self.ax.add_patch(self.selection_rect)
            
        self.draw()
        
    def load_audio(self, filepath):
        """Load and display audio file."""
        try:
            self.audio_data, self.sample_rate = sf.read(filepath)
            
            # Convert stereo to mono if needed
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
                
            # Clear selection state
            self.selection_start = None
            self.selection_end = None
            self.selection_rect = None
            
            self._plot_waveform()
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
            
    def _plot_waveform(self):
        """Plot the waveform."""
        self.ax.clear()
        self._setup_axes()
        
        if self.audio_data is None:
            self.draw()
            return
            
        # Create time axis in milliseconds
        duration_ms = len(self.audio_data) / self.sample_rate * 1000
        time_ms = np.linspace(0, duration_ms, len(self.audio_data))
        
        # Plot waveform
        self.ax.plot(time_ms, self.audio_data, color='#00d9ff', linewidth=0.5, alpha=0.8)
        self.ax.fill_between(time_ms, self.audio_data, alpha=0.3, color='#00d9ff')
        
        # Set limits
        self.ax.set_xlim(0, duration_ms)
        max_amp = np.max(np.abs(self.audio_data)) * 1.1
        self.ax.set_ylim(-max_amp, max_amp)
        
        # Add grid
        self.ax.grid(True, alpha=0.2, color='#e0e0e0')
        
        # Re-setup selection
        self._setup_selection()
        
        self.fig.tight_layout(pad=2)
        self.draw()
        
    def get_selection(self):
        """Get the selected audio data."""
        if self.selection_start is None or self.selection_end is None:
            return None, None
        return self.audio_data[self.selection_start:self.selection_end], self.sample_rate
        
    def update_playback_position(self, position_ms):
        """Update playback position indicator."""
        if self.playback_line:
            self.playback_line.remove()
            self.playback_line = None
            
        if position_ms is not None and self.audio_data is not None:
            ymin, ymax = self.ax.get_ylim()
            self.playback_line = self.ax.axvline(x=position_ms, color='#ff6b6b', linewidth=2)
            
        self.draw()


class AudioSplicer(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_folder = None
        self.current_file = None
        self.used_files = set()
        self.used_files_path = None
        self.output_folder = None
        self.is_playing = False
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playback)
        self.playback_start_time = 0
        self.playback_data = None
        self.playback_sr = None
        
        self._setup_ui()
        self._load_stylesheet()
        
    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Audio Splicer - Quran Dataset Tool")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel - File browser
        left_panel = self._create_file_panel()
        
        # Right panel - Waveform and controls
        right_panel = self._create_main_panel()
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        
        layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select a folder to load audio files")
        
    def _create_file_panel(self):
        """Create the file browser panel."""
        panel = QFrame()
        panel.setObjectName("filePanel")
        layout = QVBoxLayout(panel)
        
        # Folder selection
        folder_group = QGroupBox("Source Folder")
        folder_layout = QVBoxLayout(folder_group)
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_label)
        
        select_folder_btn = QPushButton("📁 Select Folder")
        select_folder_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(select_folder_btn)
        
        layout.addWidget(folder_group)
        
        # File list
        files_group = QGroupBox("Audio Files")
        files_layout = QVBoxLayout(files_group)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self._on_file_select)
        files_layout.addWidget(self.file_list)
        
        # File count label
        self.file_count_label = QLabel("0 files (0 used)")
        files_layout.addWidget(self.file_count_label)
        
        layout.addWidget(files_group)
        
        # Output folder
        output_group = QGroupBox("Output Folder")
        output_layout = QVBoxLayout(output_group)
        
        self.output_label = QLabel("No output folder selected")
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label)
        
        select_output_btn = QPushButton("📂 Select Output Folder")
        select_output_btn.clicked.connect(self._select_output_folder)
        output_layout.addWidget(select_output_btn)
        
        layout.addWidget(output_group)
        
        return panel
        
    def _create_main_panel(self):
        """Create the main waveform and controls panel."""
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget
        panel = QFrame()
        panel.setObjectName("mainPanel")
        layout = QVBoxLayout(panel)
        
        # Current file label
        self.current_file_label = QLabel("No file loaded")
        self.current_file_label.setObjectName("currentFileLabel")
        self.current_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_file_label)
        
        # Waveform canvas
        self.waveform = WaveformCanvas()
        self.waveform.on_selection_change = self._on_selection_change
        self.waveform.setMinimumHeight(300)
        self.waveform.setMaximumHeight(400)
        layout.addWidget(self.waveform)
        
        # Output Preview Section
        output_label = QLabel("📊 OUTPUT PREVIEW (What will be saved)")
        output_label.setObjectName("outputSectionLabel")
        layout.addWidget(output_label)
        
        self.output_canvas = WaveformCanvas()
        self.output_canvas.setMinimumHeight(200)
        self.output_canvas.setMaximumHeight(250)
        layout.addWidget(self.output_canvas)
        
        # Selection info
        selection_group = QGroupBox("Selection")
        selection_layout = QHBoxLayout(selection_group)
        
        self.selection_start_label = QLabel("Start: -- ms")
        self.selection_end_label = QLabel("End: -- ms")
        self.selection_duration_label = QLabel("Duration: -- ms")
        
        selection_layout.addWidget(self.selection_start_label)
        selection_layout.addWidget(self.selection_end_label)
        selection_layout.addWidget(self.selection_duration_label)
        
        layout.addWidget(selection_group)
        
        # Playback controls
        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout(playback_group)
        
        self.play_original_btn = QPushButton("▶ Play Original")
        self.play_original_btn.clicked.connect(self._play_original)
        playback_layout.addWidget(self.play_original_btn)
        
        self.play_selection_btn = QPushButton("▶ Play Selection")
        self.play_selection_btn.clicked.connect(self._play_selection)
        self.play_selection_btn.setEnabled(False)
        playback_layout.addWidget(self.play_selection_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        playback_layout.addWidget(self.stop_btn)
        
        layout.addWidget(playback_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)
        
        self.padding_checkbox = QCheckBox("Add 100ms padding (start & end)")
        self.padding_checkbox.setChecked(True)
        self.padding_checkbox.stateChanged.connect(self._update_output_preview)
        options_layout.addWidget(self.padding_checkbox)
        
        self.play_padded_btn = QPushButton("▶ Preview with Padding")
        self.play_padded_btn.clicked.connect(self._play_with_padding)
        self.play_padded_btn.setEnabled(False)
        options_layout.addWidget(self.play_padded_btn)
        
        layout.addWidget(options_group)
        
        # Save controls
        save_group = QGroupBox("Save")
        save_layout = QHBoxLayout(save_group)
        
        save_layout.addWidget(QLabel("Filename:"))
        
        self.filename_input = QLineEdit()
        self.filename_input.setPlaceholderText("Enter filename (without extension)")
        save_layout.addWidget(self.filename_input)
        
        self.save_btn = QPushButton("💾 Save Spliced Audio")
        self.save_btn.clicked.connect(self._save_audio)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        
        layout.addWidget(save_group)
        
        # Add some spacing at the bottom
        layout.addStretch()
        
        # Set the panel as scroll area content
        scroll_area.setWidget(panel)
        
        return scroll_area
        
    def _load_stylesheet(self):
        """Load the application stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f1a;
            }
            QFrame#filePanel, QFrame#mainPanel {
                background-color: #1a1a2e;
                border-radius: 8px;
                padding: 10px;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                color: #00d9ff;
                border: 1px solid #2d2d44;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 11px;
            }
            QLabel#currentFileLabel {
                font-size: 14px;
                font-weight: bold;
                color: #00d9ff;
                padding: 10px;
                background-color: #16213e;
                border-radius: 6px;
            }
            QLabel#outputSectionLabel {
                font-size: 13px;
                font-weight: bold;
                color: #ffd700;
                padding: 8px;
                margin-top: 10px;
                background-color: #16213e;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #2d2d44;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1f2b47;
                border-color: #00d9ff;
            }
            QPushButton:pressed {
                background-color: #00d9ff;
                color: #0f0f1a;
            }
            QPushButton:disabled {
                background-color: #0f0f1a;
                color: #4a4a6a;
                border-color: #1a1a2e;
            }
            QListWidget {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #2d2d44;
                border-radius: 6px;
                padding: 5px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #00d9ff;
                color: #0f0f1a;
            }
            QListWidget::item:hover {
                background-color: #1f2b47;
            }
            QCheckBox {
                color: #e0e0e0;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #2d2d44;
                background-color: #16213e;
            }
            QCheckBox::indicator:checked {
                background-color: #00d9ff;
                border-color: #00d9ff;
            }
            QLineEdit {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #2d2d44;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #00d9ff;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #16213e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #2d2d44;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00d9ff;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QStatusBar {
                background-color: #16213e;
                color: #e0e0e0;
                font-size: 11px;
            }
        """)
        
    def _select_folder(self):
        """Select source folder containing audio files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Audio Folder")
        if folder:
            self.current_folder = folder
            self.folder_label.setText(folder)
            self._load_files()
            
            # Load used files tracking
            self.used_files_path = os.path.join(folder, ".used_files.json")
            self._load_used_files()
            
    def _load_files(self):
        """Load audio files from selected folder."""
        self.file_list.clear()
        
        if not self.current_folder:
            return
            
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        files = []
        
        for f in os.listdir(self.current_folder):
            if Path(f).suffix.lower() in audio_extensions:
                files.append(f)
                
        files.sort()
        
        for f in files:
            item = QListWidgetItem(f)
            if f in self.used_files:
                item.setText(f"✓ {f}")
                item.setForeground(QColor("#4ade80"))
            self.file_list.addItem(item)
            
        self._update_file_count()
        self.status_bar.showMessage(f"Loaded {len(files)} audio files from folder")
        
    def _load_used_files(self):
        """Load the list of used files."""
        self.used_files = set()
        if self.used_files_path and os.path.exists(self.used_files_path):
            try:
                with open(self.used_files_path, 'r') as f:
                    self.used_files = set(json.load(f))
            except:
                pass
        self._refresh_file_list()
        
    def _save_used_files(self):
        """Save the list of used files."""
        if self.used_files_path:
            with open(self.used_files_path, 'w') as f:
                json.dump(list(self.used_files), f)
                
    def _refresh_file_list(self):
        """Refresh the file list display."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            filename = item.text().replace("✓ ", "")
            if filename in self.used_files:
                item.setText(f"✓ {filename}")
                item.setForeground(QColor("#4ade80"))
            else:
                item.setText(filename)
                item.setForeground(QColor("#e0e0e0"))
        self._update_file_count()
        
    def _update_file_count(self):
        """Update the file count label."""
        total = self.file_list.count()
        used = len(self.used_files)
        self.file_count_label.setText(f"{total} files ({used} used)")
        
    def _on_file_select(self, item):
        """Handle file selection."""
        filename = item.text().replace("✓ ", "")
        filepath = os.path.join(self.current_folder, filename)
        
        # Stop any ongoing playback before loading new file
        self._stop_playback()
        
        # Clear playback data
        self.playback_data = None
        self.playback_sr = None
        
        if self.waveform.load_audio(filepath):
            self.current_file = filename
            self.current_file_label.setText(f"📎 {filename}")
            self.play_original_btn.setEnabled(True)
            self.selection_start_label.setText("Start: -- ms")
            self.selection_end_label.setText("End: -- ms")
            self.selection_duration_label.setText("Duration: -- ms")
            self.play_selection_btn.setEnabled(False)
            self.play_padded_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            
            # Clear output preview
            self._update_output_preview()
            
            # Suggest filename
            base_name = Path(filename).stem
            self.filename_input.setText(f"{base_name}_spliced")
            
            self.status_bar.showMessage(f"Loaded: {filename}")
        else:
            QMessageBox.warning(self, "Error", f"Could not load audio file: {filename}")
            
    def _on_selection_change(self, start_ms, end_ms, duration_ms):
        """Handle selection change in waveform."""
        self.selection_start_label.setText(f"Start: {start_ms:.1f} ms")
        self.selection_end_label.setText(f"End: {end_ms:.1f} ms")
        self.selection_duration_label.setText(f"Duration: {duration_ms:.1f} ms")
        
        self.play_selection_btn.setEnabled(True)
        self.play_padded_btn.setEnabled(True)
        self.save_btn.setEnabled(self.output_folder is not None)
        
        self.status_bar.showMessage(f"Selected: {start_ms:.1f} - {end_ms:.1f} ms ({duration_ms:.1f} ms)")
        
        # Update output preview
        self._update_output_preview()
        
    def _update_output_preview(self):
        """Update the output preview showing what will be saved."""
        audio, sr = self.waveform.get_selection()
        
        # Clear if no selection
        if audio is None:
            self.output_canvas.ax.clear()
            self.output_canvas._setup_axes()
            self.output_canvas.ax.text(
                0.5, 0.5, 'Select audio to see output preview',
                ha='center', va='center', color='#888888', fontsize=12,
                transform=self.output_canvas.ax.transAxes
            )
            self.output_canvas.draw()
            return
        
        # Get padding setting
        add_padding = self.padding_checkbox.isChecked()
        
        # Build output audio
        if add_padding:
            padding_samples = int(0.1 * sr)  # 100ms
            padding = np.zeros(padding_samples)
            output_audio = np.concatenate([padding, audio, padding])
            padding_ms = 100
        else:
            output_audio = audio
            padding_ms = 0
        
        # Clear and setup
        self.output_canvas.ax.clear()
        self.output_canvas._setup_axes()
        
        # Create time axis
        duration_ms = len(output_audio) / sr * 1000
        time_ms = np.linspace(0, duration_ms, len(output_audio))
        
        # Plot waveform
        self.output_canvas.ax.plot(time_ms, output_audio, color='#00d9ff', linewidth=0.8, alpha=0.9)
        self.output_canvas.ax.fill_between(time_ms, output_audio, alpha=0.4, color='#00d9ff')
        
        # Add visual regions if padding is enabled
        if add_padding:
            ymin, ymax = self.output_canvas.ax.get_ylim()
            
            # Start padding region (yellow)
            self.output_canvas.ax.axvspan(0, padding_ms, alpha=0.25, color='#ffd700', zorder=0)
            self.output_canvas.ax.axvline(padding_ms, color='#ffd700', linewidth=2, linestyle='--', alpha=0.8)
            
            # End padding region (yellow)
            end_padding_start = duration_ms - padding_ms
            self.output_canvas.ax.axvspan(end_padding_start, duration_ms, alpha=0.25, color='#ffd700', zorder=0)
            self.output_canvas.ax.axvline(end_padding_start, color='#ffd700', linewidth=2, linestyle='--', alpha=0.8)
            
            # Audio region (green tint)
            self.output_canvas.ax.axvspan(padding_ms, end_padding_start, alpha=0.15, color='#4ade80', zorder=0)
            
            # Add text annotations
            text_y_pos = ymax * 0.75
            
            # Start padding label
            self.output_canvas.ax.text(
                padding_ms / 2, text_y_pos,
                '100ms\nSILENCE', ha='center', va='center',
                color='#ffd700', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', edgecolor='#ffd700', linewidth=2)
            )
            
            # Audio section label
            audio_duration = duration_ms - 200
            self.output_canvas.ax.text(
                padding_ms + (end_padding_start - padding_ms) / 2, text_y_pos,
                f'AUDIO\n{audio_duration:.1f}ms', ha='center', va='center',
                color='#4ade80', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', edgecolor='#4ade80', linewidth=2)
            )
            
            # End padding label
            self.output_canvas.ax.text(
                end_padding_start + padding_ms / 2, text_y_pos,
                '100ms\nSILENCE', ha='center', va='center',
                color='#ffd700', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', edgecolor='#ffd700', linewidth=2)
            )
            
            # Add timing markers at boundaries
            self.output_canvas.ax.text(
                0, ymin * 0.9, '0ms',
                ha='left', va='bottom', color='#e0e0e0', fontsize=9, fontweight='bold'
            )
            self.output_canvas.ax.text(
                padding_ms, ymin * 0.9, f'{padding_ms:.0f}ms',
                ha='center', va='bottom', color='#ffd700', fontsize=9, fontweight='bold'
            )
            self.output_canvas.ax.text(
                end_padding_start, ymin * 0.9, f'{end_padding_start:.0f}ms',
                ha='center', va='bottom', color='#ffd700', fontsize=9, fontweight='bold'
            )
            self.output_canvas.ax.text(
                duration_ms, ymin * 0.9, f'{duration_ms:.0f}ms',
                ha='right', va='bottom', color='#e0e0e0', fontsize=9, fontweight='bold'
            )
        else:
            # No padding - just show audio duration
            self.output_canvas.ax.text(
                duration_ms / 2, self.output_canvas.ax.get_ylim()[1] * 0.75,
                f'AUDIO\n{duration_ms:.1f}ms', ha='center', va='center',
                color='#00d9ff', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', edgecolor='#00d9ff', linewidth=2)
            )
        
        # Set limits
        self.output_canvas.ax.set_xlim(-5, duration_ms + 5)
        max_amp = np.max(np.abs(output_audio)) * 1.15 if len(output_audio) > 0 else 1
        self.output_canvas.ax.set_ylim(-max_amp, max_amp)
        
        # Grid
        self.output_canvas.ax.grid(True, alpha=0.2, color='#e0e0e0')
        
        # Title with total duration
        title_text = f'Total Output: {duration_ms:.1f}ms'
        if add_padding:
            title_text += f' (100ms + {duration_ms-200:.1f}ms + 100ms)'
        self.output_canvas.ax.set_title(title_text, color='#ffd700', fontsize=11, fontweight='bold', pad=10)
        
        self.output_canvas.fig.tight_layout(pad=2)
        self.output_canvas.draw()
    
    def _select_output_folder(self):
        """Select output folder for saved files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_label.setText(folder)
            
            # Enable save if we have a selection
            if self.waveform.selection_start is not None:
                self.save_btn.setEnabled(True)
                
    def _play_original(self):
        """Play the original audio file."""
        if self.waveform.audio_data is None:
            return
            
        self._stop_playback()
        self.playback_data = self.waveform.audio_data
        self.playback_sr = self.waveform.sample_rate
        self._start_playback()
        
    def _play_selection(self):
        """Play the selected portion."""
        audio, sr = self.waveform.get_selection()
        if audio is None:
            return
            
        self._stop_playback()
        self.playback_data = audio
        self.playback_sr = sr
        self._start_playback()
        
    def _play_with_padding(self):
        """Play selection with padding."""
        audio, sr = self.waveform.get_selection()
        if audio is None:
            return
            
        self._stop_playback()
        
        # Add 100ms padding
        padding_samples = int(0.1 * sr)
        padding = np.zeros(padding_samples)
        self.playback_data = np.concatenate([padding, audio, padding])
        self.playback_sr = sr
        self._start_playback()
        
    def _start_playback(self):
        """Start audio playback."""
        if self.playback_data is None:
            return
            
        self.is_playing = True
        self.playback_start_time = 0
        sd.play(self.playback_data, self.playback_sr)
        self.playback_timer.start(50)  # Update every 50ms
        self.status_bar.showMessage("Playing...")
        
    def _stop_playback(self):
        """Stop audio playback."""
        sd.stop()
        self.is_playing = False
        self.playback_timer.stop()
        self.waveform.update_playback_position(None)
        
    def _update_playback(self):
        """Update playback position indicator."""
        if not self.is_playing:
            self.playback_timer.stop()
            return
            
        # Check if playback finished
        if not sd.get_stream() or not sd.get_stream().active:
            self._stop_playback()
            self.status_bar.showMessage("Playback finished")
            return
            
    def _save_audio(self):
        """Save the spliced audio."""
        if not self.output_folder:
            QMessageBox.warning(self, "Error", "Please select an output folder first.")
            return
            
        audio, sr = self.waveform.get_selection()
        if audio is None:
            QMessageBox.warning(self, "Error", "Please select a portion of the audio first.")
            return
            
        filename = self.filename_input.text().strip()
        if not filename:
            QMessageBox.warning(self, "Error", "Please enter a filename.")
            return
            
        # Add padding if checked
        if self.padding_checkbox.isChecked():
            padding_samples = int(0.1 * sr)
            padding = np.zeros(padding_samples)
            audio = np.concatenate([padding, audio, padding])
            
        # Save file
        output_path = os.path.join(self.output_folder, f"{filename}.wav")
        
        # Check if file exists
        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self, "File Exists",
                f"File '{filename}.wav' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
                
        try:
            sf.write(output_path, audio, sr)
            
            # Mark original file as used
            if self.current_file:
                self.used_files.add(self.current_file)
                self._save_used_files()
                self._refresh_file_list()
                
            self.status_bar.showMessage(f"Saved: {output_path}")
            QMessageBox.information(self, "Success", f"Audio saved to:\n{output_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save audio:\n{str(e)}")
            
    def closeEvent(self, event):
        """Handle window close."""
        self._stop_playback()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = AudioSplicer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

