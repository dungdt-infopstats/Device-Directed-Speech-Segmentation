# TV Command Synthesis Pipeline

A comprehensive pipeline for synthesizing TV command speech data through multiple phases, from text generation to noise-augmented audio with accurate segmentation labels.

## 🎯 Overview

This pipeline generates synthetic speech data for TV/device command recognition systems. It processes data through four main phases:

1. **Phase 1**: Text synthesis and generation using LLMs
2. **Phase 2**: Speech synthesis using F5-TTS with reference audio
3. **Phase 3**: Speech cleaning with force alignment and concatenation
4. **Phase 4**: Noise augmentation for robustness

## 🏗️ Architecture

```
src/
├── core/                    # Base abstractions and configuration
├── data/                    # Data loading and handling utilities
├── phases/                  # Individual pipeline phases
│   ├── phase1_text_synthesis/
│   ├── phase2_speech_synthesis/
│   ├── phase3_force_alignment/    # Speech cleaning
│   └── phase4_noise_augmentation/
├── utils/                   # Common utilities
└── pipeline/                # Pipeline orchestration
```

### Key Features

- **Data Abstraction**: JSON-based data format with DataFrame compatibility
- **Modular Design**: Each phase can run independently or as part of full pipeline
- **Text Label Integration**: Active text tracking through all phases
- **Configurable**: YAML-based configuration system
- **Parallel Processing**: Multi-worker support for compute-intensive phases
- **Error Handling**: Robust error handling and logging

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for F5-TTS)
- OpenAI API key (for Phase 1)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd TV-command-synthesis
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install additional dependencies:**
```bash
# Install F5-TTS for Phase 2
pip install f5-tts

# Install faster-whisper for Phase 3 force alignment
pip install faster-whisper
```

5. **Set up environment variables:**
```bash
# Set OpenAI API key for Phase 1
export OPENAI_API_KEY="your-openai-api-key-here"

# Windows
set OPENAI_API_KEY=your-openai-api-key-here
```

6. **Download external datasets (optional):**
   - **VCTK Corpus** for Phase 2 reference audio: [Download here](https://datashare.ed.ac.uk/handle/10283/3443)
   - **MUSAN Dataset** for Phase 4 noise augmentation: [Download here](https://www.openslr.org/17/)

7. **Verify installation:**
```bash
python main.py --validate
```

## 📊 Data Structure

### Input Data Format

**Content Data** (`data/content/*.json`):
```json
[
  {"content": "Spotify"},
  {"content": "Netflix"},
  {"content": "YouTube"}
]
```

**Commands Data** (`data/commands/commands.json`):
```json
[
  {"command": "play"},
  {"command": "stop"},
  {"command": "search"}
]
```

### Output Data Format

**Phase 3 Annotations** (`concat_speech/{file_name}/{file_name}.json`):
```json
{
  "text_labels": "Play Netflix stop playing",
  "segments": [
    {"label": "active", "start": 0.0, "end": 1.2},
    {"label": "non_active", "start": 1.2, "end": 2.8},
    {"label": "active", "start": 2.8, "end": 3.5}
  ]
}
```

**Phase 4 Metadata** (`augmented/{file_name}/{file_name}_aug.json`):
```json
{
  "clean_path": "/path/to/clean/audio.wav",
  "noise_path": "/path/to/noise/audio.wav",
  "mode": "overlap",
  "snr_db": 15.2,
  "text_labels": "Play Netflix stop playing",
  "labels": [
    {"label": "active", "start": 0.0, "end": 1.2},
    {"label": "noise", "start": 0.5, "end": 1.7}
  ]
}
```

## 🚀 Usage

### Quick Start

1. **Validate your setup:**
```bash
python main.py --validate
```

2. **Run the complete pipeline:**
```bash
python main.py --run-all
```

3. **Check results:**
```bash
# Check output in data/processed/
ls data/processed/
```

### Command Line Interface

```bash
# Basic usage
python main.py --run-all                    # Run complete pipeline
python main.py --phases phase1 phase2       # Run specific phases
python main.py --validate                   # Validate setup
python main.py --status                     # Show pipeline status

# Advanced usage
python main.py --config configs/custom_config.yaml --run-all  # Custom config
python main.py --phases phase4 --custom-mode                  # Phase 4 with multiple SNR
python main.py --output-dir /custom/path --run-all            # Custom output directory
python main.py --run-all --verbose                            # Enable verbose logging

# Phase-specific examples
python main.py --phases phase1              # Generate text only
python main.py --phases phase2              # Synthesize speech only
python main.py --phases phase3 phase4       # Audio processing only
```

### Python API

```python
from src.pipeline.pipeline_executor import PipelineExecutor

# Initialize pipeline
pipeline = PipelineExecutor('configs/pipeline_config.yaml')

# Validate setup first
validation = pipeline.validate_setup()
if not validation['valid']:
    print("Setup issues:", validation['issues'])
    exit(1)

# Run complete pipeline
print("Running complete pipeline...")
results = pipeline.run_full_pipeline()

if results['success']:
    print(f"✅ Pipeline completed in {results['total_execution_time']:.2f}s")
    for phase, result in results['phases'].items():
        print(f"  {phase}: {result['output_records']} records")
else:
    print("❌ Pipeline failed:", results['errors'])

# Run individual phases
phase1_result = pipeline.run_phase1()
phase2_result = pipeline.run_phase2()
phase3_result = pipeline.run_phase3()
phase4_result = pipeline.run_phase4(custom_mode=True)
```

### Configuration Customization

```python
from src.core.config import Config

# Load and modify configuration
config = Config('configs/pipeline_config.yaml')

# Customize Phase 1 settings
config.set('phase1.text_gen_param.model', 'gpt-4')
config.set('phase1.batch_config.file_num', 10)

# Customize Phase 2 settings
config.set('phase2.num_workers', 2)
config.set('phase2.model_config.device', 'cpu')

# Save modified configuration
config.save_config('configs/my_custom_config.yaml')

# Use custom configuration
pipeline = PipelineExecutor('configs/my_custom_config.yaml')
```

### Processing Workflow

```bash
# Step 1: Prepare your data
ls data/content/    # Should contain: app.json, movie.json, song.json, tv.json
ls data/commands/   # Should contain: commands.json

# Step 2: Validate setup
python main.py --validate

# Step 3: Run pipeline phases
python main.py --phases phase1      # Generate text (requires OpenAI API)
python main.py --phases phase2      # Synthesize speech (requires F5-TTS)
python main.py --phases phase3      # Clean and concatenate audio (requires Whisper)
python main.py --phases phase4      # Add noise augmentation

# Step 4: Check outputs
ls data/processed/phase1/   # Generated text JSON files
ls data/processed/phase2/   # Synthesized speech files
ls data/processed/phase3/   # Cleaned and concatenated audio with labels
ls data/processed/phase4/   # Final augmented audio dataset
```

### Example: Running Only Text Generation

```bash
# Generate text without speech synthesis
python main.py --phases phase1

# Check generated text
cat data/processed/phase1/batch_100.json
```

### Example: Running with External Data

```bash
# Set paths to external datasets in config
python main.py --config configs/pipeline_config.yaml --run-all

# Or override paths programmatically
python -c "
from src.pipeline.pipeline_executor import PipelineExecutor
pipeline = PipelineExecutor()
pipeline.config.set('phase2.ref_folder_path', '/path/to/vctk/audio')
pipeline.config.set('phase4.noise_folders', ['/path/to/musan/speech'])
results = pipeline.run_full_pipeline()
"
```

## ⚙️ Configuration

### Pipeline Configuration (`configs/pipeline_config.yaml`)

```yaml
# Data directories
data:
  content_dir: "data/content"
  commands_dir: "data/commands"
  output_dir: "data/processed"

# Phase 1: Text synthesis
phase1:
  prompt_gen_param:
    num_samples_command: 5
    num_samples_content: 5
    generated_nums: 4
  text_gen_param:
    model: "gpt-4o-mini"
    temperature: 1.0

# Phase 2: Speech synthesis
phase2:
  ref_folder_path: "data/external/vctk/filtered_audio"
  json_folder_path: "data/external/vctk/filtered_json"
  num_workers: 4

# Phase 4: Noise augmentation
phase4:
  noise_folders:
    - "data/external/musan/speech/us-gov"
    - "data/external/musan/speech/librivox"
  snr_range: [-5, 20]
```

## 🔧 Phase Details

### Phase 1: Text Synthesis

Generates text commands using LLM prompting with five types:
- `non_active`: Human conversation without commands
- `single_active`: Single TV/device command
- `single_mix`: Single command + conversation
- `chain_active`: Multiple commands only
- `chain_mix`: Multiple commands + conversation

**Key Components:**
- `DataPreparation`: Abstract data loading
- `PromptGenerator`: Template-based prompt generation
- `TextGenerator`: LLM API integration

### Phase 2: Speech Synthesis

Converts text to speech using F5-TTS with reference-based generation:
- Intelligent reference selection based on text length
- Segment-wise synthesis for mix types
- Multi-worker parallel processing

**Key Components:**
- `ReferenceCache`: VCTK reference management
- `SpeechSynthesis`: F5-TTS integration

### Phase 3: Speech Cleaning

Combines force alignment and audio concatenation:
- **Force Alignment**: Uses Whisper to trim audio to actual speech content
- **Audio Concatenation**: Combines segments with precise timing
- Text label extraction from Phase 2
- Enhanced JSON format with active text

**Key Components:**
- `OptimizedSpeechCleaning`: Whisper-based force alignment
- `AudioConcatenator`: Audio processing and labeling
- `get_active_text_labels()`: Text extraction utility

### Phase 4: Noise Augmentation

Adds environmental noise for robustness:
- Multiple augmentation modes: overlap, prepend, append
- SNR-based noise scaling
- Label preservation and enhancement

**Key Components:**
- `NoiseAugmentor`: Noise processing and integration

## 📁 Data Directory Structure

```
data/
├── content/               # Content JSON files
│   ├── app.json
│   ├── movie.json
│   ├── song.json
│   └── tv.json
├── commands/              # Command definitions
│   └── commands.json
├── external/              # External datasets
│   ├── vctk/             # VCTK reference data
│   └── musan/            # MUSAN noise data
└── processed/            # Pipeline outputs
    ├── phase1/           # Generated text
    ├── phase2/           # Synthesized speech
    ├── phase3/           # Cleaned and concatenated audio
    │   ├── trimmed_speech/     # Force-aligned audio
    │   └── concatenated_speech/ # Final concatenated audio with labels
    └── phase4/           # Augmented final data
```

## 🔍 Monitoring and Debugging

### Logging

The pipeline includes comprehensive logging:
```python
# Enable verbose logging
python main.py --run-all --verbose
```

### Validation

Check setup before running:
```bash
python main.py --validate
```

### Status Monitoring

Check pipeline status:
```bash
python main.py --status
```

## 🐛 Troubleshooting

### Common Issues

1. **F5-TTS Installation**:
   ```bash
   pip install f5-tts
   # If CUDA issues, check PyTorch installation
   ```

2. **Memory Issues**:
   - Reduce `num_workers` in configuration
   - Use smaller batch sizes in Phase 1

3. **Missing Data**:
   - Run `python main.py --validate` to check data availability
   - Ensure VCTK and MUSAN datasets are properly extracted

4. **Audio Processing Errors**:
   - Check audio file formats (WAV recommended)
   - Verify sample rates match configuration

## 📈 Performance Optimization

### GPU Usage
- Phase 2 benefits most from GPU acceleration
- Set `device: "cuda"` in phase2 configuration

### Parallel Processing
- Adjust `num_workers` based on available CPU/GPU memory
- Phase 1 and 2 support multi-worker processing

### Batch Processing
- Modify batch sizes in Phase 1 configuration
- Use smaller batches for limited memory systems

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [F5-TTS](https://github.com/SWivid/F5-TTS) - Text-to-Speech synthesis
- [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443) - Reference speech data
- [MUSAN](https://www.openslr.org/17/) - Background noise dataset

## 🙋‍♂️ Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation