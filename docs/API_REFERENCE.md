# API Reference

This document provides detailed API reference for the TV Command Synthesis Pipeline.

## Pipeline Executor

### `PipelineExecutor`

Main orchestrator for the synthesis pipeline.

```python
from src.pipeline.pipeline_executor import PipelineExecutor

pipeline = PipelineExecutor(config_path="configs/pipeline_config.yaml")
```

#### Constructor

```python
PipelineExecutor(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path`: Path to YAML configuration file

#### Methods

##### `run_full_pipeline()`

```python
run_full_pipeline(
    phases: List[str] = None,
    custom_mode_phase4: bool = False
) -> Dict[str, Any]
```

Execute the complete pipeline or specified phases.

**Parameters:**
- `phases`: List of phases to run (default: all phases)
- `custom_mode_phase4`: Enable custom mode for Phase 4

**Returns:**
- Dictionary with execution results and timing

##### `run_phase1()`

```python
run_phase1(output_dir: str = None) -> Dict[str, Any]
```

Execute Phase 1: Text synthesis.

##### `run_phase2()`

```python
run_phase2(
    input_dir: str = None,
    output_dir: str = None
) -> Dict[str, Any]
```

Execute Phase 2: Speech synthesis.

##### `run_phase3()`

```python
run_phase3(
    input_dir: str = None,
    output_dir: str = None
) -> Dict[str, Any]
```

Execute Phase 3: Audio concatenation.

##### `run_phase4()`

```python
run_phase4(
    input_dir: str = None,
    output_dir: str = None,
    custom_mode: bool = False
) -> Dict[str, Any]
```

Execute Phase 4: Noise augmentation.

##### `validate_setup()`

```python
validate_setup() -> Dict[str, Any]
```

Validate pipeline setup and data availability.

**Returns:**
- Dictionary with validation results and issues

---

## Phase 1: Text Synthesis

### `Phase1TextSynthesis`

```python
from src.phases.phase1_text_synthesis.main import Phase1TextSynthesis
```

#### `DataPreparation`

```python
from src.phases.phase1_text_synthesis.data_preparation import DataPreparation

data_prep = DataPreparation(content_loader, command_loader)
```

##### Methods

```python
process(input_data=None) -> Tuple[Dict[str, List[str]], List[str]]
```

**Returns:**
- Tuple of (content_dict, commands_list)

#### `PromptGenerator`

```python
from src.phases.phase1_text_synthesis.prompt_generator import PromptGenerator

prompt_gen = PromptGenerator(config)
```

##### Methods

```python
generate_prompt(
    prompt_type: str,
    commands: List[str],
    content_dict: Dict[str, List[Dict]]
) -> str
```

**Parameters:**
- `prompt_type`: One of 'non_active', 'single_active', 'single_mix', 'chain_active', 'chain_mix'
- `commands`: List of command strings
- `content_dict`: Dictionary of content by type

#### `TextGenerator`

```python
from src.phases.phase1_text_synthesis.text_generator import TextGenerator

text_gen = TextGenerator(config)
```

##### Methods

```python
generate(prompt: str) -> str
```

Generate text using LLM API.

```python
create_dataframe(parsed_data: Dict) -> pd.DataFrame
```

Create DataFrame from parsed generation results.

---

## Phase 2: Speech Synthesis

### `Phase2SpeechSynthesis`

```python
from src.phases.phase2_speech_synthesis.main import Phase2SpeechSynthesis
```

#### `ReferenceCache`

```python
from src.phases.phase2_speech_synthesis.reference_cache import ReferenceCache

ref_cache = ReferenceCache(audio_folder_path, json_folder_path)
```

##### Methods

```python
sample_reference(target_len: int) -> Tuple[str, str, str]
```

**Parameters:**
- `target_len`: Target text length in words

**Returns:**
- Tuple of (ref_id, audio_file_path, ref_text)

#### `SpeechSynthesis`

```python
from src.phases.phase2_speech_synthesis.speech_synthesis import SpeechSynthesis

speech_synth = SpeechSynthesis(config)
```

##### Methods

```python
load_model()
```

Load F5-TTS model.

```python
audio_generate(
    command: str,
    model: Any,
    ref_file: str = "",
    ref_text: str = ""
)
```

Generate audio using F5-TTS.

---

## Phase 3: Audio Concatenation

### `Phase3Concatenation`

```python
from src.phases.phase3_concatenation.main import Phase3Concatenation
```

#### `AudioConcatenator`

```python
from src.phases.phase3_concatenation.audio_concatenator import AudioConcatenator

concatenator = AudioConcatenator(config)
```

##### Methods

```python
process(input_data) -> List[Dict]
```

**Parameters:**
- `input_data`: Tuple of (metadata, input_folder, output_folder, phase2_data)

**Returns:**
- List of processing results

---

## Phase 4: Noise Augmentation

### `Phase4NoiseAugmentation`

```python
from src.phases.phase4_noise_augmentation.main import Phase4NoiseAugmentation
```

#### `NoiseAugmentor`

```python
from src.phases.phase4_noise_augmentation.noise_augmentor import NoiseAugmentor

augmentor = NoiseAugmentor(config)
```

##### Methods

```python
augment(
    clean_path: str,
    labels_path: str,
    mode: str = None,
    custom_snr: float = None,
    custom_noise: torch.Tensor = None,
    custom_noise_path: str = None,
    phase2_data: Optional[Dict] = None
) -> Tuple[torch.Tensor, Dict]
```

**Parameters:**
- `clean_path`: Path to clean audio file
- `labels_path`: Path to labels JSON file
- `mode`: Augmentation mode ('overlap', 'prepend', 'append')
- `custom_snr`: Custom SNR value
- `custom_noise`: Custom noise tensor
- `custom_noise_path`: Path to custom noise file
- `phase2_data`: Phase 2 data for text label extraction

**Returns:**
- Tuple of (augmented_audio, metadata)

---

## Utilities

### Text Utils

```python
from src.utils.text_utils import get_active_text_labels, extract_file_id_from_path
```

#### `get_active_text_labels()`

```python
get_active_text_labels(
    phase2_data: Union[pd.DataFrame, Dict],
    file_id: str
) -> str
```

Extract and concatenate active text from Phase 2 data.

**Parameters:**
- `phase2_data`: DataFrame or dict from Phase 2
- `file_id`: ID of the file to process

**Returns:**
- Concatenated active text segments

#### `extract_file_id_from_path()`

```python
extract_file_id_from_path(
    file_name: str,
    separator: str = '_'
) -> str
```

Extract file ID from file name pattern.

---

## Data Loaders

### JSONDataLoader

```python
from src.data.loaders.json_loader import JSONDataLoader

loader = JSONDataLoader()
```

#### Methods

```python
load(source: Union[str, Path]) -> pd.DataFrame
```

Load JSON file and convert to DataFrame.

```python
load_multiple_batches(
    directory: Union[str, Path],
    pattern: str = "*.json"
) -> pd.DataFrame
```

Load and concatenate multiple JSON batch files.

```python
save_dataframe_as_json(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    metadata: Optional[Dict] = None
)
```

Save DataFrame as JSON.

### ContentLoader

```python
from src.data.loaders.content_loader import ContentLoader

content_loader = ContentLoader(content_dir)
```

#### Methods

```python
load_all_content() -> Dict[str, List[Dict]]
```

Load all content files and return as dictionary.

```python
get_content_by_type(content_type: str) -> List[str]
```

Get content list for specific type.

### CommandLoader

```python
from src.data.loaders.command_loader import CommandLoader

command_loader = CommandLoader(commands_dir)
```

#### Methods

```python
get_commands_list() -> List[str]
```

Get list of all commands.

---

## Configuration

### Config

```python
from src.core.config import Config

config = Config(config_path)
```

#### Methods

```python
get(key: str, default: Any = None) -> Any
```

Get configuration value by key path.

```python
set(key: str, value: Any)
```

Set configuration value by key path.

```python
save_config(config_path: Union[str, Path])
```

Save current configuration to file.

---

## Base Classes

### BaseProcessor

```python
from src.core.base_data_handler import BaseProcessor
```

Abstract base class for data processing operations.

#### Abstract Methods

```python
process(input_data: Any) -> Any
```

Process input data and return output.

```python
get_config() -> Dict[str, Any]
```

Return processor configuration.

### BasePhase

```python
from src.core.base_data_handler import BasePhase
```

Abstract base class for pipeline phases.

#### Abstract Methods

```python
setup()
```

Setup phase-specific components.

```python
run(input_data: Any) -> Any
```

Execute the phase.

```python
cleanup()
```

Cleanup phase resources.

---

## Example Usage

### Complete Pipeline

```python
from src.pipeline.pipeline_executor import PipelineExecutor

# Initialize with configuration
pipeline = PipelineExecutor('configs/pipeline_config.yaml')

# Validate setup
validation = pipeline.validate_setup()
if not validation['valid']:
    print("Setup issues:", validation['issues'])

# Run complete pipeline
results = pipeline.run_full_pipeline()

# Check results
if results['success']:
    print(f"Pipeline completed in {results['total_execution_time']:.2f}s")
else:
    print("Pipeline failed:", results['errors'])
```

### Individual Phase

```python
from src.phases.phase1_text_synthesis.main import Phase1TextSynthesis
from src.core.config import Config

# Load configuration
config = Config('configs/pipeline_config.yaml')

# Initialize phase
phase1 = Phase1TextSynthesis(config.config)

# Run phase
results = phase1.run('data/processed/phase1')
print(f"Generated {len(results)} text samples")
```

### Custom Processing

```python
from src.data.loaders.json_loader import JSONDataLoader
from src.utils.text_utils import get_active_text_labels

# Load data
loader = JSONDataLoader()
phase2_data = loader.load('data/processed/phase2/speech_synthesis_results.json')

# Extract active text for specific file
active_text = get_active_text_labels(phase2_data, 'abc123def')
print(f"Active commands: {active_text}")
```