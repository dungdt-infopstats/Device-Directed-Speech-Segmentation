"""
Main pipeline executor for TV command synthesis.
Orchestrates all phases with proper data flow and error handling.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..core.config import Config
from ..phases.phase1_text_synthesis.main import Phase1TextSynthesis
from ..phases.phase2_speech_synthesis.main import Phase2SpeechSynthesis
from ..phases.phase3_force_alignment.main import Phase3SpeechCleaning
from ..phases.phase4_noise_augmentation.main import Phase4NoiseAugmentation


class PipelineExecutor:
    """Main pipeline executor for the TV command synthesis system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration."""
        self.config = Config(config_path)
        self.logger = self._setup_logging()
        self.phases = {
            'phase1': None,
            'phase2': None,
            'phase3': None,
            'phase4': None
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger('tv_synthesis_pipeline')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def initialize_phases(self, phases: List[str] = None):
        """Initialize specified phases."""
        if phases is None:
            phases = ['phase1', 'phase2', 'phase3', 'phase4']

        self.logger.info(f"Initializing phases: {phases}")

        if 'phase1' in phases:
            self.phases['phase1'] = Phase1TextSynthesis(self.config.config)

        if 'phase2' in phases:
            self.phases['phase2'] = Phase2SpeechSynthesis(self.config.config)

        if 'phase3' in phases:
            self.phases['phase3'] = Phase3SpeechCleaning(self.config.config)

        if 'phase4' in phases:
            self.phases['phase4'] = Phase4NoiseAugmentation(self.config.config)

    def run_phase1(self, output_dir: str = None) -> Dict[str, Any]:
        """Execute Phase 1: Text synthesis."""
        self.logger.info("Starting Phase 1: Text Synthesis")
        start_time = time.time()

        try:
            if self.phases['phase1'] is None:
                self.initialize_phases(['phase1'])

            result_df = self.phases['phase1'].run(output_dir)

            execution_time = time.time() - start_time
            result = {
                'success': True,
                'phase': 'phase1',
                'execution_time': execution_time,
                'output_records': len(result_df),
                'output_dir': output_dir or self.config.get('data.output_dir') + '/phase1'
            }

            self.logger.info(f"Phase 1 completed in {execution_time:.2f}s - {len(result_df)} records generated")
            return result

        except Exception as e:
            self.logger.error(f"Phase 1 failed: {str(e)}")
            return {
                'success': False,
                'phase': 'phase1',
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def run_phase2(self, input_dir: str = None, output_dir: str = None) -> Dict[str, Any]:
        """Execute Phase 2: Speech synthesis."""
        self.logger.info("Starting Phase 2: Speech Synthesis")
        start_time = time.time()

        try:
            if self.phases['phase2'] is None:
                self.initialize_phases(['phase2'])

            result_df = self.phases['phase2'].run(
                input_dir=input_dir,
                output_dir=output_dir
            )

            execution_time = time.time() - start_time
            result = {
                'success': True,
                'phase': 'phase2',
                'execution_time': execution_time,
                'output_records': len(result_df),
                'output_dir': output_dir or self.config.get('data.output_dir') + '/phase2'
            }

            self.logger.info(f"Phase 2 completed in {execution_time:.2f}s - {len(result_df)} records processed")
            return result

        except Exception as e:
            self.logger.error(f"Phase 2 failed: {str(e)}")
            return {
                'success': False,
                'phase': 'phase2',
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def run_phase3(self, input_dir: str = None, output_dir: str = None,
                   enable_force_alignment: bool = True) -> Dict[str, Any]:
        """Execute Phase 3: Speech cleaning (force alignment + concatenation)."""
        self.logger.info("Starting Phase 3: Speech Cleaning")
        start_time = time.time()

        try:
            if self.phases['phase3'] is None:
                self.initialize_phases(['phase3'])

            results = self.phases['phase3'].run(
                input_dir=input_dir,
                output_dir=output_dir,
                enable_force_alignment=enable_force_alignment
            )

            execution_time = time.time() - start_time
            concat_results = results.get('concatenation', [])
            alignment_results = results.get('force_alignment', {})

            result = {
                'success': True,
                'phase': 'phase3',
                'execution_time': execution_time,
                'output_records': len(concat_results) if concat_results else 0,
                'output_dir': output_dir or self.config.get('data.output_dir') + '/phase3',
                'force_alignment_enabled': enable_force_alignment,
                'alignment_stats': alignment_results
            }

            self.logger.info(f"Phase 3 completed in {execution_time:.2f}s - {len(concat_results) if concat_results else 0} files processed")
            return result

        except Exception as e:
            self.logger.error(f"Phase 3 failed: {str(e)}")
            return {
                'success': False,
                'phase': 'phase3',
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def run_phase4(self, input_dir: str = None, output_dir: str = None,
                   custom_mode: bool = False) -> Dict[str, Any]:
        """Execute Phase 4: Noise augmentation."""
        self.logger.info("Starting Phase 4: Noise Augmentation")
        start_time = time.time()

        try:
            if self.phases['phase4'] is None:
                self.initialize_phases(['phase4'])

            results = self.phases['phase4'].run(
                input_dir=input_dir,
                output_dir=output_dir,
                custom_mode=custom_mode
            )

            execution_time = time.time() - start_time
            result = {
                'success': True,
                'phase': 'phase4',
                'execution_time': execution_time,
                'output_records': len(results),
                'output_dir': output_dir or self.config.get('data.output_dir') + '/phase4',
                'custom_mode': custom_mode
            }

            self.logger.info(f"Phase 4 completed in {execution_time:.2f}s - {len(results)} files processed")
            return result

        except Exception as e:
            self.logger.error(f"Phase 4 failed: {str(e)}")
            return {
                'success': False,
                'phase': 'phase4',
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def run_full_pipeline(self, phases: List[str] = None,
                         custom_mode_phase4: bool = False) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        if phases is None:
            phases = ['phase1', 'phase2', 'phase3', 'phase4']

        self.logger.info(f"Starting full pipeline with phases: {phases}")
        pipeline_start = time.time()

        results = {
            'pipeline_start_time': pipeline_start,
            'phases': {},
            'total_execution_time': 0,
            'success': True,
            'errors': []
        }

        # Initialize all required phases
        self.initialize_phases(phases)

        # Execute phases in order
        phase_methods = {
            'phase1': self.run_phase1,
            'phase2': self.run_phase2,
            'phase3': self.run_phase3,
            'phase4': lambda: self.run_phase4(custom_mode=custom_mode_phase4)
        }

        for phase in phases:
            if phase in phase_methods:
                self.logger.info(f"Executing {phase}")
                phase_result = phase_methods[phase]()
                results['phases'][phase] = phase_result

                if not phase_result['success']:
                    results['success'] = False
                    results['errors'].append(f"{phase}: {phase_result.get('error', 'Unknown error')}")
                    self.logger.error(f"Pipeline stopped at {phase} due to error")
                    break

        # Cleanup phases
        for phase_obj in self.phases.values():
            if phase_obj:
                phase_obj.cleanup()

        results['total_execution_time'] = time.time() - pipeline_start

        if results['success']:
            self.logger.info(f"Pipeline completed successfully in {results['total_execution_time']:.2f}s")
        else:
            self.logger.error(f"Pipeline failed after {results['total_execution_time']:.2f}s")

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration."""
        return {
            'config': self.config.config,
            'initialized_phases': [k for k, v in self.phases.items() if v is not None],
            'data_directories': {
                'content_dir': self.config.get('data.content_dir'),
                'commands_dir': self.config.get('data.commands_dir'),
                'output_dir': self.config.get('data.output_dir')
            }
        }

    def validate_setup(self) -> Dict[str, Any]:
        """Validate pipeline setup and data availability."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        # Check data directories
        content_dir = Path(self.config.get('data.content_dir'))
        commands_dir = Path(self.config.get('data.commands_dir'))

        if not content_dir.exists():
            validation_results['issues'].append(f"Content directory not found: {content_dir}")
            validation_results['valid'] = False

        if not commands_dir.exists():
            validation_results['issues'].append(f"Commands directory not found: {commands_dir}")
            validation_results['valid'] = False

        # Check for content files
        expected_content_files = ['app.json', 'movie.json', 'song.json', 'tv.json']
        for content_file in expected_content_files:
            if not (content_dir / content_file).exists():
                validation_results['warnings'].append(f"Content file not found: {content_file}")

        # Check commands file
        if not (commands_dir / 'commands.json').exists():
            validation_results['issues'].append("Commands file not found: commands.json")
            validation_results['valid'] = False

        # Check external dependencies for Phase 2
        ref_folder = self.config.get('phase2.ref_folder_path')
        if ref_folder and not Path(ref_folder).exists():
            validation_results['warnings'].append(f"Reference audio folder not found: {ref_folder}")

        # Check noise folders for Phase 4
        noise_folders = self.config.get('phase4.noise_folders', [])
        for noise_folder in noise_folders:
            if not Path(noise_folder).exists():
                validation_results['warnings'].append(f"Noise folder not found: {noise_folder}")

        return validation_results