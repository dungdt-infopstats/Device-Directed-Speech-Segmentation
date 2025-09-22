"""
Prompt generator for Phase 1 text synthesis.
Refactored version with configuration management.
"""

import random
from typing import Dict, List
from ...core.base_data_handler import BaseProcessor


class PromptGenerator(BaseProcessor):
    """Generate prompts for text synthesis."""

    def __init__(self, config: Dict):
        self.config = config
        self.prompt_templates = self._load_prompt_templates()

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        return {
            "non_active": """
You are to generate {generated_nums} sentences of type "non_active".

**Definition of "non_active":**
- Sentences represent natural, everyday indoor human conversation.
- Absolutely no TV/device commands.
- Mentioning or talking about a command is still considered non_active — only direct device instructions count as active.

**Input content:**
- Use the following content list (remove prefix like movie, song, ... when use): {content_list}

**Rules:**
1. Use only human-to-human conversation, not device commands.
2. Even if the sentence contains TV-related words, if it is not a direct command to the device, it is still non_active.
3. Sentences must sound natural and conversational.
4. Use the specific content ideas from the list.

**Output format (JSON only):**
```json
{{
  "non_active": [
    {{ "text": "...", }}
  ]
}}
```
""",
            "single_active": """
You are to generate {generated_nums} sentences of type "single_active".

**Definition of "single_active":**
- Each sentence is a direct, clear TV/device command.
- No human conversation or extra wording beyond the command.
- The command must be something that could be spoken to a device to perform an action immediately.

**Input:**
- TV commands list: {command_list}
- Content list (remove prefix like movie, song, ... when use): {content_list}

**Rules:**
1. Each sentence = exactly one direct TV/device command from the TV commands list **combined with one item from the content list** to form a complete, natural device instruction.
2. Do not use questions, hypotheticals, or descriptions — only imperative commands.
3. No unrelated conversation or comments.
4. Must sound natural as a spoken device instruction.

**Output format (JSON only):**
```json
{{
  "single_active": [
    {{ "text": "..." }}
  ]
}}
```
""",
            "single_mix": """
You are to generate {generated_nums} sentences of type "single_mix".

**Definition of "single_mix":**
- Each sentence must contain exactly ONE direct, clear TV/device command (from the TV commands list combined with one item from the content list) + one unrelated human conversation.
- The command must be an imperative statement addressed to the device, not a question or discussion.
- The two parts must be completely unrelated.

**Position requirement:**
- In this task, the TV/device command should appear {command_position} in the sentence.

**Input:**
- TV commands list: {command_list}
- Content list (remove prefix like movie, song, ... when use): {content_list}

**Rules:**
1. Exactly one direct TV/device command (command + content) + one unrelated human conversation.
2. The command must be in imperative form (telling the device to do something immediately).
3. No hypotheticals, descriptions, or indirect mentions of commands.
4. Commands and conversation must follow the specified position rule above.
5. Sentences must sound natural.
6. When splitting into segments, include every single word from the original sentence — no removals. The order of segments must follow the order of text command and conversation.
7. In segments, `"type"` must be `"active"` for commands and `"non_active"` for conversation. A command + content is still a `"active"`.

**Output format (JSON only):**
```json
{{
  "single_mix": [
    {{
      "text": "...",
      "segments": [
        {{ "0": "...", "type": "active|non_active" }},
        {{ "1": "...", "type": "non_active|active" }}
      ]
    }}
  ]
}}
```
""",
            "chain_active": """
You are to generate {generated_nums} sentences of type "chain_active".

**Definition of "chain_active":**
- Sentences contain multiple direct TV/device commands only.
- All commands must be imperative instructions to the device.
- Each command must be formed by combining one item from the TV commands list with one item from the content list.

**Input:**
- TV commands list: {command_list}
- Content list (remove prefix like movie, song, ... when use): {content_list}

**Rules:**
1. Each sentence must have two or more imperative device commands (each command = command + content).
2. No human conversation, questions, or descriptive phrases.
3. Commands can be related or unrelated, but all must be valid direct instructions.
4. Sentences should sound natural as a spoken command sequence.

**Output format (JSON only):**
```json
{{
  "chain_active": [
    {{
      "text": "..."
    }}
  ]
}}
```
""",
            "chain_mix": """
You are to generate {generated_nums} sentences of type "chain_mix".

**Definition of "chain_mix":**
- Sentences must contain multiple direct TV/device commands (each command from the TV commands list combined with one item from the content list) + at least one unrelated human conversation.
- Commands must be imperative instructions to the device, not questions or descriptions.

**Position requirement:**
- In this task, the sequence of TV/device commands should appear {command_position} in the sentence.

**Input:**
- TV commands list: {command_list}
- Content list (remove prefix like movie, song, ... when use): {content_list}

**Rules:**
1. Each sentence must contain 2 or more direct imperative commands (command + content) from the provided lists.
2. Must also contain at least one unrelated human conversation segment.
3. Commands and conversation must follow the specified position rule above.
4. No hypotheticals, indirect mentions, or descriptions — only direct instructions are active.
5. Sentences must sound natural.
6. Segments must cover the entire sentence with no missing words or characters. The order of segments must follow the order of commands and conversation in the text.
7. In segments, `"type"` must be `"active"` for commands and `"non_active"` for conversation. A command + content is still a `"active"`.

**Output format (JSON only):**
```json
{{
  "chain_mix": [
    {{
      "text": "...",
      "segments": [
        {{ "0": "...", "type": "active|non_active" }},
        {{ "1": "...", "type": "non_active|active" }}
      ]
    }}
  ]
}}
```
"""
        }

    def content_sample(self, content_dict: Dict[str, List[Dict]]) -> List[List[str]]:
        """Sample content from each data corpus."""
        sample_list = []
        num_samples = self.config.get('num_samples_content', 5)

        for content_type, items in content_dict.items():
            content_items = [item['content'] for item in items]
            if len(content_items) >= num_samples:
                sample_list.extend(random.sample(content_items, num_samples))
            else:
                sample_list.extend(content_items)

        return sample_list

    def command_sample(self, commands: List[str]) -> List[str]:
        """Sample commands."""
        num_samples = self.config.get('num_samples_command', 5)
        if len(commands) >= num_samples:
            return random.sample(commands, num_samples)
        return commands

    def position_sample(self) -> str:
        """Sample command positions for mix types."""
        positions = [
            "at the beginning",
            "at the end",
            "in the middle",
            "scattered throughout",
            "before the conversation part",
            "after the conversation part"
        ]
        return random.choice(positions)

    def generate_prompt(self, prompt_type: str, commands: List[str],
                       content_dict: Dict[str, List[Dict]]) -> str:
        """Generate prompt for specific type."""
        sampled_commands = self.command_sample(commands)
        sampled_contents = self.content_sample(content_dict)
        generated_nums = self.config.get('generated_nums', 4)

        template = self.prompt_templates[prompt_type]

        if prompt_type in ("single_mix", "chain_mix"):
            command_position = self.position_sample()
            return template.format(
                command_list=sampled_commands,
                content_list=sampled_contents,
                generated_nums=generated_nums,
                command_position=command_position
            )
        else:
            return template.format(
                command_list=sampled_commands,
                content_list=sampled_contents,
                generated_nums=generated_nums
            )

    def process(self, input_data) -> Dict[str, str]:
        """Generate prompts for all types."""
        commands, content_dict = input_data
        prompt_types = self.config.get('prompt_types', [
            'non_active', 'single_active', 'single_mix', 'chain_active', 'chain_mix'
        ])

        prompts = {}
        for prompt_type in prompt_types:
            prompts[prompt_type] = self.generate_prompt(prompt_type, commands, content_dict)

        return prompts

    def get_config(self) -> Dict:
        """Return processor configuration."""
        return self.config