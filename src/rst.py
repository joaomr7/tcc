from pathlib import Path
from typing import Dict, List, Union

import torch
from transformers import AutoModel, AutoTokenizer

from pathlib import Path
import sys

# Add DMRST Parser to the Python import path
PROJECT_ROOT = Path.cwd().resolve().parent
DMRST_PATH = PROJECT_ROOT / 'external' / 'DMRST_Parser'
sys.path.insert(0, str(DMRST_PATH))

from model_depth import ParsingNet
from MUL_main_Infer import inference

class DMRSTParser:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        cache_dir: Union[str, Path],
        batch_size: int = 1,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                'This DMRST_Parser version requires GPU with CUDA support.'
            )

        self.batch_size = batch_size
        self.checkpoint_path = Path(checkpoint_path)
        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f'Checkpoint not find: '
                f'{self.checkpoint_path.resolve()}'
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            'xlm-roberta-base',
            use_fast=True,
            cache_dir=str(self.cache_dir),
        )

        language_model = AutoModel.from_pretrained(
            'xlm-roberta-base',
            cache_dir=str(self.cache_dir),
        ).cuda()

        for parameter in language_model.parameters():
            parameter.requires_grad = False

        self.model = ParsingNet(
            language_model,
            bert_tokenizer=self.tokenizer,
        ).cuda()

        state_dict = torch.load(
            str(self.checkpoint_path),
            map_location="cuda",
        )

        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        obsolete_keys = [
            key
            for key in state_dict
            if key.endswith('.embeddings.position_ids')
        ]

        for key in obsolete_keys:
            print(f'Ignoring old checkpoint key: {key}')
            state_dict.pop(key)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def parse(
        self,
        texts: Union[str, List[str]],
    ) -> Union[Dict, List[Dict]]:
        received_single_text = isinstance(texts, str)

        if received_single_text:
            documents = [texts]
        else:
            documents = list(texts)

        documents = [document.strip() for document in documents]

        if not documents or any(not document for document in documents):
            raise ValueError('Text can\'t be empty')

        tokens_batch, breaks_batch, trees_batch = inference(
            model=self.model,
            tokenizer=self.tokenizer,
            input_sentences=documents,
            batch_size=self.batch_size,
        )

        results = []

        for original_text, tokens, edu_breaks, tree in zip(
            documents,
            tokens_batch,
            breaks_batch,
            trees_batch,
        ):
            edus = self._reconstruct_edus(tokens, edu_breaks)

            tree_value = (
                tree[0]
                if isinstance(tree, list) and len(tree) == 1
                else tree
            )

            results.append(
                {
                    'text': original_text,
                    'tokens': tokens,
                    'edu_breaks': edu_breaks,
                    'edus': edus,
                    'tree': tree_value,
                }
            )

        return results[0] if received_single_text else results

    def _reconstruct_edus(
        self,
        tokens: List[str],
        edu_breaks: List[int],
    ) -> List[str]:
        edus = []
        start = 0

        for end in edu_breaks:
            edu_tokens = tokens[start : end + 1]

            edu_text = self.tokenizer.convert_tokens_to_string(
                edu_tokens
            ).strip()

            edus.append(edu_text)
            start = end + 1

        return edus

