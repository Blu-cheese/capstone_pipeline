"""
DHGAT extractor adapter.

Wraps the Blu-cheese/capstone_dialouge-re model to produce List[Triple]
in the same format as the LLM extractor.

This is the BASELINE extractor for your paper. It:
1. Runs spaCy NER on each utterance to find entity mentions
2. Generates candidate entity pairs
3. Feeds them through the trained DHGAT model for relation classification
4. Outputs Triple objects

SETUP:
    1. Clone your fork: git clone https://github.com/Blu-cheese/capstone_dialouge-re
    2. Install deps: pip install -r requirements-colab.txt
    3. Train the model: python main.py --mode train
    4. Point this adapter at the checkpoint: DHGATExtractor(ckpt_path="runs/.../best.pt")

NOTE: The DHGAT model is trained on DialogRE's 36 relation types.
These are Friends TV show relations (per:girl/boyfriend, per:schools_attended, etc).
They won't match your meeting domain — that's expected and is part of your paper's
domain-gap analysis (Experiment A).
"""

import json
from typing import List, Optional, Dict
from pathlib import Path

from utils.models import ConversationWindow, Triple


# DialogRE relation labels (the 36 types the model was trained on)
DIALOGRE_RELATIONS = [
    "per:positive_impression", "per:negative_impression", "per:acquaintance",
    "per:alumni", "per:boss", "per:subordinate", "per:client", "per:dates",
    "per:friends", "per:girl/boyfriend", "per:neighbor", "per:roommate",
    "per:children", "per:other_family", "per:parents", "per:siblings",
    "per:spouse", "per:place_of_residence", "per:place_of_birth",
    "per:visited_place", "per:origin", "per:employee_or_member_of",
    "per:schools_attended", "per:works", "per:age", "per:date_of_birth",
    "per:major", "per:place_of_work", "per:title", "per:alternate_names",
    "per:pet", "per:visited", "gpe:births_in_place", "gpe:residents_of_place",
    "org:students", "unanswerable",
]


class DHGATExtractor:
    """
    Wrapper around the DHGAT model for the shared pipeline interface.
    
    This requires the capstone_dialouge-re repo to be installed and 
    a trained checkpoint to be available.
    """

    def __init__(
        self,
        repo_path: str = "./capstone_dialouge-re",
        ckpt_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.3,
    ):
        self.repo_path = Path(repo_path)
        self.ckpt_path = ckpt_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.nlp = None

        # Resolve device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def load(self):
        """
        Load the DHGAT model and spaCy NER.
        Call this once before extraction.
        """
        # Load spaCy for NER
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("[DHGAT] spaCy NER loaded")
        except Exception as e:
            print(f"[DHGAT] spaCy load failed: {e}")
            print("  Install: pip install spacy && python -m spacy download en_core_web_sm")
            return False

        # Load DHGAT model
        try:
            import sys
            sys.path.insert(0, str(self.repo_path))
            from model.trainer import Train_GraphDialogRe, load_checkpoint
            from utils.config import config as dhgat_config

            dhgat_config.device = self.device
            self.model = Train_GraphDialogRe(dhgat_config).to(self.device)

            if self.ckpt_path:
                checkpoint, missing, unexpected = load_checkpoint(
                    self.ckpt_path, self.model, map_location=self.device
                )
                print(f"[DHGAT] Model loaded from {self.ckpt_path}")
                if missing:
                    print(f"  Warning: missing keys: {missing}")
            else:
                print("[DHGAT] No checkpoint specified — model has random weights")
                print("  Train first: python main.py --mode train")

            self.model.eval()
            return True
        except Exception as e:
            print(f"[DHGAT] Model load failed: {e}")
            print(f"  Make sure {self.repo_path} exists and deps are installed")
            return False

    def _extract_entities(self, text: str) -> List[Dict]:
        """Run spaCy NER on text, return entity spans with types."""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            # Map spaCy types to our taxonomy
            type_map = {
                "PERSON": "PERSON",
                "ORG": "DEPARTMENT",
                "DATE": "DEADLINE",
                "TIME": "DEADLINE",
                "GPE": "RESOURCE",
                "FAC": "RESOURCE",
                "EVENT": "EVENT",
                "WORK_OF_ART": "COURSE",
            }
            mapped_type = type_map.get(ent.label_, "TOPIC")
            entities.append({
                "text": ent.text,
                "type": mapped_type,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return entities

    def extract(self, window: ConversationWindow) -> List[Triple]:
        """
        Extract triples from a conversation window.
        
        Current implementation: spaCy NER + heuristic relation assignment.
        
        TODO for your team (person 4):
            Replace the heuristic relation assignment with actual DHGAT inference.
            This requires formatting the window's text + entity pairs into the 
            format DHGAT expects (see dataset/train.json for the expected schema)
            and running model forward pass.
        """
        all_entities = []
        for utt in window.utterances:
            ents = self._extract_entities(utt.text)
            for e in ents:
                e["speaker"] = utt.speaker_id
                e["timestamp"] = utt.start_time
            all_entities.extend(ents)

        if len(all_entities) < 2:
            return []

        # Generate entity pairs and classify
        # NOTE: This is the heuristic placeholder.
        # Person 4 should replace this with actual DHGAT model inference.
        triples = []
        seen_pairs = set()

        for i, e1 in enumerate(all_entities):
            for j, e2 in enumerate(all_entities):
                if i == j:
                    continue
                pair_key = (e1["text"].lower(), e2["text"].lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Heuristic: assign relation based on entity types
                relation = self._heuristic_relation(e1, e2)
                if relation:
                    triples.append(Triple(
                        subject=e1["text"],
                        subject_type=e1["type"],
                        relation=relation,
                        object=e2["text"],
                        object_type=e2["type"],
                        confidence=0.5,  # Low confidence for heuristic
                        source_meeting=window.meeting_id,
                        timestamp=e1.get("timestamp", 0.0),
                    ))

        return triples

    def _heuristic_relation(self, e1: Dict, e2: Dict) -> Optional[str]:
        """
        Placeholder heuristic relation assignment.
        Replace with DHGAT model inference.
        """
        t1, t2 = e1["type"], e2["type"]

        type_pair_relations = {
            ("PERSON", "DEPARTMENT"): "member_of",
            ("PERSON", "COURSE"): "teaches",
            ("PERSON", "COMMITTEE"): "member_of",
            ("PERSON", "PERSON"): "discussed_with",
            ("DEPARTMENT", "COURSE"): "offers",
            ("PERSON", "EVENT"): "participates_in",
            ("PERSON", "PROJECT"): "assigned_to",
        }

        return type_pair_relations.get((t1, t2))

    def extract_meeting(self, windows: List[ConversationWindow]) -> List[Triple]:
        all_triples = []
        for i, window in enumerate(windows):
            print(f"  [DHGAT] Extracting window {i+1}/{len(windows)}...")
            triples = self.extract(window)
            all_triples.extend(triples)
            print(f"    Found {len(triples)} triples")
        return all_triples
