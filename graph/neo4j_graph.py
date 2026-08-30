"""
Neo4j knowledge graph integration.

Inserts resolved triples into Neo4j and provides query utilities.
Uses MERGE to prevent duplicate nodes.

SETUP:
    1. Install Neo4j Community Edition: https://neo4j.com/download/
    2. Start it: neo4j console
    3. Default credentials: neo4j/neo4j (you'll be prompted to change password)
    4. pip install neo4j

Alternatively, use Neo4j Desktop or Docker:
    docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
"""

import json
from typing import List, Dict, Optional
from datetime import datetime

from utils.models import Triple


class KnowledgeGraph:
    """
    Neo4j-backed knowledge graph.
    Stores entities as nodes and relations as edges,
    both with temporal metadata.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self) -> bool:
        """Connect to Neo4j. Returns True on success."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"[Neo4j] Connected to {self.uri}")
            return True
        except Exception as e:
            print(f"[Neo4j] Connection failed: {e}")
            print("  Make sure Neo4j is running and credentials are correct.")
            return False

    def close(self):
        if self.driver:
            self.driver.close()

    def setup_constraints(self):
        """Create uniqueness constraints for entity names."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    # Community edition may not support all constraint types
                    print(f"  [WARN] Constraint skipped: {e}")
        print("[Neo4j] Constraints set up")

    def insert_triples(self, triples: List[Triple], meeting_id: str = ""):
        """
        Insert a batch of triples into the graph.
        Uses MERGE to avoid duplicates.
        """
        if not self.driver:
            print("[Neo4j] Not connected")
            return

        timestamp = datetime.now().isoformat()
        inserted = 0

        with self.driver.session() as session:
            for t in triples:
                try:
                    session.run(
                        """
                        MERGE (s:Entity {name: $subject})
                        ON CREATE SET 
                            s.type = $subject_type,
                            s.first_seen = $timestamp,
                            s.first_meeting = $meeting_id
                        SET s.last_seen = $timestamp,
                            s.last_meeting = $meeting_id

                        MERGE (o:Entity {name: $object})
                        ON CREATE SET 
                            o.type = $object_type,
                            o.first_seen = $timestamp,
                            o.first_meeting = $meeting_id
                        SET o.last_seen = $timestamp,
                            o.last_meeting = $meeting_id

                        CREATE (s)-[r:RELATION {
                            type: $relation,
                            confidence: $confidence,
                            source_meeting: $meeting_id,
                            timestamp: $timestamp,
                            utterance_time: $utterance_time
                        }]->(o)
                        """,
                        subject=t.subject,
                        subject_type=t.subject_type,
                        object=t.object,
                        object_type=t.object_type,
                        relation=t.relation,
                        confidence=t.confidence,
                        meeting_id=meeting_id or t.source_meeting,
                        timestamp=timestamp,
                        utterance_time=t.timestamp,
                    )
                    inserted += 1
                except Exception as e:
                    print(f"  [WARN] Failed to insert triple: {t.subject} -[{t.relation}]-> {t.object}: {e}")

        print(f"[Neo4j] Inserted {inserted}/{len(triples)} triples from meeting '{meeting_id}'")

    def get_graph_stats(self) -> Dict:
        """Return counts of nodes and relationships."""
        if not self.driver:
            return {}

        with self.driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
            types = session.run(
                "MATCH (n:Entity) RETURN n.type AS type, count(n) AS c ORDER BY c DESC"
            ).data()
            rel_types = session.run(
                "MATCH ()-[r:RELATION]->() RETURN r.type AS type, count(r) AS c ORDER BY c DESC"
            ).data()

        return {
            "total_entities": nodes,
            "total_relations": rels,
            "entity_types": {r["type"]: r["c"] for r in types},
            "relation_types": {r["type"]: r["c"] for r in rel_types},
        }

    def query_entity_history(self, entity_name: str) -> List[Dict]:
        """
        Get all relations involving an entity across meetings.
        This is the temporal evolution query for your demo.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            results = session.run(
                """
                MATCH (e:Entity {name: $name})-[r:RELATION]-(other:Entity)
                RETURN 
                    e.name AS entity,
                    r.type AS relation,
                    other.name AS related_entity,
                    r.source_meeting AS meeting,
                    r.timestamp AS timestamp,
                    r.confidence AS confidence,
                    startNode(r).name = e.name AS is_outgoing
                ORDER BY r.timestamp
                """,
                name=entity_name,
            ).data()

        return results

    def query_meeting_decisions(self, meeting_id: str) -> List[Dict]:
        """Get all decision-type relations from a specific meeting."""
        if not self.driver:
            return []

        decision_relations = [
            "approved", "rejected", "decided_on", "postponed",
            "proposed", "assigned_to", "scheduled_for",
        ]

        with self.driver.session() as session:
            results = session.run(
                """
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE r.source_meeting = $meeting_id 
                  AND r.type IN $decision_types
                RETURN 
                    s.name AS subject,
                    r.type AS decision,
                    o.name AS object,
                    r.confidence AS confidence,
                    r.timestamp AS timestamp
                ORDER BY r.timestamp
                """,
                meeting_id=meeting_id,
                decision_types=decision_relations,
            ).data()

        return results

    def query_cross_meeting_evolution(self, entity_name: str) -> List[Dict]:
        """
        Track how an entity's relationships change across meetings.
        Key query for demonstrating temporal knowledge evolution.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            results = session.run(
                """
                MATCH (e:Entity)-[r:RELATION]-(other:Entity)
                WHERE e.name CONTAINS $name
                WITH r.source_meeting AS meeting, 
                     collect({
                         relation: r.type, 
                         entity: other.name, 
                         direction: CASE WHEN startNode(r).name = e.name 
                                    THEN 'outgoing' ELSE 'incoming' END,
                         confidence: r.confidence
                     }) AS relations
                RETURN meeting, relations
                ORDER BY meeting
                """,
                name=entity_name,
            ).data()

        return results

    def clear_all(self):
        """Delete all nodes and relationships. Use with caution."""
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[Neo4j] Graph cleared")


class FlatFileGraph:
    """
    Fallback graph store using JSON files.
    Use this if Neo4j isn't available — lets you develop and test
    the rest of the pipeline without the database dependency.
    """

    def __init__(self, output_dir: str = "./graph_output"):
        self.output_dir = output_dir
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []
        import os
        os.makedirs(output_dir, exist_ok=True)

    def connect(self) -> bool:
        print(f"[FlatFileGraph] Using JSON files in {self.output_dir}")
        return True

    def close(self):
        self._save()

    def setup_constraints(self):
        pass

    def insert_triples(self, triples: List[Triple], meeting_id: str = ""):
        timestamp = datetime.now().isoformat()
        for t in triples:
            # Upsert entities
            self.entities[t.subject.lower()] = {
                "name": t.subject,
                "type": t.subject_type,
                "last_seen": timestamp,
                "last_meeting": meeting_id,
            }
            self.entities[t.object.lower()] = {
                "name": t.object,
                "type": t.object_type,
                "last_seen": timestamp,
                "last_meeting": meeting_id,
            }
            # Add relation
            self.relations.append({
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "confidence": t.confidence,
                "meeting": meeting_id,
                "timestamp": timestamp,
            })

        self._save()
        print(f"[FlatFileGraph] Stored {len(triples)} triples from '{meeting_id}'")

    def get_graph_stats(self) -> Dict:
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
        }

    def _save(self):
        import os
        with open(os.path.join(self.output_dir, "entities.json"), "w", encoding="utf-8") as f:
            json.dump(self.entities, f, indent=2)
        with open(os.path.join(self.output_dir, "relations.json"), "w", encoding="utf-8") as f:
            json.dump(self.relations, f, indent=2)

    def clear_all(self):
        self.entities = {}
        self.relations = []
        self._save()
