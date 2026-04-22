"""
Natural Language Query Interface for the Knowledge Graph.

Takes a plain English question, uses an LLM to generate a Cypher query,
runs it against Neo4j, and returns a natural language answer.

Also supports the flat file graph backend for development without Neo4j.

Usage:
    python query.py "What decisions were made about the AI elective?"
    python query.py "How did the GPU budget change across meetings?"
    python query.py --interactive   # Chat-style Q&A loop
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional

from graph.neo4j_graph import KnowledgeGraph, FlatFileGraph


# Schema description for the LLM — tells it what's in the graph
GRAPH_SCHEMA = """
The Neo4j knowledge graph has the following structure:

NODE:
  (:Entity)
    Properties: name (string), type (string), first_seen (datetime), 
                last_seen (datetime), first_meeting (string), last_meeting (string)
    Entity types: PERSON, COURSE, DEPARTMENT, COMMITTEE, PROJECT, 
                  RESOURCE, EVENT, DEADLINE, POLICY, TOPIC

RELATIONSHIP:
  (:Entity)-[:RELATION]->(:Entity)
    Properties: type (string), confidence (float), source_meeting (string),
                timestamp (datetime), utterance_time (float)
    Relation types include: teaches, heads, member_of, assigned_to, proposed,
                           approved, rejected, postponed, deadline_for, reports_to,
                           depends_on, blocked_by, discussed, decided_on, 
                           scheduled_for, budget_for, supervises, part_of

IMPORTANT QUERY PATTERNS:
- Relations are stored with a 'type' property, NOT as separate relationship types.
  Use: MATCH ()-[r:RELATION]->() WHERE r.type = 'approved'
  NOT:  MATCH ()-[r:approved]->()
- Entity names are stored with original casing.
- Use CONTAINS for partial name matching.
- source_meeting tracks which meeting a fact came from.
- Order by r.timestamp to see temporal evolution.
"""

CYPHER_SYSTEM_PROMPT = """You are a Cypher query generator for a Neo4j knowledge graph built from college staff meeting transcripts.

{schema}

Given a natural language question, generate a Cypher query that answers it.

RULES:
1. Return ONLY the Cypher query. No explanation, no markdown fences, no comments.
2. Always use the RELATION relationship type with a 'type' property filter.
3. Use CONTAINS for name matching (handles partial matches).
4. Use OPTIONAL MATCH if some results might not have certain properties.
5. Always RETURN readable aliases (AS keyword) so the results make sense.
6. Limit results to 25 rows max.
7. For temporal queries, ORDER BY r.timestamp or r.source_meeting.

EXAMPLES:
Question: What decisions were made?
Query: MATCH (s:Entity)-[r:RELATION]->(o:Entity) WHERE r.type IN ['approved', 'rejected', 'decided_on', 'postponed', 'proposed'] RETURN s.name AS subject, r.type AS decision, o.name AS object, r.source_meeting AS meeting ORDER BY r.timestamp LIMIT 25

Question: Who is assigned to what?
Query: MATCH (p:Entity)-[r:RELATION]->(t:Entity) WHERE r.type = 'assigned_to' RETURN p.name AS person, t.name AS task, r.source_meeting AS meeting, r.confidence AS confidence ORDER BY r.timestamp LIMIT 25

Question: How did the budget change?
Query: MATCH (e:Entity)-[r:RELATION]-(other:Entity) WHERE e.name CONTAINS 'budget' OR other.name CONTAINS 'budget' OR r.type = 'budget_for' RETURN e.name AS entity, r.type AS relation, other.name AS related_to, r.source_meeting AS meeting ORDER BY r.timestamp LIMIT 25"""


ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about college staff meetings based on knowledge graph query results.

Given the user's original question and the query results from the knowledge graph, provide a clear, natural language answer.

RULES:
1. Answer directly and concisely.
2. If results are empty, say you couldn't find relevant information.
3. If the results show changes across meetings, highlight the temporal evolution.
4. Mention which meeting(s) the information comes from.
5. Don't mention Cypher, Neo4j, or the graph — just answer naturally as if you know the information.
6. If confidence scores are low (below 0.7), mention that the information is uncertain."""


def call_llm(system_prompt: str, user_prompt: str, model: str = "", api_key: str = "", base_url: str = "") -> str:
    """Call LLM via OpenAI-compatible API. Same approach as the extractor."""
    import urllib.request
    import ssl

    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or os.getenv("LLM_MODEL", "gemini-2.0-flash")

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    if "localhost" in base_url or "127.0.0.1" in base_url:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"].strip()


def generate_cypher(question: str, **llm_kwargs) -> str:
    """Generate a Cypher query from a natural language question."""
    system = CYPHER_SYSTEM_PROMPT.format(schema=GRAPH_SCHEMA)
    response = call_llm(system, question, **llm_kwargs)

    # Clean up: remove markdown fences if the LLM adds them
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response = "\n".join(lines).strip()

    return response


def run_cypher(graph: KnowledgeGraph, query: str) -> List[Dict]:
    """Execute a Cypher query against Neo4j and return results."""
    try:
        with graph.driver.session() as session:
            result = session.run(query)
            return result.data()
    except Exception as e:
        return [{"error": str(e)}]


def generate_answer(question: str, results: List[Dict], **llm_kwargs) -> str:
    """Generate a natural language answer from query results."""
    if not results:
        results_text = "No results found."
    elif "error" in results[0]:
        results_text = f"Query error: {results[0]['error']}"
    else:
        results_text = json.dumps(results, indent=2, default=str)

    user_prompt = f"""Question: {question}

Query results:
{results_text}

Provide a clear answer based on these results."""

    return call_llm(ANSWER_SYSTEM_PROMPT, user_prompt, **llm_kwargs)


def query_flat_file(question: str, output_dir: str = "./output", **llm_kwargs) -> str:
    """
    Query the flat file graph when Neo4j isn't available.
    Loads the JSON files and lets the LLM answer from them directly.
    """
    entities_path = os.path.join(output_dir, "entities.json")
    relations_path = os.path.join(output_dir, "relations.json")

    entities = {}
    relations = []

    if os.path.exists(entities_path):
        with open(entities_path) as f:
            entities = json.load(f)

    if os.path.exists(relations_path):
        with open(relations_path) as f:
            relations = json.load(f)

    if not entities and not relations:
        # Try loading individual triple files
        all_triples = []
        for fname in sorted(os.listdir(output_dir)):
            if fname.startswith("triples_") and fname.endswith(".json"):
                with open(os.path.join(output_dir, fname)) as f:
                    all_triples.extend(json.load(f))

        if not all_triples:
            return "No knowledge graph data found. Run the pipeline first to extract triples."

        graph_data = json.dumps(all_triples, indent=2)
    else:
        graph_data = json.dumps({
            "entities": entities,
            "relations": relations,
        }, indent=2)

    system = """You are a helpful assistant that answers questions about college staff meetings.
You have access to a knowledge graph extracted from meeting transcripts.
Answer the question based on the graph data provided.
Be specific — mention names, dates, and meetings when available.
If the information isn't in the data, say so."""

    user_prompt = f"""Question: {question}

Knowledge graph data:
{graph_data}

Answer the question based on this data."""

    return call_llm(system, user_prompt, **llm_kwargs)


def ask(question: str, graph=None, output_dir: str = "./output", verbose: bool = False, **llm_kwargs) -> str:
    """
    Main query function. Works with both Neo4j and flat file backends.
    
    Args:
        question: Natural language question
        graph: KnowledgeGraph instance (if using Neo4j)
        output_dir: Directory containing flat file graph data
        verbose: Print intermediate Cypher query
        **llm_kwargs: model, api_key, base_url for LLM
    
    Returns:
        Natural language answer
    """
    if graph and graph.driver:
        # Neo4j path: question → Cypher → execute → answer
        cypher = generate_cypher(question, **llm_kwargs)
        if verbose:
            print(f"\n[Cypher] {cypher}\n")

        results = run_cypher(graph, cypher)
        if verbose:
            print(f"[Results] {json.dumps(results, indent=2, default=str)}\n")

        answer = generate_answer(question, results, **llm_kwargs)
        return answer
    else:
        # Flat file path: question + all data → LLM answers directly
        return query_flat_file(question, output_dir=output_dir, **llm_kwargs)


def interactive_mode(graph=None, output_dir: str = "./output", verbose: bool = False, **llm_kwargs):
    """Interactive Q&A loop."""
    print("=" * 60)
    print("Knowledge Graph Query Interface")
    print("Ask questions about your meetings in plain English.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    mode = "Neo4j" if (graph and graph.driver) else "Flat file"
    print(f"Backend: {mode}\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            answer = ask(question, graph=graph, output_dir=output_dir,
                        verbose=verbose, **llm_kwargs)
            print(f"\nAnswer: {answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Query the Knowledge Graph")

    parser.add_argument("question", nargs="?", default=None,
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive Q&A mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show generated Cypher queries and raw results")

    # Graph backend
    parser.add_argument("--no-neo4j", action="store_true",
                        help="Query flat file graph instead of Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--output-dir", default="./output",
                        help="Directory containing flat file graph data")

    # LLM settings
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)

    args = parser.parse_args()

    llm_kwargs = {}
    if args.llm_model:
        llm_kwargs["model"] = args.llm_model
    if args.api_key:
        llm_kwargs["api_key"] = args.api_key
    if args.api_base:
        llm_kwargs["base_url"] = args.api_base

    # Connect to graph backend
    graph = None
    if not args.no_neo4j:
        graph = KnowledgeGraph(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
        )
        if not graph.connect():
            print("[WARN] Neo4j not available. Falling back to flat file query.")
            graph = None

    if args.interactive or args.question is None:
        interactive_mode(graph=graph, output_dir=args.output_dir,
                        verbose=args.verbose, **llm_kwargs)
    else:
        answer = ask(args.question, graph=graph, output_dir=args.output_dir,
                    verbose=args.verbose, **llm_kwargs)
        print(answer)

    if graph:
        graph.close()


if __name__ == "__main__":
    main()
