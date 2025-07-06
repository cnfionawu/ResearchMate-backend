from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time

from app.database import (
    init_db,
    fetch_arxiv,
    fetch_semantic_scholar,
    fetch_openalex,
    save_papers,
    query_papers_from_db,
    get_query_last_fetched,
    update_query_timestamp,
    get_all_papers_from_db,
)
from app.retrieval import hybrid_search
from app.summarizer import summarize

app = Flask(__name__)
CORS(app)
init_db()

STALE_SECONDS = 7 * 24 * 3600  # 1 week
TOP_K = 5


@app.route('/search')
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter required"}), 400

    # Refresh if stale or first time
    last_fetched = get_query_last_fetched(query)
    if last_fetched is None or time.time() - last_fetched > STALE_SECONDS:
        new_arxiv = fetch_arxiv(query)
        new_semantic = fetch_semantic_scholar(query)
        new_openalex = fetch_openalex(query)
        new_papers = new_arxiv + new_semantic + new_openalex
        save_papers(new_papers)
        update_query_timestamp(query)
    else:
        print(f"Using cached data for '{query}'")

    local_results = query_papers_from_db(query)
    print(f"Local results found: {len(local_results)}")

    if not local_results:
        return jsonify({"message": f"No papers found for query '{query}'"}), 404
    elif len(local_results) < TOP_K:
        print("Fallback: searching all papers in DB")
        local_results = get_all_papers_from_db()

    ranked = hybrid_search(query, local_results)
    top = ranked[:TOP_K]
    summaries = summarize([paper[3] for paper in top])  # abstract index

    print(f"Hybrid searched {len(top)} papers for query '{query}'")

    return jsonify([
        {
            "title": paper[1],
            "authors": paper[2],
            "summary": s,
            "source": paper[4]
        } for paper, s in zip(top, summaries)
    ])


@app.route('/refresh')
def refresh():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter required"}), 400

    new_arxiv = fetch_arxiv(query)
    new_semantic = fetch_semantic_scholar(query)
    new_openalex = fetch_openalex(query)
    new_papers = new_arxiv + new_semantic + new_openalex
    save_papers(new_papers)
    update_query_timestamp(query)

    return jsonify({
        "message": f"Refreshed {len(new_papers)} papers for query '{query}'."
    })