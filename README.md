# 📚 Zotero RAG Navigator

A question-answering system for your uploaded PDFs or Zotero library featuring a multi-stage RAG pipeline with GROBID parsing, Qdrant hybrid search, contextualization, reranking, and extractive QA. Generate precise answers from your research papers with automatic highlighting and question expansion.

## Features

**3-Stage RAG Pipeline**: Qdrant Retrieval → CrossEncoder reranking → Extractive QA

- **Flexible PDF sources**: PDF upload or Zotero library selection
- GROBID sentence-level parsing with coordinates for precise PDF highlighting
- Dense + Sparse retrieval with Qdrant hybrid search
- Chunk contextualization via Ollama before embedding (optional)
- Optional question paraphrases generated on request and included only when selected
- Three adaptive rerank threshold presets plus manual custom thresholds
- Sliding window QA for answers spanning paragraph boundaries
- Multi-color highlighting for multiple queries
- Streamlit web interface with real-time progress tracking

## Requirements

- Python 3.11+
- Zotero (with local PDF storage) **OR** PDF files on your device
- GROBID server (Docker)
- Qdrant server (Docker)
- Ollama server (Docker)
- PyTorch with MPS/CUDA support (optional, CPU works too)

## Project Structure

```
.
├── zotero_rag/                 # Main package
│   ├── app.py                  # Streamlit web interface
│   ├── embedding_manager.py    # Dense, sparse, and contextual embeddings
│   ├── highlighter.py          # PDF annotation using coordinates
│   ├── models.py               # Data classes
│   ├── pdf_cache_handler.py    # Local PDF cache management
│   ├── pdf_processor.py        # GROBID client and TEI parsing
│   ├── pdf_utils.py            # PDF utilities
│   ├── qa_engine.py            # Extractive QA with question expansion
│   ├── qdrant_manager.py       # Qdrant hybrid indexing and search
│   ├── reranker.py             # CrossEncoder reranking
│   ├── run_from_config.py      # Programmatic YAML config runner
│   ├── zotero_db.py            # Zotero SQLite database interface
│   └── zotero_rag.py           # Main orchestration class
│
├── example_configs/                # Example YAML configurations
│   ├── example_config.yaml         # Full YAML config with documentation
│   ├── advanced_config.yaml        # Advanced YAML config with custom paraphrases
│   ├── folder_example_config.yaml  # Example config for PDF upload workflows
│   ├── threshold_presets.yaml      # Threshold preset reference
│   └── highlighter_colors.html     # Color reference for highlighting
│
├── pyproject.toml           # Poetry dependencies
├── README.md                # This file
└── LICENSE                  # GPL v3.0 license

output/                     # Output directory (indexes, cache, highlighted PDFs)
├── highlighted_results
│   └──{title}.pdf          #Highlighted PDFs
|
├── pdf_cache
│   └──{hash}.pdf           #Indexed PDFs cache
|
└── tei_cache/
    └── {hash}.tei.xml      #GROBID output cache
```

## Installation

### 1. Clone & Setup Environment

```bash
git clone https://github.com/eliroc98/zoteroRAG.git
cd zoteroRAG
poetry install
```

### 2. Start GROBID Service

```bash
docker run -d -p 8070:8070 grobid/grobid:latest
```

Verify it's running:
```bash
curl http://localhost:8070/api/isalive
```

### 3. Start Qdrant Service

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Verify it's running:
```bash
curl http://localhost:6333/
```

### 4. Start Ollama Service (Optional)

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec -it ollama ollama pull llama3.2:3b
```

Verify it's running and contains the model:
```bash
docker exec -it ollama ollama list
```

### 5. Run the App

```bash
poetry run streamlit run zotero_rag/app.py
```

Then navigate to `http://localhost:8501`

## Usage

### Initial Setup

1. **Output Directory** (first page)
   - Specify where to store PDF cache, TEI cache, and highlighted PDFs
   - Default: `./literature_output`

2. **GROBID Configuration**
   - Service URL (default: `http://localhost:8070`)
   - Leave default if running locally
   - Required for new PDFs; cached TEI files are reused

3. **Qdrant Configuration**
    - Service URL (default: `http://localhost:6333`)
    - Leave default if running locally
    - Used for hybrid indexing and retrieval

4. **Ollama Configuration (Optional)**
    - Service URL (default: `http://localhost:11434`)
    - Leave default if running locally
    - Used for chunk contextualization before embedding

5. **Select Embedding Model**
   - Model name: Any HuggingFace SentenceTransformer (e.g., `BAAI/bge-base-en-v1.5`)
   - Device: auto/cpu/mps/cuda
   - Batch sizes: Auto-detected or manual override
   - Click "Load Model"

6. **Add PDFs to Index**
    
    Choose between two ways of selecting PDFs to index:

    - **📤 Upload PDFs**: Add one or more PDF files directly from your device

    - **📚 Zotero Collection**: Choose a collection or "All Library", all PDFs in the selected Zotero scope will be indexed

### Question Answering

1. **Enter a Question**
   - Natural language queries supported
   - Example: "What methods are used for circuit interpretability?"

2. **Choose a Threshold Preset**
    - **Low**: Retrieve 0.35; rerank at mean - 2 * std to retain more candidates.
    - **Medium**: Retrieve 0.45; rerank at mean - std for general use.
    - **High**: Retrieve 0.45; rerank at mean + 2 * std for stricter relevance.
    - **Custom**: Manually set Retrieve, Rerank, and QA thresholds.
    - QA threshold is 0.10 for the three presets.


3. **Question Expansion** (Optional)
   - Generate paraphrases to improve retrieval
    - Select/edit generated variations; none are selected by default, and the original question is always included
    - If no paraphrase is generated or selected, only the original question is searched
   - Automatically merges results from all variations

4. **Search & Navigate**
   - View answers with context and scores
   - Navigate results with Previous/Next buttons
   - See PDF source, page number, and section
   - Click "Open PDF" to view in default viewer

5. **Highlight Color**
   - Choose from presets (Yellow, Cyan, Orange, Green, Pink, Purple)
   - Or use custom RGB color picker

6. **Highlight PDF**
   - Select only the answers you like to highlight
   - Click "Highlight PDF" to add colored annotations

7. **Multi-Query Highlighting**
   - Run multiple queries with different colors
   - All highlights accumulate in the same PDF
   - Perfect for exploring different aspects of a paper

## Architecture

### Data Flow

```
PDF Selection (PDF upload OR Zotero Library)
    ↓
GROBID Processing (sentence segmentation + coordinates)
    ↓
TEI Cache (mtime-keyed, persistent)
    ↓
Paragraph Extraction (section classification)
    ↓
Chunk Contextualization (Ollama, optional)
    ↓
Dense + Sparse Encoding (auto batch-size)
    ↓
Qdrant Hybrid Index
```

### Query Pipeline

```
User Question
    ↓
Question Expansion (optional paraphrasing)
Only generated and explicitly selected paraphrases are included; otherwise the original question is used.
    ↓
Qdrant Hybrid Retrieval (dense + sparse, all variations)
    ↓
CrossEncoder Reranking (adaptive threshold)
    ↓
Extractive QA (sliding window with context overlap)
    ↓
Answer Deduplication & Scoring
    ↓
PDF Highlighting (TEI coordinate mapping)
```

## Configuration

### Environment Variables (Optional)

```bash
# Default Zotero directory (auto-detected from ~/Zotero, ~/Documents/Zotero, ~/.zotero)
export ZOTERO_DATA_DIR=/path/to/zotero

# GROBID timeout (seconds)
export GROBID_TIMEOUT=180

# Disable progress bars (useful in headless environments)
export TQDM_DISABLE=1

# Tokenizer parallelism
export TOKENIZERS_PARALLELISM=false
```

### Threshold Presets

The threshold reference is in `example_configs/threshold_presets.yaml`. Presets calculate the rerank cutoff from the current score distribution; selecting **Custom** uses the supplied rerank value directly.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{zoterorag2026,
  author = {Elisabetta Rocchetti},
  title = {Zotero RAG Navigator: Multi-Stage Question Answering for Research Libraries},
  year = {2026},
  url = {https://github.com/eliroc98/zoteroRAG}
}
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
