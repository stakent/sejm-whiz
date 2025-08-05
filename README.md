# Sejm-Whiz: Polish Legal Document Analysis Research

> **Personal Research Project**: Exploring AI architecture patterns for legal document analysis using open-source Polish NLP models. Developed during personal time for educational and skill development purposes.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## About This Project

This is a personal exploration of component-based AI architectures, undertaken outside of any employment context using publicly available data and open-source tools. The project investigates how modern NLP techniques can be applied to Polish legal texts while maintaining production-ready design principles.

**Research Focus**: Component isolation, scalable architecture patterns, and efficient NLP processing for morphologically complex languages like Polish.

## Motivation

Like many developers, I believe in continuous learning through hands-on experimentation. This project emerged from personal curiosity about:

- How component-based architectures (Polylith) can simplify AI system complexity
- Whether Polish BERT models can effectively handle legal document semantics
- What architectural patterns enable cost-efficient scaling for specialized NLP tasks

## Technical Exploration

### Architecture Experiments

The project explores **Polylith architecture** - a component-based approach that treats code as composable blocks. This personal research investigates:

- **Component Isolation**: Can AI components be truly independent and swappable?
- **Semantic Search at Scale**: How efficient is PostgreSQL + pgvector compared to specialized vector databases?
- **Resource Optimization**: What are practical approaches for GPU-constrained environments?

### Key Learning Areas

Through this personal project, I've explored:

- **Polish NLP Challenges**: Morphological complexity, legal terminology disambiguation
- **Embedding Strategies**: Bag-of-embeddings for document-level similarity
- **Production Patterns**: Monitoring, caching, and deployment strategies
- **Cost-Efficient Design**: Avoiding vendor lock-in while maintaining performance

## Implementation Highlights

The research implementation includes several experimental components:

```
Components (Research Modules):
├── embeddings/        # HerBERT integration experiments
├── legal_nlp/         # Legal document analysis patterns
├── semantic_search/   # Vector similarity approaches
├── text_processing/   # Polish text preprocessing
└── vector_db/         # pgvector optimization studies

Projects (Integration Tests):
├── api_server/        # FastAPI service patterns
├── data_processor/    # Batch processing experiments
└── web_ui/           # Monitoring dashboard prototype
```

## Research Findings

### What Works Well

1. **Polylith for AI**: Component isolation genuinely simplifies testing and development
1. **PostgreSQL + pgvector**: Surprisingly effective for moderate-scale vector operations
1. **HerBERT**: Handles Polish legal language better than multilingual models
1. **Component Testing**: Independent component tests catch issues early

### Ongoing Investigations

- Optimal chunking strategies for legal documents
- Cross-reference graph construction from legal citations
- Prediction accuracy for legislative change patterns

## Getting Started

This project is shared for educational purposes. To explore the code:

```bash
# Clone the repository
git clone https://github.com/yourusername/sejm-whiz.git
cd sejm-whiz

# Install dependencies (requires Python 3.12+)
uv sync --dev

# Run component tests
uv run poly test

# Explore individual components
uv run python development/main.py
```

### Prerequisites for Local Experimentation

- Python 3.12+ (for modern async features)
- PostgreSQL with pgvector (for vector similarity)
- NVIDIA GPU (optional, for faster embeddings)
- Redis (for caching experiments)

## Personal Learning Outcomes

This project has deepened my understanding of:

- **System Design**: How to structure AI applications for maintainability
- **Polish NLP**: Unique challenges in processing morphologically rich languages
- **Production Patterns**: Practical approaches to scaling AI systems
- **Open Source**: Contributing back to the community through shared learnings

## Technology Choices

Selected for learning and experimentation:

- **Python**: Rich ecosystem for AI/ML experimentation
- **Polylith**: Exploring component-based architecture patterns
- **FastAPI**: Modern async patterns and automatic documentation
- **PostgreSQL**: Production-grade database with vector extensions
- **HerBERT**: State-of-the-art Polish language understanding

## Future Explorations

Areas I'm considering for future personal research:

- Graph neural networks for legal citation networks
- Multi-lingual legal document alignment
- Streaming architectures for real-time processing
- Federated learning for privacy-preserving legal AI

## Community and Learning

This project is part of my personal journey in AI/ML. I'm always interested in:

- Discussing architectural patterns for AI systems
- Learning about NLP challenges in other languages
- Sharing experiences with component-based design
- Exploring production deployment strategies

Feel free to explore the code, and if you're working on similar challenges, I'd love to connect and exchange ideas!

## License

MIT License - This personal research project is open source for educational purposes.

## Acknowledgments

- The Polish NLP community for creating HerBERT
- Polylith creators for the architectural inspiration
- Open data initiatives making legal documents accessible
- The broader open-source community for tools and libraries

______________________________________________________________________

*This is a personal research project developed during my free time for continuous learning and skill development. All work was conducted outside of any employment context using publicly available resources.*

## Contact

If you're interested in discussing AI architectures, Polish NLP, or component-based design:

- LinkedIn: https://www.linkedin.com/in/dariusz-walat
- GitHub: [@stakent]

*Note: This is an ongoing personal exploration. Code quality and completeness vary as I experiment with different approaches.*
