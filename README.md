# Sejm-Whiz: Polish Legal Document Analysis Research

> **Personal Research Project**: Exploring AI architecture patterns for legal document analysis using open-source Polish NLP models. Developed during personal time for educational and skill development purposes.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Semantic Versioning](https://img.shields.io/badge/semver-2.0.0-blue)](https://semver.org/)
[![Conventional Commits](https://img.shields.io/badge/conventional%20commits-1.0.0-%23FE5196)](https://conventionalcommits.org/)

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

## Deployment Options

For production-style deployments, several approaches are documented:

### Baremetal Deployment (Recommended)

- **[Complete Baremetal Setup](deployments/baremetal/README.md)** - Production deployment on Debian 12
  - **Infrastructure Guides**:
    - [PostgreSQL 17 + pgvector Setup](deployments/baremetal/docs/P7_POSTGRESQL_SETUP.md) - Database configuration for AI workloads
    - [Redis 7.0 Setup](deployments/baremetal/docs/P7_REDIS_SETUP.md) - High-performance caching configuration
  - SystemD service management
  - Network-accessible deployment architecture
  - Performance optimization for AI workloads

### Alternative Deployments

- **[Docker Deployment](deployments/docker/README.md)** - Containerized deployment with Docker Compose
- **[K3s Deployment](deployments/k3s/README.md)** - Kubernetes deployment for container orchestration

> **Note**: The baremetal approach provides the most control and performance for AI workloads, with comprehensive infrastructure setup guides specifically written for Debian 12 environments.

## Version Management

This project follows [Semantic Versioning 2.0.0](https://semver.org/) and uses [Conventional Commits](https://conventionalcommits.org/) for automated version management:

### Version Format

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes that require code updates
- **MINOR**: New features that are backward compatible
- **PATCH**: Bug fixes that are backward compatible

### Making Releases

```bash
# Automatic version bump based on commit messages
uv run cz bump

# Manual version increments
uv run cz bump --increment PATCH  # 0.1.0 → 0.1.1
uv run cz bump --increment MINOR  # 0.1.0 → 0.2.0
uv run cz bump --increment MAJOR  # 0.1.0 → 1.0.0

# Alternative with bumpversion
uv run bumpversion patch  # Bug fixes
uv run bumpversion minor  # New features
uv run bumpversion major  # Breaking changes
```

### Commit Message Format

Use conventional commit format for automatic version detection:

```
feat: add new document parsing feature     # Minor version bump
fix: resolve embedding generation issue    # Patch version bump
BREAKING CHANGE: change API response       # Major version bump
docs: update installation instructions    # No version bump
refactor: improve code readability        # Patch version bump
```

### Release Process

1. Follow conventional commit format in your commits
1. Run `uv run cz bump` to automatically determine version increment
1. Review the generated changelog in `CHANGELOG.md`
1. Push tags: `git push --tags`

The version is automatically updated in:

- `components/sejm_whiz/__init__.py`
- `CHANGELOG.md` (with automatic generation)

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
