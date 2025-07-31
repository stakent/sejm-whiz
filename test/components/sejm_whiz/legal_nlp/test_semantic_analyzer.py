"""Tests for legal_nlp semantic analyzer functionality."""

from sejm_whiz.legal_nlp.semantic_analyzer import LegalSemanticAnalyzer, SemanticField


class TestLegalSemanticAnalyzer:
    """Test cases for LegalSemanticAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LegalSemanticAnalyzer()

        # Sample legal text with various semantic fields
        self.civil_law_text = """
        Kodeks cywilny reguluje stosunki cywilnoprawne między osobami fizycznymi i prawnymi.
        Własność jest prawem rzeczowym, które daje właścicielowi pełną władzę nad rzeczą.
        Umowa zobowiązuje strony do wykonania świadczeń zgodnie z jej treścią.
        W przypadku niewykonania zobowiązania, dłużnik ponosi odpowiedzialność odszkodowawczą.
        Małżeństwo jest związkiem kobiety i mężczyzny zawartym w celu wspólnego pożycia.
        """

        self.criminal_law_text = """
        Kodeks karny określa przestępstwa i wykroczenia oraz kary za ich popełnienie.
        Zabójstwo to umyślne pozbawienie życia człowieka i podlega karze więzienia.
        Kradzież polega na zabieraniu cudzej rzeczy ruchomej w celu przywłaszczenia.
        Prokurator prowadzi postępowanie przygotowawcze przeciwko podejrzanemu.
        Za przestępstwo grozi kara grzywny lub pozbawienia wolności.
        """

        self.definition_text = """
        Przez nieruchomość rozumie się grunty oraz budynki trwale z gruntem związane.
        Osoba fizyczna oznacza człowieka od chwili urodzenia do śmierci.
        Przedsiębiorca to osoba fizyczna, prawna lub jednostka organizacyjna nieposiadająca osobowości prawnej.
        Konsument w rozumieniu niniejszej ustawy to osoba fizyczna dokonująca czynności prawnej.
        """

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, "semantic_fields")
        assert hasattr(self.analyzer, "semantic_patterns")
        assert len(self.analyzer.semantic_fields) > 0

    def test_semantic_fields_structure(self):
        """Test semantic fields structure."""
        for field_name, field_info in self.analyzer.semantic_fields.items():
            assert isinstance(field_info, SemanticField)
            assert isinstance(field_info.field_name, str)
            assert isinstance(field_info.terms, list)
            assert len(field_info.terms) > 0
            assert isinstance(field_info.weight, (int, float))
            assert field_info.weight > 0
            assert isinstance(field_info.context_terms, list)

    def test_identify_semantic_fields_civil_law(self):
        """Test semantic field identification for civil law."""
        fields = self.analyzer.identify_semantic_fields(self.civil_law_text)

        assert isinstance(fields, dict)
        assert len(fields) > 0

        # Should identify civil law as the primary field
        assert "civil_law" in fields
        assert fields["civil_law"] > 0

        # Should have higher score than other fields
        if len(fields) > 1:
            civil_law_score = fields["civil_law"]
            other_scores = [
                score for field, score in fields.items() if field != "civil_law"
            ]
            assert civil_law_score >= max(other_scores, default=0)

    def test_identify_semantic_fields_criminal_law(self):
        """Test semantic field identification for criminal law."""
        fields = self.analyzer.identify_semantic_fields(self.criminal_law_text)

        assert isinstance(fields, dict)
        assert "criminal_law" in fields
        assert fields["criminal_law"] > 0

    def test_identify_semantic_fields_empty_text(self):
        """Test semantic field identification with empty text."""
        fields = self.analyzer.identify_semantic_fields("")
        assert fields == {}

        fields = self.analyzer.identify_semantic_fields(None)
        assert fields == {}

    def test_extract_semantic_relations(self):
        """Test semantic relation extraction."""
        relations = self.analyzer.extract_semantic_relations(self.civil_law_text)

        assert isinstance(relations, dict)

        # Check expected relation types
        expected_types = [
            "causal_relations",
            "temporal_relations",
            "modal_expressions",
            "conditional_relations",
        ]
        for rel_type in expected_types:
            assert rel_type in relations
            assert isinstance(relations[rel_type], list)

    def test_extract_semantic_relations_empty_text(self):
        """Test semantic relation extraction with empty text."""
        relations = self.analyzer.extract_semantic_relations("")
        assert relations == {}

    def test_semantic_relation_structure(self):
        """Test structure of extracted semantic relations."""
        relations = self.analyzer.extract_semantic_relations(self.civil_law_text)

        for rel_type, rel_list in relations.items():
            for relation in rel_list:
                assert isinstance(relation, dict)
                assert "relation_text" in relation
                assert "extracted_content" in relation
                assert "position" in relation
                assert "context" in relation

                assert isinstance(relation["position"], dict)
                assert "start" in relation["position"]
                assert "end" in relation["position"]

    def test_analyze_conceptual_density(self):
        """Test conceptual density analysis."""
        density = self.analyzer.analyze_conceptual_density(self.civil_law_text)

        assert isinstance(density, dict)
        assert "total_words" in density
        assert "legal_terms_count" in density
        assert "unique_legal_terms" in density
        assert "density_score" in density
        assert "unique_terms_ratio" in density
        assert "legal_terms_found" in density

        assert density["total_words"] > 0
        assert density["legal_terms_count"] >= 0
        assert density["unique_legal_terms"] >= 0
        assert 0 <= density["density_score"] <= 1
        assert 0 <= density["unique_terms_ratio"] <= 1
        assert isinstance(density["legal_terms_found"], list)

    def test_analyze_conceptual_density_empty_text(self):
        """Test conceptual density with empty text."""
        density = self.analyzer.analyze_conceptual_density("")

        assert density["total_words"] == 0
        assert density["legal_terms_count"] == 0
        assert density["density_score"] == 0.0

    def test_extract_legal_definitions(self):
        """Test legal definition extraction."""
        definitions = self.analyzer.extract_legal_definitions(self.definition_text)

        assert isinstance(definitions, list)
        assert len(definitions) > 0

        for definition in definitions:
            assert isinstance(definition, dict)
            assert "term" in definition
            assert "definition" in definition
            assert "position" in definition
            assert "context" in definition
            assert "confidence" in definition

            assert isinstance(definition["term"], str)
            assert len(definition["term"]) > 0
            assert isinstance(definition["definition"], str)
            assert len(definition["definition"]) > 0
            assert 0 <= definition["confidence"] <= 1

    def test_extract_legal_definitions_empty_text(self):
        """Test definition extraction with empty text."""
        definitions = self.analyzer.extract_legal_definitions("")
        assert definitions == []

    def test_definition_confidence_calculation(self):
        """Test definition confidence calculation."""
        # Test with a good definition
        confidence1 = self.analyzer._calculate_definition_confidence(
            "nieruchomość",
            "grunty oraz budynki trwale z gruntem związane zgodnie z przepisami prawa",
        )

        # Test with a poor definition
        confidence2 = self.analyzer._calculate_definition_confidence("x", "y")

        assert 0 <= confidence1 <= 1
        assert 0 <= confidence2 <= 1
        assert (
            confidence1 > confidence2
        )  # Better definition should have higher confidence

    def test_deduplicate_definitions(self):
        """Test definition deduplication."""
        definitions = [
            {"term": "test", "definition": "def1", "confidence": 0.8},
            {"term": "test", "definition": "def2", "confidence": 0.6},
            {"term": "other", "definition": "def3", "confidence": 0.7},
        ]

        unique_defs = self.analyzer._deduplicate_definitions(definitions)

        assert len(unique_defs) == 2  # Should remove one duplicate
        terms = [d["term"] for d in unique_defs]
        assert "test" in terms
        assert "other" in terms
        assert terms.count("test") == 1  # Only one instance of 'test'

    def test_analyze_argumentative_structure(self):
        """Test argumentative structure analysis."""
        arg_text = """
        Ponieważ prawo własności jest prawem podstawowym, dlatego jego ochrona jest konieczna.
        Jednak w niektórych przypadkach można ograniczyć to prawo.
        Uzasadnienie takiego ograniczenia musi być proporcjonalne do celu.
        Z drugiej strony, zbyt szerokie ograniczenia mogą naruszać konstytucję.
        """

        structure = self.analyzer.analyze_argumentative_structure(arg_text)

        assert isinstance(structure, dict)
        assert "arguments" in structure
        assert "argument_counts" in structure
        assert "total_argumentative_elements" in structure
        assert "argumentative_complexity" in structure

        arguments = structure["arguments"]
        assert isinstance(arguments, dict)

        # Check for expected argument types
        expected_types = [
            "premises",
            "conclusions",
            "counterarguments",
            "justifications",
        ]
        for arg_type in expected_types:
            assert arg_type in arguments
            assert isinstance(arguments[arg_type], list)

    def test_analyze_argumentative_structure_empty_text(self):
        """Test argumentative structure analysis with empty text."""
        structure = self.analyzer.analyze_argumentative_structure("")

        assert structure["total_argumentative_elements"] == 0
        assert structure["argumentative_complexity"] == 0

    def test_perform_comprehensive_semantic_analysis(self):
        """Test comprehensive semantic analysis."""
        analysis = self.analyzer.perform_comprehensive_semantic_analysis(
            self.civil_law_text
        )

        assert isinstance(analysis, dict)
        assert "semantic_fields" in analysis
        assert "semantic_relations" in analysis
        assert "conceptual_density" in analysis
        assert "legal_definitions" in analysis
        assert "argumentative_structure" in analysis
        assert "analysis_metadata" in analysis

        # Check metadata
        metadata = analysis["analysis_metadata"]
        assert "text_length" in metadata
        assert "word_count" in metadata
        assert metadata["text_length"] > 0
        assert metadata["word_count"] > 0

    def test_perform_comprehensive_semantic_analysis_empty_text(self):
        """Test comprehensive analysis with empty text."""
        analysis = self.analyzer.perform_comprehensive_semantic_analysis("")
        assert analysis == {}

    def test_extract_context(self):
        """Test context extraction."""
        text = "This is a test text for context extraction testing purposes."
        start = 10
        end = 14  # "test"

        context = self.analyzer._extract_context(text, start, end, window=5)

        assert isinstance(context, str)
        assert "test" in context
        assert len(context) <= len(text)


class TestSemanticField:
    """Test SemanticField dataclass."""

    def test_semantic_field_creation(self):
        """Test SemanticField instantiation."""
        field = SemanticField(
            field_name="test_field",
            terms=["term1", "term2"],
            weight=1.5,
            context_terms=["context1"],
        )

        assert field.field_name == "test_field"
        assert field.terms == ["term1", "term2"]
        assert field.weight == 1.5
        assert field.context_terms == ["context1"]

    def test_semantic_field_default_context(self):
        """Test SemanticField with default context terms."""
        field = SemanticField(field_name="test_field", terms=["term1"], weight=1.0)

        assert field.context_terms == []


class TestSemanticPatterns:
    """Test semantic pattern matching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LegalSemanticAnalyzer()

    def test_causal_pattern_matching(self):
        """Test causal relation pattern matching."""
        text = "Na skutek naruszenia umowy powstała szkoda."
        relations = self.analyzer.extract_semantic_relations(text)

        if relations.get("causal_relations"):
            causal = relations["causal_relations"][0]
            assert "skutek" in causal["relation_text"].lower()

    def test_modal_pattern_matching(self):
        """Test modal expression pattern matching."""
        text = "Właściciel może rozporządzać swoją rzeczą."
        relations = self.analyzer.extract_semantic_relations(text)

        if relations.get("modal_expressions"):
            modal = relations["modal_expressions"][0]
            assert "może" in modal["relation_text"].lower()

    def test_conditional_pattern_matching(self):
        """Test conditional relation pattern matching."""
        text = "Jeśli umowa zostanie naruszona, wówczas powstaje roszczenie."
        relations = self.analyzer.extract_semantic_relations(text)

        if relations.get("conditional_relations"):
            conditional = relations["conditional_relations"][0]
            assert "jeśli" in conditional["relation_text"].lower()

    def test_temporal_pattern_matching(self):
        """Test temporal relation pattern matching."""
        text = "Po zakończeniu postępowania wydaje się orzeczenie."
        relations = self.analyzer.extract_semantic_relations(text)

        if relations.get("temporal_relations"):
            temporal = relations["temporal_relations"][0]
            assert "po" in temporal["relation_text"].lower()
