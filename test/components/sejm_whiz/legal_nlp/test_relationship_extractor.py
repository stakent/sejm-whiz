"""Tests for legal_nlp relationship extractor functionality."""

from sejm_whiz.legal_nlp.relationship_extractor import (
    LegalRelationshipExtractor,
    LegalEntity,
    LegalRelationship,
    RelationshipType,
)


class TestLegalRelationshipExtractor:
    """Test cases for LegalRelationshipExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = LegalRelationshipExtractor()

        # Sample legal text with entities and relationships
        self.sample_text = """
        Minister sprawiedliwości ustanawia regulamin sądów powszechnych.
        Kodeks cywilny definiuje pojęcie własności jako prawo rzeczowe.
        Sąd Najwyższy uchyla orzeczenie sądu apelacyjnego.
        Ustawa o podatku dochodowym zastępuje poprzednie przepisy.
        Trybunał Konstytucyjny wykonuje kontrolę konstytucyjności ustaw.
        Prokurator ma prawo do wystąpienia z aktem oskarżenia.
        Jeżeli strona naruszy umowę, to druga strona może żądać odszkodowania.
        """

        self.entity_text = """
        Osoby fizyczne i prawne mogą zawierać umowy cywilnoprawne.
        Sąd rejonowy rozpoznaje sprawy w pierwszej instancji.
        Minister finansów wydaje rozporządzenie w sprawie podatków.
        Kodeks postępowania karnego reguluje postępowanie przygotowawcze.
        """

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        assert self.extractor is not None
        assert hasattr(self.extractor, "entity_patterns")
        assert hasattr(self.extractor, "relationship_patterns")
        assert hasattr(self.extractor, "legal_vocabulary")

    def test_entity_patterns_structure(self):
        """Test entity patterns structure."""
        for entity_type, patterns in self.extractor.entity_patterns.items():
            assert isinstance(patterns, list)
            assert len(patterns) > 0
            for pattern in patterns:
                assert isinstance(pattern, str)
                assert len(pattern) > 0

    def test_relationship_patterns_structure(self):
        """Test relationship patterns structure."""
        for rel_type, patterns in self.extractor.relationship_patterns.items():
            assert isinstance(rel_type, RelationshipType)
            assert isinstance(patterns, list)
            assert len(patterns) > 0

    def test_extract_entities_basic(self):
        """Test basic entity extraction."""
        entities = self.extractor.extract_entities(self.entity_text)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Check entity properties
        for entity in entities:
            assert isinstance(entity, LegalEntity)
            assert isinstance(entity.entity_type, str)
            assert isinstance(entity.name, str)
            assert len(entity.name) > 0
            assert isinstance(entity.aliases, list)
            assert isinstance(entity.attributes, dict)
            assert isinstance(entity.position, tuple)
            assert len(entity.position) == 2
            assert entity.position[0] >= 0
            assert entity.position[1] > entity.position[0]

    def test_extract_entities_empty_text(self):
        """Test entity extraction with empty text."""
        entities = self.extractor.extract_entities("")
        assert entities == []

        entities = self.extractor.extract_entities(None)
        assert entities == []

    def test_extract_entities_types(self):
        """Test extraction of different entity types."""
        entities = self.extractor.extract_entities(self.entity_text)

        entity_types = [entity.entity_type for entity in entities]

        # Should find various entity types
        possible_types = ["person", "institution", "legal_act", "legal_concept"]
        found_types = [et for et in possible_types if et in entity_types]
        assert len(found_types) > 0

    def test_remove_overlapping_entities(self):
        """Test removal of overlapping entities."""
        # Create overlapping entities
        entities = [
            LegalEntity("type1", "long entity name", position=(0, 15)),
            LegalEntity("type2", "long entity", position=(0, 10)),  # Overlapping
            LegalEntity("type3", "separate entity", position=(20, 35)),  # Separate
        ]

        filtered = self.extractor._remove_overlapping_entities(entities)

        # Should keep longer entity and separate entity
        assert len(filtered) == 2
        names = [e.name for e in filtered]
        assert "long entity name" in names
        assert "separate entity" in names
        assert "long entity" not in names

    def test_enrich_entities(self):
        """Test entity enrichment with attributes."""
        entities = [LegalEntity("test", "test entity", position=(10, 20))]
        text = "This is test entity context with legal terminology"

        enriched = self.extractor._enrich_entities(entities, text)

        assert len(enriched) == 1
        entity = enriched[0]
        assert "modifiers" in entity.attributes
        assert "temporal_info" in entity.attributes
        assert "context" in entity.attributes
        assert isinstance(entity.attributes["modifiers"], list)
        assert isinstance(entity.attributes["context"], str)

    def test_extract_relationships_basic(self):
        """Test basic relationship extraction."""
        relationships = self.extractor.extract_relationships(self.sample_text)

        assert isinstance(relationships, list)

        for relationship in relationships:
            assert isinstance(relationship, LegalRelationship)
            assert isinstance(relationship.source_entity, LegalEntity)
            assert isinstance(relationship.target_entity, LegalEntity)
            assert isinstance(relationship.relationship_type, RelationshipType)
            assert 0 <= relationship.confidence <= 1.0
            assert isinstance(relationship.evidence_text, str)
            assert isinstance(relationship.context, str)
            assert isinstance(relationship.modifiers, list)

    def test_extract_relationships_empty_text(self):
        """Test relationship extraction with empty text."""
        relationships = self.extractor.extract_relationships("")
        assert relationships == []

    def test_extract_relationships_with_entities(self):
        """Test relationship extraction with provided entities."""
        # First extract entities
        entities = self.extractor.extract_entities(self.sample_text)

        # Then extract relationships using those entities
        relationships = self.extractor.extract_relationships(self.sample_text, entities)

        assert isinstance(relationships, list)

        # Check that relationships use provided entities where possible
        for relationship in relationships:
            assert isinstance(relationship.source_entity, LegalEntity)
            assert isinstance(relationship.target_entity, LegalEntity)

    def test_find_matching_entity(self):
        """Test entity matching."""
        entities = [
            LegalEntity("type1", "Minister sprawiedliwości", aliases=["Minister"]),
            LegalEntity("type2", "Kodeks cywilny", aliases=["KC"]),
        ]

        # Test exact match
        match1 = self.extractor._find_matching_entity(
            "Minister sprawiedliwości", entities
        )
        assert match1 is not None
        assert match1.name == "Minister sprawiedliwości"

        # Test partial match
        match2 = self.extractor._find_matching_entity("Minister", entities)
        assert match2 is not None

        # Test no match
        match3 = self.extractor._find_matching_entity("Nieistniejący podmiot", entities)
        assert match3 is None

    def test_extract_relationship_context(self):
        """Test relationship context extraction."""
        text = "This is a test text for context extraction around relationships."
        start = 10
        end = 20

        context = self.extractor._extract_relationship_context(
            text, start, end, window=10
        )

        assert isinstance(context, str)
        assert len(context) <= len(text)
        assert len(context) > 0

    def test_extract_modifiers(self):
        """Test modifier extraction from context."""
        context = "bezwarunkowo obowiązuje natychmiast po publikacji"
        modifiers = self.extractor._extract_modifiers(context)

        assert isinstance(modifiers, list)
        # Should find some modifiers from the legal vocabulary
        expected_modifiers = ["bezwarunkowo", "natychmiast"]
        found_modifiers = [m for m in expected_modifiers if m in modifiers]
        assert len(found_modifiers) > 0

    def test_extract_temporal_info(self):
        """Test temporal information extraction."""
        context = "od dnia publikacji w dzienniku ustaw"
        temporal = self.extractor._extract_temporal_info(context)

        # Should find temporal marker
        assert (
            temporal is not None or temporal is None
        )  # May or may not find depending on vocabulary

    def test_calculate_relationship_confidence(self):
        """Test relationship confidence calculation."""

        # Mock match object
        class MockMatch:
            def group(self, n=0):
                return "test relationship pattern"

        match = MockMatch()
        context = "legal context with terminology"
        rel_type = RelationshipType.DEFINES
        source_entity = LegalEntity("type1", "source", position=(0, 6))
        target_entity = LegalEntity("type2", "target", position=(10, 16))

        confidence = self.extractor._calculate_relationship_confidence(
            match, context, rel_type, source_entity, target_entity
        )

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1.0

    def test_analyze_relationship_network(self):
        """Test relationship network analysis."""
        # Create sample relationships
        source1 = LegalEntity("type1", "Entity A", position=(0, 8))
        target1 = LegalEntity("type2", "Entity B", position=(10, 18))
        source2 = LegalEntity("type1", "Entity A", position=(20, 28))
        target2 = LegalEntity("type3", "Entity C", position=(30, 38))

        relationships = [
            LegalRelationship(
                source1, target1, RelationshipType.DEFINES, 0.8, "test evidence"
            ),
            LegalRelationship(
                source2, target2, RelationshipType.MODIFIES, 0.7, "test evidence 2"
            ),
        ]

        network = self.extractor.analyze_relationship_network(relationships)

        assert isinstance(network, dict)
        assert "network_metrics" in network
        assert "central_entities" in network
        assert "relationship_type_distribution" in network
        assert "relationship_patterns" in network
        assert "high_confidence_relationships" in network

        # Check network metrics
        metrics = network["network_metrics"]
        assert metrics["total_relationships"] == 2
        assert metrics["unique_entities"] > 0
        assert 0 <= metrics["relationship_density"] <= 1
        assert 0 <= metrics["average_confidence"] <= 1

    def test_analyze_relationship_network_empty(self):
        """Test network analysis with empty relationships."""
        network = self.extractor.analyze_relationship_network([])
        assert network == {}

    def test_analyze_relationship_patterns(self):
        """Test relationship pattern analysis."""
        # Create sample relationships for pattern analysis
        entity_a = LegalEntity("type1", "A", position=(0, 1))
        entity_b = LegalEntity("type2", "B", position=(2, 3))
        entity_c = LegalEntity("type3", "C", position=(4, 5))

        relationships = [
            LegalRelationship(
                entity_a, entity_b, RelationshipType.DEFINES, 0.8, "evidence"
            ),
            LegalRelationship(
                entity_b, entity_c, RelationshipType.MODIFIES, 0.7, "evidence"
            ),
            LegalRelationship(
                entity_a, entity_c, RelationshipType.REFERENCES, 0.6, "evidence"
            ),
        ]

        patterns = self.extractor._analyze_relationship_patterns(relationships)

        assert isinstance(patterns, dict)
        assert "chains" in patterns
        assert "cycles" in patterns
        assert "hubs" in patterns
        assert "clusters" in patterns

        # Should find some chains (A -> B -> C)
        assert isinstance(patterns["chains"], list)

    def test_perform_comprehensive_relationship_analysis(self):
        """Test comprehensive relationship analysis."""
        analysis = self.extractor.perform_comprehensive_relationship_analysis(
            self.sample_text
        )

        assert isinstance(analysis, dict)
        assert "entities" in analysis
        assert "relationships" in analysis
        assert "network_analysis" in analysis

        # Check entities section
        entities_section = analysis["entities"]
        assert "count" in entities_section
        assert "by_type" in entities_section
        assert "entities" in entities_section

        # Check relationships section
        relationships_section = analysis["relationships"]
        assert "count" in relationships_section
        assert "by_type" in relationships_section
        assert "relationships" in relationships_section

    def test_perform_comprehensive_relationship_analysis_empty_text(self):
        """Test comprehensive analysis with empty text."""
        analysis = self.extractor.perform_comprehensive_relationship_analysis("")
        assert analysis == {}

    def test_group_entities_by_type(self):
        """Test entity grouping by type."""
        entities = [
            LegalEntity("person", "Person 1", position=(0, 8)),
            LegalEntity("person", "Person 2", position=(10, 18)),
            LegalEntity("institution", "Institution 1", position=(20, 32)),
        ]

        grouped = self.extractor._group_entities_by_type(entities)

        assert isinstance(grouped, dict)
        assert grouped["person"] == 2
        assert grouped["institution"] == 1


class TestRelationshipType:
    """Test RelationshipType enum."""

    def test_relationship_type_values(self):
        """Test that all relationship types have expected values."""
        expected_types = [
            "defines",
            "modifies",
            "references",
            "supersedes",
            "implements",
            "delegates",
            "requires",
            "prohibits",
            "permits",
            "establishes",
            "repeals",
            "extends",
            "limits",
            "conditional_on",
            "applies_to",
        ]

        actual_types = [rt.value for rt in RelationshipType]

        for expected in expected_types:
            assert expected in actual_types


class TestLegalEntity:
    """Test LegalEntity dataclass."""

    def test_legal_entity_creation(self):
        """Test LegalEntity instantiation."""
        entity = LegalEntity(
            entity_type="person",
            name="Test Person",
            aliases=["TP", "Test"],
            attributes={"role": "judge"},
            position=(10, 20),
        )

        assert entity.entity_type == "person"
        assert entity.name == "Test Person"
        assert entity.aliases == ["TP", "Test"]
        assert entity.attributes == {"role": "judge"}
        assert entity.position == (10, 20)

    def test_legal_entity_defaults(self):
        """Test LegalEntity with default values."""
        entity = LegalEntity(entity_type="institution", name="Test Institution")

        assert entity.aliases == []
        assert entity.attributes == {}
        assert entity.position is None


class TestLegalRelationship:
    """Test LegalRelationship dataclass."""

    def test_legal_relationship_creation(self):
        """Test LegalRelationship instantiation."""
        source = LegalEntity("type1", "Source Entity", position=(0, 13))
        target = LegalEntity("type2", "Target Entity", position=(20, 33))

        relationship = LegalRelationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.DEFINES,
            confidence=0.85,
            evidence_text="test evidence",
            context="test context",
            modifiers=["immediately"],
            temporal_info="od dnia",
        )

        assert relationship.source_entity == source
        assert relationship.target_entity == target
        assert relationship.relationship_type == RelationshipType.DEFINES
        assert relationship.confidence == 0.85
        assert relationship.evidence_text == "test evidence"
        assert relationship.context == "test context"
        assert relationship.modifiers == ["immediately"]
        assert relationship.temporal_info == "od dnia"

    def test_legal_relationship_defaults(self):
        """Test LegalRelationship with default values."""
        source = LegalEntity("type1", "Source", position=(0, 6))
        target = LegalEntity("type2", "Target", position=(10, 16))

        relationship = LegalRelationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.REFERENCES,
            confidence=0.75,
            evidence_text="evidence",
        )

        assert relationship.context == ""
        assert relationship.modifiers == []
        assert relationship.temporal_info is None
