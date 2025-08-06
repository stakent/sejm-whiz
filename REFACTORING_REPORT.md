# Code Readability Refactoring Report

*Generated: 2025-08-06*

## Executive Summary

Analysis of the sejm-whiz codebase identified **20 high-complexity functions** (C-D grade) requiring refactoring. Items are sorted by complexity score (highest first) to prioritize the most critical refactoring needs.

## Refactoring Items by Complexity (Highest Priority First)

### 1. 🔴 DocumentIndexer.batch_index_documents

**File:** `components/sejm_whiz/semantic_search/indexer.py:157`
**Complexity:** D (21) - **CRITICAL**
**Issue:** Massive method handling batch processing, error handling, progress tracking, and result aggregation

**Refactoring Strategy:**

- Extract `_process_single_document_batch()` method
- Extract `_handle_batch_errors()` method
- Extract `_update_progress_tracking()` method
- Extract `_aggregate_batch_results()` method

### 2. 🟠 HerBERTEmbedder.\_initialize_model

**File:** `components/sejm_whiz/embeddings/herbert_embedder.py:62`
**Complexity:** C (16)
**Issue:** Model initialization with device detection, memory optimization, and configuration

**Refactoring Actions:**

- Extract `_detect_optimal_device()` method
- Extract `_configure_memory_settings()` method
- Extract `_load_tokenizer_and_model()` method

### 3. 🟠 EliApiClient.batch_get_documents

**File:** `components/sejm_whiz/eli_api/client.py:412`
**Complexity:** C (16)
**Issue:** Complex batch processing with concurrency control and rate limiting

**Refactoring Actions:**

- Extract `_create_document_batches()` method
- Extract `_process_batch_with_semaphore()` method
- Extract `_handle_batch_failures()` method

### 4. 🟠 SVMLegalClassifier.extract_features

**File:** `components/sejm_whiz/prediction_models/classification.py:383`
**Complexity:** C (15)
**Issue:** Feature extraction with multiple vectorization strategies

**Refactoring Actions:**

- Extract `_extract_tfidf_features()` method
- Extract `_extract_legal_specific_features()` method
- Extract `_combine_feature_vectors()` method

### 5. 🟡 BlendingEnsemble.predict

**File:** `components/sejm_whiz/prediction_models/ensemble.py:347`
**Complexity:** C (14)
**Issue:** Complex ensemble prediction logic with multiple model blending

**Refactoring Actions:**

- Extract `_validate_prediction_inputs()` method
- Extract `_blend_model_predictions()` method
- Extract `_apply_blending_weights()` method

### 6. 🟡 EliApiClient.search_documents

**File:** `components/sejm_whiz/eli_api/client.py:171`
**Complexity:** C (14)
**Issue:** Document search with complex query building and pagination

**Refactoring Actions:**

- Extract `_build_search_query()` method
- Extract `_paginate_search_results()` method
- Extract `_validate_search_params()` method

### 7. 🟡 SecureDocumentOperations.search_documents

**File:** `components/sejm_whiz/database/operations_secure.py:278`
**Complexity:** C (14)
**Issue:** Secure document search with access controls and sanitization

**Refactoring Actions:**

- Extract `_sanitize_search_inputs()` method
- Extract `_build_secure_query()` method
- Extract `_apply_access_controls()` method

### 8. 🟡 CosineDistancePredictor.\_aggregate_similar_predictions

**File:** `components/sejm_whiz/prediction_models/similarity.py:173`
**Complexity:** C (13)
**Issue:** Complex similarity prediction aggregation logic

**Refactoring Actions:**

- Extract `_calculate_similarity_weights()` method
- Extract `_filter_relevant_predictions()` method
- Extract `_compute_final_prediction()` method

### 9. 🟡 EuclideanDistancePredictor.\_aggregate_similar_predictions

**File:** `components/sejm_whiz/prediction_models/similarity.py:289`
**Complexity:** C (13)
**Issue:** Euclidean distance-based prediction aggregation

**Refactoring Actions:**

- Extract `_calculate_distance_weights()` method
- Extract `_normalize_predictions()` method
- Extract `_aggregate_weighted_predictions()` method

### 10. 🟡 StackingEnsemble.predict

**File:** `components/sejm_whiz/prediction_models/ensemble.py:223`
**Complexity:** C (13)
**Issue:** Stacking ensemble prediction with meta-model

**Refactoring Actions:**

- Extract `_generate_base_predictions()` method
- Extract `_prepare_meta_features()` method
- Extract `_predict_with_meta_model()` method

### 11. 🟡 PolishTokenizer.tokenize_sentences

**File:** `components/sejm_whiz/text_processing/tokenizer.py:84`
**Complexity:** C (13)
**Issue:** Complex sentence tokenization with legal document specifics

**Refactoring Actions:**

- Extract `_preprocess_for_tokenization()` method
- Extract `_apply_legal_sentence_rules()` method
- Extract `_post_process_sentences()` method

### 12. 🟡 TextProcessor.validate_document

**File:** `components/sejm_whiz/document_ingestion/text_processor.py:476`
**Complexity:** C (13)
**Issue:** Document validation with quality checks

**Refactoring Actions:**

- Extract `_validate_document_structure()` method
- Extract `_check_content_quality()` method
- Extract `_assess_legal_validity()` method

### 13. 🟡 VectorSimilaritySearch.find_similar_documents

**File:** `components/sejm_whiz/vector_db/embeddings.py:37`
**Complexity:** C (12)
**Issue:** Vector similarity search with filtering

**Refactoring Actions:**

- Extract `_prepare_search_vector()` method
- Extract `_apply_similarity_filters()` method
- Extract `_rank_search_results()` method

### 14. 🟡 SecureVectorOperations.find_similar_documents

**File:** `components/sejm_whiz/database/operations_secure.py:352`
**Complexity:** C (12)
**Issue:** Secure vector operations with access controls

**Refactoring Actions:**

- Extract `_validate_vector_access()` method
- Extract `_secure_vector_search()` method
- Extract `_filter_accessible_results()` method

### 15. 🟡 PolishTokenizer.tokenize_words

**File:** `components/sejm_whiz/text_processing/tokenizer.py:60`
**Complexity:** C (12)
**Issue:** Word tokenization with Polish language specifics

**Refactoring Actions:**

- Extract `_handle_polish_morphology()` method
- Extract `_process_legal_terms()` method
- Extract `_clean_tokenized_words()` method

### 16. 🟡 CrossRegisterMatcher.\_find_semantic_matches

**File:** `components/sejm_whiz/semantic_search/cross_register.py:219`
**Complexity:** C (12)
**Issue:** Cross-register semantic matching

**Refactoring Actions:**

- Extract `_prepare_semantic_vectors()` method
- Extract `_calculate_cross_similarities()` method
- Extract `_filter_semantic_matches()` method

### 17-20. Lower Priority C-Grade Functions (C 11)

#### 17. VectorSimilaritySearch.find_similar_by_document_id

**File:** `components/sejm_whiz/vector_db/embeddings.py:119` | **Complexity:** C (11)

#### 18. TemporalSimilarityPredictor.\_aggregate_temporal_predictions

**File:** `components/sejm_whiz/prediction_models/similarity.py:529` | **Complexity:** C (11)

#### 19. LegalNLPAnalyzer.detect_amendments

**File:** `components/sejm_whiz/legal_nlp/core.py:315` | **Complexity:** C (11)

#### 20. LegalRelationshipExtractor.extract_relationships

**File:** `components/sejm_whiz/legal_nlp/relationship_extractor.py:338` | **Complexity:** C (11)

## Type Annotation Critical Issues (17 items)

### Priority 1 - Incompatible Type Assignments:

1. **legal_parser.py:41** - `list[LegalProvision]` assigned None
1. **legal_parser.py:248,319** - Missing type annotations for variables
1. **semantic_analyzer.py:15,236,450** - Missing type hints

### Priority 2 - Exception Constructor Issues:

4. **sejm_api/exceptions.py:329-341** - 8 incompatible argument types

## Implementation Timeline

**Week 1 (Critical):** Items 1-4 (Complexity 21, 16, 16, 15)
**Week 2 (High):** Items 5-8 (Complexity 14, 14, 14, 13)
**Week 3 (Medium):** Items 9-12 (Complexity 13, 13, 13, 13)
**Week 4 (Low):** Items 13-20 (Complexity 12, 11)

**Estimated Total Effort:** 88-120 hours (11-15 developer days)
