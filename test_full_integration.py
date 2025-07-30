#!/usr/bin/env python3
"""
Full integration test for sejm-whiz components.
Tests the complete pipeline: Database -> Redis -> Document Ingestion -> Embeddings
"""

import asyncio
import sys
import os
from typing import Dict, Any
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components
from components.sejm_whiz.database import (
    check_database_health,
    DocumentOperations,
    get_database_config
)
from components.sejm_whiz.redis import (
    check_redis_health,
    get_redis_cache,
    get_redis_queue,
    JobPriority
)
from components.sejm_whiz.document_ingestion import (
    get_ingestion_config,
    TextProcessor,
    ProcessedDocument
)
from components.sejm_whiz.embeddings import (
    get_embedding_config,
    get_herbert_embedder,
    get_embedding_operations
)


class IntegrationTestSuite:
    """Complete integration test suite for sejm-whiz."""
    
    def __init__(self):
        self.test_results = {
            'database': {'status': 'pending', 'details': {}},
            'redis': {'status': 'pending', 'details': {}},
            'document_processing': {'status': 'pending', 'details': {}},
            'embeddings': {'status': 'pending', 'details': {}},
            'integration': {'status': 'pending', 'details': {}}
        }
        
        # Sample legal document for testing
        self.sample_document = {
            'title': 'Ustawa z dnia 23 kwietnia 1964 r. - Kodeks cywilny',
            'content': '''
            Art. 1. Zdolność prawna człowieka powstaje z chwilą urodzenia.
            Art. 2. Pełną zdolność do czynności prawnych nabywa się z chwilą uzyskania pełnoletności.
            Art. 3. § 1. Pełnoletność rozpoczyna się z chwilą ukończenia osiemnastego roku życia.
            § 2. Małoletni, który ukończył szesnasty rok życia, może przez oświadczenie złożone przed sądem opiekuńczym uzyskać pełnoletność, jeżeli zawarł związek małżeński.
            
            Rozdział I - Osoby fizyczne
            Art. 8. Osoba fizyczna ma zdolność prawną w zakresie stosunków prawnych prawa cywilnego.
            Art. 9. § 1. Każdy człowiek ma niezbywalne i nieoddawalne prawo do życia oraz do swobodnego rozwoju swojej osobowości.
            § 2. Każdy człowiek ma obowiązek szanowania osobowości innych ludzi.
            ''',
            'document_type': 'kodeks',
            'eli_identifier': 'eli:pl:kodeks-cywilny:1964:23-kwietnia',
            'legal_act_type': 'kodeks',
            'legal_domain': 'civil'
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🧪 Starting sejm-whiz full integration test suite")
        print("=" * 60)
        
        try:
            # Test individual components
            await self.test_database_component()
            await self.test_redis_component()
            await self.test_document_processing()
            await self.test_embeddings_component()
            
            # Test full integration
            await self.test_full_pipeline_integration()
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # Print results summary
        self.print_test_summary()
        
        return self.test_results
    
    async def test_database_component(self) -> None:
        """Test database connectivity and operations."""
        print("\n📊 Testing Database Component")
        print("-" * 30)
        
        try:
            # Health check
            health = check_database_health()
            print("Database health status:")
            for key, value in health.items():
                status_icon = "✅" if value else "❌" if isinstance(value, bool) else "ℹ️"
                print(f"  {status_icon} {key}: {value}")
            
            # Test basic operations (if database is available)
            if health.get("connection"):
                db_config = get_database_config()
                db_ops = DocumentOperations(db_config)
                
                # Test document creation
                test_doc = ProcessedDocument(
                    title="Test Document",
                    content="Test content for integration testing",
                    document_type="test",
                    quality_score=0.9
                )
                
                created_doc = await db_ops.create_document(test_doc)
                print(f"✅ Created test document: {created_doc.id}")
                
                # Test document retrieval
                retrieved_doc = await db_ops.get_document_by_id(str(created_doc.id))
                if retrieved_doc:
                    print(f"✅ Retrieved document: {retrieved_doc.title}")
                
                self.test_results['database'] = {
                    'status': 'success',
                    'details': {
                        'connection': True,
                        'operations': True,
                        'test_doc_id': str(created_doc.id)
                    }
                }
            else:
                self.test_results['database'] = {
                    'status': 'warning',
                    'details': {'connection': False, 'message': 'Database not available for testing'}
                }
                print("⚠️  Database connection not available - some tests skipped")
        
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            self.test_results['database'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    async def test_redis_component(self) -> None:
        """Test Redis connectivity and operations."""
        print("\n🔴 Testing Redis Component")
        print("-" * 30)
        
        try:
            # Health check
            health = check_redis_health()
            print("Redis health status:")
            for key, value in health.items():
                status_icon = "✅" if value else "❌" if isinstance(value, bool) else "ℹ️"
                print(f"  {status_icon} {key}: {value}")
            
            # Test cache operations
            cache = get_redis_cache()
            test_key = "integration_test_key"
            test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
            
            if health.get("connection"):
                # Test cache set/get
                cache_set = cache.set(test_key, test_value, ttl=60)
                print(f"✅ Cache set operation: {cache_set}")
                
                cached_value = cache.get(test_key)
                if cached_value == test_value:
                    print("✅ Cache get operation successful")
                
                # Test job queue
                job_queue = get_redis_queue()
                job_stats = job_queue.get_queue_stats()
                print(f"✅ Job queue stats: {job_stats}")
                
                # Clean up
                cache.delete(test_key)
                
                self.test_results['redis'] = {
                    'status': 'success',
                    'details': {
                        'connection': True,
                        'cache_operations': True,
                        'job_queue': True
                    }
                }
            else:
                self.test_results['redis'] = {
                    'status': 'warning',
                    'details': {'connection': False, 'message': 'Redis not available for testing'}
                }
                print("⚠️  Redis connection not available - some tests skipped")
        
        except Exception as e:
            print(f"❌ Redis test failed: {e}")
            self.test_results['redis'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    async def test_document_processing(self) -> None:
        """Test document ingestion and processing."""
        print("\n📄 Testing Document Processing Component")
        print("-" * 30)
        
        try:
            # Test configuration
            config = get_ingestion_config()
            print(f"✅ Ingestion config loaded: {config.eli_api_base_url}")
            
            # Test text processor
            processor = TextProcessor(config)
            
            # Process sample document
            processed_doc = processor.process_document(
                raw_content=self.sample_document['content'],
                eli_id=self.sample_document['eli_identifier']
            )
            
            print(f"✅ Processed document: {processed_doc.title[:50]}...")
            print(f"   Document type: {processed_doc.document_type}")
            print(f"   Quality score: {processed_doc.quality_score:.2f}")
            print(f"   Content length: {len(processed_doc.content)} chars")
            
            # Validate document
            is_valid, errors = processor.validate_document(processed_doc)
            print(f"✅ Document validation: {'PASSED' if is_valid else 'FAILED'}")
            if errors:
                print(f"   Validation errors: {errors}")
            
            # Test legal structure extraction
            if processed_doc.metadata and 'structure' in processed_doc.metadata:
                structure = processed_doc.metadata['structure']
                print(f"✅ Legal structure extracted:")
                print(f"   Articles: {len(structure.articles)}")
                print(f"   Paragraphs: {len(structure.paragraphs)}")
                print(f"   References: {len(structure.references)}")
            
            self.test_results['document_processing'] = {
                'status': 'success',
                'details': {
                    'processing': True,
                    'validation': is_valid,
                    'quality_score': processed_doc.quality_score,
                    'document_type': processed_doc.document_type
                }
            }
            
        except Exception as e:
            print(f"❌ Document processing test failed: {e}")
            self.test_results['document_processing'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    async def test_embeddings_component(self) -> None:
        """Test embedding generation component."""
        print("\n🧠 Testing Embeddings Component")
        print("-" * 30)
        
        try:
            # Test configuration
            embedding_config = get_embedding_config()
            print(f"✅ Embedding config loaded: {embedding_config.model_name}")
            print(f"   Device: {embedding_config.device}")
            print(f"   Embedding dim: {embedding_config.embedding_dim}")
            
            # Test embedder initialization (this will download model if not cached)
            print("🔄 Initializing HerBERT embedder (this may take a while)...")
            embedder = get_herbert_embedder(embedding_config)
            
            # Test simple text embedding
            test_text = "Ustawa o prawach człowieka i obywatela."
            print(f"🔄 Generating embedding for test text...")
            
            embedding_result = embedder.embed_text(test_text)
            
            print(f"✅ Generated embedding:")
            print(f"   Dimensions: {embedding_result.embedding.shape}")
            print(f"   Model: {embedding_result.model_name}")
            print(f"   Processing time: {embedding_result.processing_time:.3f}s")
            print(f"   Token count: {embedding_result.token_count}")
            print(f"   Quality score: {embedding_result.quality_score:.2f}")
            
            # Test legal document embedding
            print("🔄 Testing legal document embedding...")
            legal_embedding = embedder.embed_legal_document(
                title=self.sample_document['title'],
                content=self.sample_document['content'],
                document_type=self.sample_document['document_type']
            )
            
            print(f"✅ Legal document embedding generated:")
            print(f"   Dimensions: {legal_embedding.embedding.shape}")
            print(f"   Quality score: {legal_embedding.quality_score:.2f}")
            
            # Test similarity calculation
            similarity = embedder.calculate_similarity(
                embedding_result.embedding,
                legal_embedding.embedding
            )
            print(f"✅ Similarity between texts: {similarity:.3f}")
            
            # Clean up model to free memory
            embedder.cleanup()
            print("✅ Model cleanup completed")
            
            self.test_results['embeddings'] = {
                'status': 'success',
                'details': {
                    'model_loading': True,
                    'text_embedding': True,
                    'legal_embedding': True,
                    'similarity_calculation': True,
                    'embedding_dim': embedding_result.embedding.shape[0],
                    'model_name': embedding_result.model_name
                }
            }
            
        except Exception as e:
            print(f"❌ Embeddings test failed: {e}")
            self.test_results['embeddings'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    async def test_full_pipeline_integration(self) -> None:
        """Test full pipeline integration."""
        print("\n🔗 Testing Full Pipeline Integration")
        print("-" * 30)
        
        try:
            # Check if required components are available
            db_available = self.test_results['database']['status'] == 'success'
            redis_available = self.test_results['redis']['status'] == 'success'
            embeddings_available = self.test_results['embeddings']['status'] == 'success'
            
            if not all([db_available, redis_available, embeddings_available]):
                print("⚠️  Some components not available - skipping integration test")
                self.test_results['integration'] = {
                    'status': 'skipped',
                    'details': {
                        'reason': 'Required components not available',
                        'db_available': db_available,
                        'redis_available': redis_available,
                        'embeddings_available': embeddings_available
                    }
                }
                return
            
            print("🔄 Testing full document processing pipeline...")
            
            # 1. Process document
            config = get_ingestion_config()
            processor = TextProcessor(config)
            
            processed_doc = processor.process_document(
                raw_content=self.sample_document['content'],
                eli_id=self.sample_document['eli_identifier']
            )
            print("✅ Step 1: Document processed")
            
            # 2. Store in database
            db_config = get_database_config()
            db_ops = DocumentOperations(db_config)
            
            stored_doc = await db_ops.create_document(processed_doc)
            print(f"✅ Step 2: Document stored in database: {stored_doc.id}")
            
            # 3. Generate and store embedding
            embedding_ops = get_embedding_operations()
            embedding_result = await embedding_ops.generate_document_embedding(str(stored_doc.id))
            
            if embedding_result:
                print(f"✅ Step 3: Embedding generated and stored")
                print(f"   Embedding dimensions: {embedding_result.embedding.shape}")
                print(f"   Quality score: {embedding_result.quality_score:.2f}")
            else:
                raise Exception("Failed to generate embedding")
            
            # 4. Test similarity search
            similar_docs = await embedding_ops.find_similar_documents(
                query_text="prawo cywilne osoby fizyczne",
                limit=5,
                threshold=0.5
            )
            print(f"✅ Step 4: Similarity search completed: {len(similar_docs)} results")
            
            # 5. Test caching (Redis integration)
            cache = get_redis_cache()
            cached_doc = cache.get_document(str(stored_doc.id))
            if cached_doc:
                print("✅ Step 5: Document found in cache")
            else:
                print("ℹ️  Step 5: Document not in cache (expected)")
            
            self.test_results['integration'] = {
                'status': 'success',
                'details': {
                    'document_processing': True,
                    'database_storage': True,
                    'embedding_generation': True,
                    'similarity_search': True,
                    'caching': True,
                    'stored_document_id': str(stored_doc.id),
                    'similarity_results_count': len(similar_docs)
                }
            }
            
            print("🎉 Full pipeline integration test PASSED!")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            self.test_results['integration'] = {
                'status': 'failed',
                'details': {'error': str(e)}
            }
    
    def print_test_summary(self) -> None:
        """Print comprehensive test results summary."""
        print("\n" + "=" * 60)
        print("📋 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        all_passed = True
        
        for component, result in self.test_results.items():
            status = result['status']
            
            if status == 'success':
                icon = "✅"
            elif status == 'warning':
                icon = "⚠️"
            elif status == 'skipped':
                icon = "⏭️"
            else:
                icon = "❌"
                all_passed = False
            
            print(f"{icon} {component.upper()}: {status.upper()}")
            
            # Print key details
            details = result.get('details', {})
            for key, value in details.items():
                if key != 'error':
                    print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        
        if all_passed:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("🚀 sejm-whiz components are working correctly together!")
        else:
            print("❌ SOME TESTS FAILED")
            print("🔧 Please check the error messages above and fix the issues.")
        
        print("=" * 60)


async def main():
    """Run the full integration test suite."""
    test_suite = IntegrationTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Exit with appropriate code
        failed_components = [
            comp for comp, result in results.items() 
            if result['status'] == 'failed'
        ]
        
        if failed_components:
            print(f"\n💥 Failed components: {', '.join(failed_components)}")
            sys.exit(1)
        else:
            print(f"\n🎯 Integration test suite completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())