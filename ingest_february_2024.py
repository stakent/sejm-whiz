#!/usr/bin/env python3
"""
February 2024 Document Ingestion - Continue Testing Baremetal PostgreSQL
Ingest documents specifically from February 2024 to further verify baremetal deployment works.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, UTC
from pathlib import Path

# Set up environment for baremetal deployment
os.environ["DEPLOYMENT_ENV"] = "baremetal"

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "components"))
sys.path.insert(0, str(project_root / "bases"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('february_2024_ingestion.log')
    ]
)

logger = logging.getLogger(__name__)


async def get_current_document_count():
    """Get current document count for progress tracking."""
    try:
        from sejm_whiz.database.connection import get_db_session
        from sqlalchemy import text
        
        with get_db_session() as session:
            result = session.execute(text("SELECT COUNT(*) FROM legal_documents"))
            return result.scalar() or 0
    except Exception as e:
        logger.error(f"Failed to get document count: {e}")
        return 0


async def create_february_2024_documents():
    """Create February 2024 documents as synthetic data."""
    logger.info("📅 Creating February 2024 documents...")
    
    try:
        from sejm_whiz.database.connection import get_db_session
        from sqlalchemy import text
        import uuid
        from datetime import datetime, UTC, timedelta
        
        # February 2024 document templates
        february_2024_docs = [
            {
                "title": "Ustawa z dnia 5 lutego 2024 r. o cyberbezpieczeństwie infrastruktury krytycznej",
                "type": "ustawa_february_2024",
                "content": """
                USTAWA z dnia 5 lutego 2024 r. o cyberbezpieczeństwie infrastruktury krytycznej
                
                Rozdział I - Przepisy ogólne
                
                Art. 1. Ustawa określa zasady ochrony cyberbezpieczeństwa infrastruktury krytycznej w Rzeczypospolitej Polskiej.
                
                Art. 2. Ilekroć w ustawie jest mowa o:
                1) infrastrukturze krytycznej - oznacza to systemy informatyczne niezbędne do funkcjonowania państwa;
                2) incydencie cybernetycznym - oznacza to zdarzenie naruszające bezpieczeństwo systemów informatycznych.
                
                Rozdział II - Obowiązki operatorów
                
                Art. 3. Operatorzy infrastruktury krytycznej zobowiązani są do:
                1) wdrożenia środków technicznych i organizacyjnych zabezpieczeń;
                2) zgłaszania incydentów cybernetycznych;
                3) prowadzenia audytów bezpieczeństwa.
                
                [Dokument February 2024 - Test baremetal PostgreSQL]
                """
            },
            {
                "title": "Rozporządzenie Ministra Zdrowia z dnia 10 lutego 2024 r. w sprawie telemedycyny",
                "type": "rozporządzenie_february_2024", 
                "content": """
                ROZPORZĄDZENIE MINISTRA ZDROWIA z dnia 10 lutego 2024 r. w sprawie świadczeń telemedycznych
                
                § 1. Rozporządzenie określa warunki udzielania świadczeń zdrowotnych przy wykorzystaniu systemów teleinformatycznych.
                
                § 2. Świadczenia telemedyczne obejmują:
                1) konsultacje lekarskie na odległość;
                2) monitoring stanu zdrowia pacjentów;
                3) edukację zdrowotną;
                4) rehabilitację zdalną.
                
                § 3. Świadczenia mogą być udzielane przez:
                1) lekarzy posiadających odpowiednie kwalifikacje;
                2) pielęgniarki w zakresie określonym odrębnymi przepisami;
                3) fizjoterapeutów w ramach rehabilitacji.
                
                [Dokument February 2024 - Test baremetal PostgreSQL]
                """
            },
            {
                "title": "Posiedzenie Sejmu RP z dnia 14 lutego 2024 r. - 17. posiedzenie X kadencji",
                "type": "sejm_proceeding_february_2024",
                "content": """
                POSIEDZENIE SEJMU RZECZYPOSPOLITEJ POLSKIEJ
                17. posiedzenie Sejmu Rzeczypospolitej Polskiej X kadencji
                w dniu 14 lutego 2024 r.
                
                Porządek obrad:
                1. Pierwsze czytanie projektu ustawy o cyberbezpieczeństwie
                2. Drugie czytanie projektu ustawy o telemedycynie  
                3. Debata nad stanem służby zdrowia
                4. Głosowanie nad projektami ustaw
                
                MARSZAŁEK: Witam państwa na 17. posiedzeniu Sejmu X kadencji.
                
                Przystępujemy do pierwszego punktu porządku obrad - projektu ustawy o cyberbezpieczeństwie infrastruktury krytycznej.
                
                Głos zabiera poseł sprawozdawca...
                
                [Stenogram February 2024 - Test baremetal PostgreSQL]
                """
            },
            {
                "title": "Ustawa z dnia 20 lutego 2024 r. o wsparciu przedsiębiorczości kobiet",
                "type": "ustawa_february_2024",
                "content": """
                USTAWA z dnia 20 lutego 2024 r. o wsparciu przedsiębiorczości kobiet
                
                Art. 1. Ustawa określa formy wsparcia dla kobiet prowadzących działalność gospodarczą.
                
                Art. 2. Wsparcie obejmuje:
                1) preferencyjne kredyty na rozpoczęcie działalności;
                2) doradztwo biznesowe;
                3) szkolenia z zakresu przedsiębiorczości;
                4) mentoring biznesowy.
                
                Art. 3. Ze wsparcia mogą skorzystać kobiety:
                1) rozpoczynające działalność gospodarczą;
                2) prowadzące działalność nie dłużej niż 24 miesiące;
                3) rozwijające innowacyjne projekty biznesowe.
                
                Art. 4. Środki na wsparcie pochodzą z budżetu państwa oraz funduszy europejskich.
                
                [Dokument February 2024 - Test baremetal PostgreSQL]
                """
            },
            {
                "title": "Rozporządzenie Rady Ministrów z dnia 25 lutego 2024 r. w sprawie energetyki odnawialnej",
                "type": "rozporządzenie_february_2024",
                "content": """
                ROZPORZĄDZENIE RADY MINISTRÓW z dnia 25 lutego 2024 r. w sprawie rozwoju energetyki odnawialnej
                
                § 1. Rozporządzenie określa cele i kierunki rozwoju odnawialnych źródeł energii.
                
                § 2. Do odnawialnych źródeł energii zalicza się:
                1) energię słoneczną;
                2) energię wiatrową;
                3) energię wodną;
                4) energię geotermalną;
                5) biomasę i biogaz.
                
                § 3. Cele rozwoju energetyki odnawialnej:
                1) osiągnięcie 30% udziału OZE w finalnym zużyciu energii do 2030 r.;
                2) redukcja emisji CO2 o 40% do 2030 r.;
                3) zwiększenie bezpieczeństwa energetycznego.
                
                § 4. Wsparcie rozwoju OZE obejmuje system aukcyjny i feed-in tariff.
                
                [Dokument February 2024 - Test baremetal PostgreSQL]
                """
            },
            {
                "title": "Posiedzenie Sejmu RP z dnia 28 lutego 2024 r. - 18. posiedzenie X kadencji",
                "type": "sejm_proceeding_february_2024",
                "content": """
                POSIEDZENIE SEJMU RZECZYPOSPOLITEJ POLSKIEJ
                18. posiedzenie Sejmu Rzeczypospolitej Polskiej X kadencji
                w dniu 28 lutego 2024 r.
                
                Porządek obrad:
                1. Trzecie czytanie projektu ustawy o wsparciu przedsiębiorczości kobiet
                2. Pierwsze czytanie projektu ustawy o energetyce odnawialnej
                3. Sprawozdanie z wykonania budżetu za I kwartał 2024
                4. Informacja o sytuacji na Ukrainie
                
                MARSZAŁEK: Dzień dobry państwu. Otwieramy ostatnie w tym miesiącu posiedzenie Sejmu.
                
                Przed przystąpieniem do obrad pragnę poinformować, że wpłynęły nowe projekty ustaw...
                
                [Stenogram February 2024 - Test baremetal PostgreSQL]
                """
            }
        ]
        
        created_count = 0
        
        for i, doc_data in enumerate(february_2024_docs):
            try:
                # Create specific February 2024 dates
                feb_day = (i * 4) + 5  # Days 5, 9, 13, 17, 21, 25  
                if feb_day > 28:  # February has 29 days in 2024 (leap year)
                    feb_day = 28
                published_date = datetime(2024, 2, feb_day, tzinfo=UTC)
                
                with get_db_session() as session:
                    doc_id = str(uuid.uuid4())
                    
                    session.execute(text("""
                        INSERT INTO legal_documents 
                        (id, title, content, document_type, created_at, updated_at, published_at)
                        VALUES (:id, :title, :content, :doc_type, :created_at, :updated_at, :published_at)
                    """), {
                        "id": doc_id,
                        "title": doc_data["title"],
                        "content": doc_data["content"],
                        "doc_type": doc_data["type"],
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC),
                        "published_at": published_date
                    })
                    session.commit()
                
                created_count += 1
                logger.info(f"✅ Created document {i+1}/6: {doc_data['title'][:50]}...")
                
                # Small delay between inserts
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Failed to create document {i+1}: {e}")
                continue
        
        logger.info(f"📊 Successfully created {created_count}/6 February 2024 documents")
        return created_count
        
    except Exception as e:
        logger.error(f"Document creation failed: {e}")
        raise


async def verify_february_documents():
    """Verify the February 2024 documents were stored correctly."""
    logger.info("🔍 Verifying February 2024 documents in database...")
    
    try:
        from sejm_whiz.database.connection import get_db_session
        from sqlalchemy import text
        
        with get_db_session() as session:
            # Count February 2024 documents
            result = session.execute(text("""
                SELECT document_type, COUNT(*) as count 
                FROM legal_documents 
                WHERE document_type LIKE :pattern
                GROUP BY document_type
                ORDER BY count DESC
            """), {"pattern": "%february_2024%"})
            
            february_docs = dict(result.fetchall())
            
            if february_docs:
                logger.info("📋 February 2024 documents by type:")
                total = 0
                for doc_type, count in february_docs.items():
                    logger.info(f"   {doc_type}: {count}")
                    total += count
                
                logger.info(f"✅ Total February 2024 documents: {total}")
                
                # Get sample document details
                result = session.execute(text("""
                    SELECT title, published_at 
                    FROM legal_documents 
                    WHERE document_type LIKE :pattern
                    ORDER BY published_at
                    LIMIT 4
                """), {"pattern": "%february_2024%"})
                
                logger.info("📄 Sample documents:")
                for title, pub_date in result:
                    logger.info(f"   {pub_date.strftime('%Y-%m-%d')}: {title[:60]}...")
                
                return total
            else:
                logger.warning("⚠️ No February 2024 documents found")
                return 0
                
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return 0


async def verify_all_2024_documents():
    """Verify both January and February 2024 documents."""
    logger.info("🔍 Verifying all 2024 test documents...")
    
    try:
        from sejm_whiz.database.connection import get_db_session
        from sqlalchemy import text
        
        with get_db_session() as session:
            # Count all 2024 test documents
            result = session.execute(text("""
                SELECT 
                    CASE 
                        WHEN document_type LIKE :jan_pattern THEN 'January 2024'
                        WHEN document_type LIKE :feb_pattern THEN 'February 2024'
                        ELSE 'Other'
                    END as month,
                    COUNT(*) as count
                FROM legal_documents 
                WHERE document_type LIKE :jan_pattern OR document_type LIKE :feb_pattern
                GROUP BY month
                ORDER BY month
            """), {
                "jan_pattern": "%january_2024%",
                "feb_pattern": "%february_2024%"
            })
            
            month_docs = dict(result.fetchall())
            
            logger.info("📊 Summary of 2024 test documents:")
            total_2024 = 0
            for month, count in month_docs.items():
                logger.info(f"   {month}: {count} documents")
                total_2024 += count
                
            logger.info(f"🎯 Total 2024 test documents: {total_2024}")
            return total_2024
                
    except Exception as e:
        logger.error(f"2024 verification failed: {e}")
        return 0


async def main():
    """Main execution function for February 2024 ingestion."""
    logger.info("🚀 Starting February 2024 Document Ingestion")
    logger.info("🎯 Continue testing baremetal PostgreSQL with new document ingestion")
    logger.info("=" * 70)
    
    start_time = datetime.now(UTC)
    initial_count = await get_current_document_count()
    logger.info(f"📊 Initial document count: {initial_count}")
    
    try:
        # Create February 2024 documents
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 1: CREATING FEBRUARY 2024 DOCUMENTS")
        logger.info("=" * 50)
        
        created_count = await create_february_2024_documents()
        
        # Verify February documents
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 2: VERIFYING FEBRUARY DOCUMENT STORAGE")
        logger.info("=" * 50)
        
        verified_count = await verify_february_documents()
        
        # Verify all 2024 documents
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 3: VERIFYING ALL 2024 TEST DOCUMENTS")
        logger.info("=" * 50)
        
        total_2024_count = await verify_all_2024_documents()
        
        # Final summary
        end_time = datetime.now(UTC)
        final_count = await get_current_document_count()
        duration = (end_time - start_time).total_seconds()
        added_count = final_count - initial_count
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 FEBRUARY 2024 INGESTION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"⏱️  Total time: {duration:.2f} seconds")
        logger.info(f"📈 Documents added: {added_count}")
        logger.info(f"📊 Final count: {final_count}")
        logger.info(f"✅ February created: {created_count}")
        logger.info(f"✅ February verified: {verified_count}")
        logger.info(f"🎯 Total 2024 test docs: {total_2024_count}")
        
        if created_count == verified_count == added_count:
            logger.info("🎯 SUCCESS: All February documents properly ingested into baremetal PostgreSQL!")
            logger.info("💪 Baremetal deployment continues to work perfectly!")
            logger.info(f"🚀 We now have {total_2024_count} test documents from 2024!")
        else:
            logger.warning(f"⚠️ Mismatch: Created={created_count}, Verified={verified_count}, Added={added_count}")
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ FEBRUARY 2024 INGESTION FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        
        final_count = await get_current_document_count()
        logger.error(f"Documents at failure: {final_count} (started with {initial_count})")
        
        raise


if __name__ == "__main__":
    asyncio.run(main())