from app.services.cross_encoder_model import LocalReranker
import asyncio
from collections import defaultdict
from app.services.weaviate_client import get_weaviate_client
from app.services.redis import get_cached_context, set_cached_context
from weaviate.classes.query import Filter, MetadataQuery
from app.config import RAG_CONFIG
import os
import logging
from typing import List, Dict
import hashlib


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global reranker (loaded once at startup)
try:
    reranker = LocalReranker()
    logger.info("✅ Reranker initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize reranker: {e}")
    reranker = None


def _version_number(v) -> float:
    """Convert version string to float for sorting"""
    try:
        if isinstance(v, str):
            v = v.lower().strip()
            if v.startswith("v"):
                v = v[1:]
            return float(v)
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def pick_latest_per_title(objects) -> List:
    """Group documents by title and pick latest version"""
    grouped = defaultdict(list)
    for obj in objects:
        title = obj.properties.get("title", "")
        grouped[title].append(obj)

    latest = []
    for title, objs in grouped.items():
        best = max(objs, key=lambda o: _version_number(o.properties.get("version", 0)))
        latest.append(best)
    return latest


async def rerank_documents_async(query: str, documents: list, top_k: int = 5) -> List:
    """Async wrapper for reranking"""
    if not documents:
        return []
    
    # Fallback if reranker is not available
    if reranker is None:
        logger.warning("⚠️ Reranker not available, returning original order")
        return [{"document": doc, "relevance_score": None} for doc in documents[:top_k]]
    
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            reranker.rerank,
            query,
            documents,
            top_k
        )
    except Exception as e:
        logger.error(f"❌ Reranking failed: {e}")
        return [{"document": doc, "relevance_score": None} for doc in documents[:top_k]]
    
async def search_law_collection_async(
    client, 
    collection_name: str, 
    query_text: str, 
    limit: int
) -> List:
    """Search a single law collection"""
    try:
        collection = client.collections.get(collection_name)
        response = collection.query.near_text(
            query=query_text,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )
        
        if response and response.objects:
            logger.info(f"⚖️ {collection_name}: {len(response.objects)} results")
            return response.objects
        return []
    except Exception as e:
        logger.warning(f"⚠️ {collection_name} search failed: {e}")
        return []



async def build_context_from_weaviate_results(
    organization: str, 
    query_text: str,
    question_type: str
) -> Dict:
    """
    Build context from Weaviate with optimizations:
    - Only fetch law docs when needed
    - Parallel law collection searches
    - Redis caching with organization isolation
    
    CACHE STRATEGY:
    - Organization-specific cache keys prevent data leakage
    - Law context can be shared across orgs (same legal documents)
    - Org context is always organization-specific
    """
    # Generate organization-specific cache key
    query_hash = hashlib.md5(query_text.encode()).hexdigest()
    cache_key = f"context:v2:{organization}:{question_type}:{query_hash}"
    
    # Try cache first
    cached = await get_cached_context(cache_key)
    if cached:
        logger.info(f"✅ Cache hit for {organization}")
        return cached
    
    client = get_weaviate_client()
    
    try:
        if not client.is_connected():
            client.connect()

        # ========== STEP 1: Fetch organization documents ==========
        logger.info(f"🔍 Searching organization: {organization}")
        
        try:
            org_collection = client.collections.get(organization)
            org_results = org_collection.query.fetch_objects()
            org_latest = pick_latest_per_title(org_results.objects)
            logger.info(f"📄 Organization docs: {len(org_latest)}")
        except Exception as e:
            logger.warning(f"⚠️ Organization {organization} not found: {e}")
            org_latest = []
        
        # ========== STEP 2: Conditionally fetch law documents ==========
        law_documents = []
        
        # For MIXED questions, ALWAYS fetch law context
        # For LAW questions, fetch law context
        # For POLICY questions, optionally fetch law for reference
        if question_type in ["LAW", "MIXED"]:
            # Try to get law context from shared cache first
            law_cache_key = f"law:v2:{question_type}:{query_hash}"
            cached_law = await get_cached_context(law_cache_key)
            
            if cached_law:
                logger.info(f"✅ Law cache hit (shared across orgs)")
                law_documents = cached_law.get("law_docs", [])
            else:
                logger.info(f"⚖️ Fetching law context for {question_type} question")
                
                law_collections = ["AgedCareAct", "HomeCareAct", "NDIS", "GeneralAct", "Others"]
                limit_per_collection = RAG_CONFIG["initial_fetch"] // len(law_collections)
                
                # Parallel search across law collections
                search_tasks = [
                    search_law_collection_async(
                        client, 
                        collection_name, 
                        query_text, 
                        limit_per_collection
                    )
                    for collection_name in law_collections
                ]
                
                law_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                for result in law_results:
                    if isinstance(result, list):
                        law_documents.extend(result)
                
                logger.info(f"⚖️ Total law docs: {len(law_documents)}")
                
                # Cache law context separately (can be shared across organizations)
                await set_cached_context(
                    law_cache_key, 
                    {"law_docs": law_documents}, 
                    RAG_CONFIG["cache_ttl"] * 2  # Law docs can be cached longer
                )
        else:
            logger.info(f"⏭️ Skipping law context for {question_type} question")

        # ========== STEP 3: Vector search in organization ==========
        org_context = []
        if org_latest:
            try:
                org_vector_response = org_collection.query.near_text(
                    query=query_text,
                    filters=Filter.by_id().contains_any([str(obj.uuid) for obj in org_latest]),
                    limit=RAG_CONFIG["initial_fetch"] // 2,
                    return_metadata=MetadataQuery(score=True)
                )
                
                if org_vector_response and org_vector_response.objects:
                    org_reranked = await rerank_documents_async(
                        query=query_text,
                        documents=org_vector_response.objects,
                        top_k=RAG_CONFIG["rerank_top_k"]
                    )
                    org_context = [item['document'] for item in org_reranked]
            except Exception as e:
                logger.error(f"❌ Org vector search failed: {e}")
        
        # ========== STEP 4: Rerank law documents ==========
        law_context = []
        if law_documents:
            law_reranked = await rerank_documents_async(
                query=query_text,
                documents=law_documents,
                top_k=RAG_CONFIG["rerank_top_k"]
            )
            law_context = [item['document'] for item in law_reranked]
        result = {
            "org_context": org_context,
            "law_context": law_context
        }
        
        # Cache the organization-specific result
        await set_cached_context(cache_key, result, RAG_CONFIG["cache_ttl"])
        
        logger.info(f"✅ Context built for {organization}: {len(org_context)} org + {len(law_context)} law")
        return result

    except Exception as e:
        logger.error(f"❌ Context build failed: {e}")
        # Return empty context instead of raising
        return {"org_context": [], "law_context": []}
    finally:
        if client.is_connected():
            client.close()
