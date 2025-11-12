from openai import OpenAI
import logging
from typing import List, Dict
import json
import numpy as np
from fastapi import HTTPException, UploadFile
from app.services.extract_content import extract_content_from_uploadpdf
from functools import lru_cache
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.services.weaviate_client import get_weaviate_client
from app.config import OPENAI_API_KEY

# Create a sync wrapper for async compatibility
def run_in_executor(func, *args):
    """Run sync function in thread pool"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args)


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@lru_cache(maxsize=50)
def fetch_weaviate_policies_cached(collection_name: str, cache_key: str = "default"):
    """Cache করা Weaviate policies fetch"""
    try:
        client = get_weaviate_client()
        if not client.is_connected():
            client.connect()
        collection = client.collections.get(collection_name)
        response = collection.query.fetch_objects(limit=1000)
        return response.objects
    except Exception as e:
        logger.error(f"Weaviate fetch error: {str(e)}")
        return []
    finally:
        if client.is_connected():
            client.close()


def fetch_weaviate_policies(collection_name: str):
    """Non-cached version for direct calls"""
    import time
    cache_key = str(int(time.time() // 300))  # 5 minute cache
    return fetch_weaviate_policies_cached(collection_name, cache_key)


def chunk_text(text: str, max_chunk_size: int = 8000, overlap: int = 300) -> List[str]:
    """Optimized chunking with smaller chunks"""
    if len(text) <= max_chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


async def summarize_large_text(openai_client: OpenAI, text: str, title: str) -> str:
    """Optimized summarization with parallel chunk processing"""
    if not text:
        return ""
    
    chunks = chunk_text(text, max_chunk_size=8000, overlap=300)
    
    if len(chunks) == 1:
        try:
            resp = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Concise policy summary in 6-8 bullets."},
                    {"role": "user", "content": f"Title: {title}\n\n{chunks[0]}"}
                ],
                temperature=0.1,
                max_tokens=600
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Summary failed: {str(e)}")
            return text[:2000]
    
    # Multiple chunks - parallel processing
    async def summarize_chunk(idx: int, chunk: str) -> str:
        try:
            resp = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract key points in 4-5 bullets."},
                    {"role": "user", "content": f"{title} Part {idx+1}:\n{chunk}"}
                ],
                temperature=0.1,
                max_tokens=400
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return chunk[:800]
    
    # Process chunks in parallel
    part_summaries = await asyncio.gather(*[summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)])
    
    # Quick final combination
    combined = "\n\n".join(part_summaries)
    if len(combined) <= 3000:
        return combined
    
    # Final compression if needed
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Combine into 6-8 key bullets."},
                {"role": "user", "content": f"{title}:\n{combined}"}
            ],
            temperature=0.1,
            max_tokens=600
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return combined[:3000]


async def fetch_weaviate_full_text(collection_name: str) -> str:
    """Concatenate full text from organization-specific and general policies"""
    # Fetch organization-specific policies
    org_objects = fetch_weaviate_policies(collection_name)
    
    # Fetch general policies from GeneralLaw collection
    general_objects = fetch_weaviate_policies('GeneralLaw')
    
    texts: List[str] = []
    
    # Add organization-specific policies
    for obj in org_objects:
        title = obj.properties.get("title", "Policy")
        text = obj.properties.get("text", "")
        if text:
            texts.append(f"Title: {title}\n{text}")
    
    # Add general policies
    for obj in general_objects:
        title = obj.properties.get("title", "General Policy")
        text = obj.properties.get("text", "")
        if text:
            texts.append(f"Title: {title}\n{text}")
    
    return "\n\n\n".join(texts)


async def extract_pdf_content(file: UploadFile):
    """Async PDF content extraction"""
    try:
        await file.seek(0)
        text, title = await extract_content_from_uploadpdf(file)
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF."
            )
        return text.strip(), title or "Untitled Document"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process PDF: {str(e)}"
        )


async def cosine_similarity_test(file: UploadFile, organization_type: str):
    """Optimized cosine similarity with async execution against org and general policies"""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Parallel execution for both org and general policies
    pdf_task = extract_pdf_content(file)
    org_policies_task = asyncio.to_thread(fetch_weaviate_policies, organization_type)
    general_policies_task = asyncio.to_thread(fetch_weaviate_policies, 'GeneralLaw')
    
    (full_text, _title), org_policies, general_policies = await asyncio.gather(
        pdf_task, org_policies_task, general_policies_task
    )
    
    # Combine both policy sets
    policies = org_policies + general_policies
    
    try:
        # Generate embedding
        max_chars = 8192 * 3
        if len(full_text) > max_chars:
            embedding_text = (
                full_text[:max_chars // 2]
                + "\n\n[...CONTENT_TRUNCATED...]\n\n"
                + full_text[-max_chars // 2:]
            )
        else:
            embedding_text = full_text
            
        embedding_response = await asyncio.to_thread(
            openai_client.embeddings.create,
            model="text-embedding-3-small",
            input=embedding_text
        )
        uploaded_embedding = embedding_response.data[0].embedding
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {str(e)}"
        )
    
    similarities = []
    for policy_obj in policies:
        policy_embedding = policy_obj.properties.get("embedding", [])
        if policy_embedding:
            similarity_score = cosine_similarity(uploaded_embedding, policy_embedding)
            similarities.append(similarity_score)
    
    if not similarities:
        raise HTTPException(
            status_code=500,
            detail="No valid embeddings found for comparison"
        )
    
    max_similarity = max(similarities)
    alignment_percent = round(max_similarity * 100, 2)
    not_alignment_percent_cosine = round(100 - alignment_percent, 2)
    
    return {
        "not_alignment_percent": not_alignment_percent_cosine
    }


# async def combined_alignment_analysis(text: str, title: str, organization_type: str) -> Dict[str, object]:
#     """Ultra-fast parallel analysis with optimized OpenAI calls"""
#     openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
#     # Step 1: Fetch Weaviate data (PDF text already provided)
#     weaviate_full_text = await fetch_weaviate_full_text(organization_type)

    
#     # Step 2: Parallel summarization (text already extracted)
#     pdf_summary, weaviate_summary = await asyncio.gather(
#         summarize_large_text(openai_client, text, title),
#         summarize_large_text(openai_client, weaviate_full_text, "Main Laws")
#     )
    
#     # Step 3: Single optimized LLM call for everything
#     combined_prompt = (
#         f"Analyze these policy summaries and provide a complete analysis:\n\n"
#         f"PDF Summary:\n{pdf_summary}\n\n"
#         f"Policy Summary:\n{weaviate_summary}\n\n"
#         f"Return strict JSON with this exact structure:\n"
#         f"{{\n"
#         f'  "direct_conflict": true or false,\n'
#         f'  "conflicts": ["conflict1", "conflict2", ...],\n'
#         f'  "differences": ["diff1", "diff2", ...],\n'
#         f'  "paragraph": "Single paragraph (4-6 sentences) explaining conflicts or key differences without mentioning system names."\n'
#         f"}}"
#     )
    
#     try:
#         resp = await asyncio.to_thread(
#             openai_client.chat.completions.create,
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a precise policy analyst. Return only valid JSON."},
#                 {"role": "user", "content": combined_prompt}
#             ],
#             temperature=0.1,
#             max_tokens=1000
#         )
        
#         total_tokens = resp.usage.total_tokens
#         content = resp.choices[0].message.content.strip()
        
#         # Clean JSON response
#         if content.startswith("```json"):
#             content = content[7:]
#         if content.endswith("```"):
#             content = content[:-3]
        
#         result = json.loads(content.strip())
        
#         # Calculate alignment score
#         if result.get("direct_conflict", False):
#             conflict_count = len(result.get("conflicts", []))
#             not_alignment_percent = min(85 + (conflict_count * 5), 100)
#         else:
#             difference_count = len(result.get("differences", []))
#             not_alignment_percent = min(10 + (difference_count * 5), 35)
        
#         return {
#             "not_alignment_percent": round(not_alignment_percent, 1),
#             "contradiction_paragraph": result.get("paragraph", "Analysis unavailable."),
#             "tokens_used": total_tokens
#         }
        
#     except json.JSONDecodeError as e:
#         logger.error(f"JSON parse error: {str(e)}")
#         return {
#             "not_alignment_percent": 25.0,
#             "contradiction_paragraph": "Analysis completed but formatting error occurred.",
#             "tokens_used": 0
#         }
#     except Exception as e:
#         logger.warning(f"Combined analysis failed: {str(e)}")
#         return {
#             "not_alignment_percent": 25.0,
#             "contradiction_paragraph": "Analysis unavailable due to processing error.",
#             "tokens_used": 0
#         }

async def combined_alignment_analysis(text: str, title: str, organization_type: str) -> Dict[str, object]:
    """Ultra-fast parallel analysis with actionable suggestions"""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Step 1: Fetch Weaviate data (PDF text already provided)
    weaviate_full_text = await fetch_weaviate_full_text(organization_type)

    
    # Step 2: Parallel summarization (text already extracted)
    pdf_summary, weaviate_summary = await asyncio.gather(
        summarize_large_text(openai_client, text, title),
        summarize_large_text(openai_client, weaviate_full_text, "Main Laws")
    )
    
    # Step 3: Single optimized LLM call with suggestions
    combined_prompt = (
        f"Analyze these policy summaries and provide a complete analysis with actionable suggestions:\n\n"
        f"PDF Summary:\n{pdf_summary}\n\n"
        f"Policy Summary:\n{weaviate_summary}\n\n"
        f"Return strict JSON with this exact structure:\n"
        f"{{\n"
        f'  "direct_conflict": true or false,\n'
        f'  "conflicts": ["conflict1", "conflict2", ...],\n'
        f'  "differences": ["diff1", "diff2", ...],\n'
        f'  "suggestions": [\n'
        f'    {{\n'
        f'      "issue": "Brief description of the conflict/difference",\n'
        f'      "recommendation": "Specific action to resolve it",\n'
        f'      "priority": "high/medium/low"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "paragraph": "Single paragraph (4-6 sentences) explaining conflicts or key differences without mentioning system names."\n'
        f"}}\n\n"
        f"IMPORTANT: For each conflict or significant difference, provide a clear suggestion with:\n"
        f"- The specific issue that needs attention\n"
        f"- A concrete recommendation on how to fix it\n"
        f"- Priority level (high for conflicts, medium/low for differences)\n"
        f"If no conflicts exist, provide suggestions for improvement or alignment."
    )
    
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise policy analyst. Return only valid JSON with actionable suggestions."},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.1,
            max_tokens=1500  # Increased for suggestions
        )
        
        total_tokens = resp.usage.total_tokens
        content = resp.choices[0].message.content.strip()
        
        # Clean JSON response
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        result = json.loads(content.strip())
        
        # Calculate alignment score
        if result.get("direct_conflict", False):
            conflict_count = len(result.get("conflicts", []))
            not_alignment_percent = min(85 + (conflict_count * 5), 100)
        else:
            difference_count = len(result.get("differences", []))
            not_alignment_percent = min(10 + (difference_count * 5), 35)
        
        # Ensure suggestions exist
        suggestions = result.get("suggestions", [])
        if not suggestions and (result.get("conflicts") or result.get("differences")):
            # Fallback: Generate basic suggestions from conflicts/differences
            suggestions = []
            for conflict in result.get("conflicts", [])[:3]:
                suggestions.append({
                    "issue": conflict,
                    "recommendation": "Review and align this section with regulatory requirements",
                    "priority": "high"
                })
            for diff in result.get("differences", [])[:2]:
                suggestions.append({
                    "issue": diff,
                    "recommendation": "Consider updating to match best practices",
                    "priority": "medium"
                })
        
        return {
            "not_alignment_percent": round(not_alignment_percent, 1),
            "contradiction_paragraph": result.get("paragraph", "Analysis unavailable."),
            "suggestions": suggestions,
            "tokens_used": total_tokens
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        return {
            "not_alignment_percent": 0,
            "contradiction_paragraph": "Analysis completed but formatting error occurred.",
            "suggestions": [],
            "tokens_used": 0,
            "error": str(e)
        }
    except Exception as e:
        logger.warning(f"Combined analysis failed: {str(e)}")
        return {
            "not_alignment_percent": 0,
            "contradiction_paragraph": "Analysis unavailable due to processing error.",
            "suggestions": [],
            "tokens_used": 0,
            "error": str(e)
        }
    
# ---------------------------------------------------------------------------------------------------
async def summarize_pdf_and_policies(text: str, title: str, organization_type: str) -> Dict[str, str]:
    """Fast parallel summarization of PDF and policies"""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Fetch Weaviate data (PDF text already provided)
    weaviate_full_text = await asyncio.to_thread(fetch_weaviate_full_text, organization_type)
    
    # Parallel summarization
    pdf_summary, weaviate_summary = await asyncio.gather(
        summarize_large_text(openai_client, text, title),
        summarize_large_text(openai_client, weaviate_full_text, "Main Policies")
    )
    
    return {
        "pdf_summary": pdf_summary,
        "weaviate_summary": weaviate_summary,
    }


# Legacy sync functions kept for backwards compatibility
async def compare_summaries_with_llm(openai_client: OpenAI, pdf_summary: str, weaviate_summary: str) -> Dict[str, str]:
    """Legacy function - use combined_alignment_analysis instead"""
    prompt = (
        "Compare two summaries (A: PDF, B: Policies).\n"
        "Decide alignment (ALIGNED or NOT_ALIGNED) with short reasoning (max 6 lines).\n\n"
        f"Summary A:\n{pdf_summary}\n\n"
        f"Summary B:\n{weaviate_summary}\n\n"
        "Return JSON: {{\"alignment_status\": \"ALIGNED|NOT_ALIGNED\", \"reasoning\": \"...\"}}"
    )
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        content = resp.choices[0].message.content.strip()
        obj = json.loads(content)
        return {
            "alignment_status": obj.get("alignment_status", "UNKNOWN"),
            "reasoning": obj.get("reasoning", "")
        }
    except Exception as e:
        logger.warning(f"Comparison failed: {str(e)}")
        return {"alignment_status": "UNKNOWN", "reasoning": "Comparison failed."}


async def detect_conflicts_or_differences(openai_client: OpenAI, pdf_summary: str, weaviate_summary: str) -> Dict[str, object]:
    """Legacy function - use combined_alignment_analysis instead"""
    prompt = (
        f"Analyze summaries for conflicts:\nPDF: {pdf_summary}\nPolicies: {weaviate_summary}\n"
        f"Return JSON: {{\"direct_conflict\": true/false, \"conflicts\": [...], \"differences\": [...]}}"
    )
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return strict JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        content = resp.choices[0].message.content.strip()
        obj = json.loads(content)
        return {
            "direct_conflict": bool(obj.get("direct_conflict", False)),
            "conflicts": obj.get("conflicts", [])[:8],
            "differences": obj.get("differences", [])[:8],
            "note": obj.get("note", "")
        }
    except Exception as e:
        logger.warning(f"Detection failed: {str(e)}")
        return {"direct_conflict": False, "conflicts": [], "differences": [], "note": "Failed."}


async def generate_contradiction_paragraph(openai_client: OpenAI, pdf_summary: str, weaviate_summary: str) -> str:
    """Legacy function - use combined_alignment_analysis instead"""
    prompt = (
        f"ONE paragraph (4-6 sentences) on contradictions between:\n"
        f"PDF: {pdf_summary}\nPolicies: {weaviate_summary}"
    )
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Single paragraph only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Paragraph generation failed: {str(e)}")
        return "Analysis unavailable."