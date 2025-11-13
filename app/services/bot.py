import asyncio
import openai
from fastapi.responses import JSONResponse
from app.config import OPENAI_API_KEY
from openai import AsyncOpenAI
from app.services.system_prompt_builder import build_system_prompt
from app.services.fetch_history import fetch_history_async
from app.services.store_data import save_data_parallel
from app.services.build_context import build_context_from_weaviate_results
from app.services.content_formatter import build_user_message_with_context, build_minimal_history
from app.services.content_formatter import formatted_content
from app.config import RAG_CONFIG
from app.services.cross_encoder_model import LocalReranker
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global reranker
reranker = LocalReranker()

def classify_question_type(question: str) -> str:
    """
    Classify question as CASUAL, EMERGENCY, INFORMATIVE, LAW, POLICY, or MIXED
    
    Returns: "CASUAL", "EMERGENCY", "INFORMATIVE", "LAW", "POLICY", or "MIXED"
    """
    q_lower = question.lower().strip()
    
    # CASUAL CHAT - Simple greetings and personal questions
    casual_indicators = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how about you', 'what\'s your name', 'who are you',
        'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'nice to meet you'
    ]
    
    # EMERGENCY - Urgent situations requiring immediate help
    emergency_indicators = [
        'earthquake', 'fire', 'accident', 'emergency', 'help', 'urgent', 'crisis',
        'vumikompo', 'ভূমিকম্প', 'আগুন', 'দুর্ঘটনা', 'জরুরি', 'সাহায্য'
    ]
    
    # Check for casual chat
    if any(indicator in q_lower for indicator in casual_indicators) or len(q_lower) <= 3:
        return "CASUAL"
    
    # Check for emergency situations
    if any(indicator in q_lower for indicator in emergency_indicators):
        return "EMERGENCY"
    
    # ONLY LAW - User explicitly wants just legal information
    only_law_indicators = [
        'what does the law say', 'according to law only',
        'legal requirement only', 'just the legislation',
        'only the act', 'law states', 'legally required'
    ]
    
    # ONLY POLICY - User explicitly wants just org policy
    only_policy_indicators = [
        'our policy', 'our organization', 'our procedure',
        'we use', 'our guideline', 'company policy',
        'internal procedure', 'our approach', 'how do we',
        'what is our', 'our process'
    ]
    
    # Check for explicit "only law" requests
    if any(indicator in q_lower for indicator in only_law_indicators):
        return "LAW"
    
    # Check for explicit "only policy" requests  
    if any(indicator in q_lower for indicator in only_policy_indicators):
        return "POLICY"
    
    # INFORMATIVE - General knowledge questions (default for most questions)
    # Examples: "What is medication management?", "How to handle falls?"
    return "INFORMATIVE"
def log_cache_stats(response):
    """
    Extract and log cache statistics from OpenAI response
    """
    usage = response.usage
    
    # Basic token counts
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    
    # Try to get cache info (if available)
    cached_tokens = 0
    cache_creation_tokens = 0
    
    if hasattr(usage, 'prompt_tokens_details'):
        details = usage.prompt_tokens_details
        
        # Cached tokens (tokens read from cache)
        if hasattr(details, 'cached_tokens'):
            cached_tokens = details.cached_tokens
        
        # Cache creation tokens (new tokens being cached)
        if hasattr(details, 'cache_creation_tokens'):
            cache_creation_tokens = details.cache_creation_tokens
    
    # Calculate metrics
    cache_hit_rate = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
    new_tokens = prompt_tokens - cached_tokens
    
    # Log with emojis for visibility
    logger.info("=" * 60)
    logger.info("📊 TOKEN USAGE BREAKDOWN")
    logger.info("=" * 60)
    logger.info(f"Total tokens used: {total_tokens}")
    logger.info(f"  ├─ Prompt tokens: {prompt_tokens}")
    logger.info(f"  │   ├─ 🎯 Cached (from cache): {cached_tokens}")
    logger.info(f"  │   ├─ 💾 Cache creation (being cached): {cache_creation_tokens}")
    logger.info(f"  │   └─ ⚡ New (not cached): {new_tokens}")
    logger.info(f"  └─ Completion tokens: {completion_tokens}")
    logger.info(f"")
    logger.info(f"📈 Cache Performance:")
    logger.info(f"  ├─ Cache hit rate: {cache_hit_rate:.1f}%")
    logger.info(f"  └─ Status: {'✅ Excellent!' if cache_hit_rate > 70 else '⚠️ Low - check system prompt' if cache_hit_rate > 0 else '❌ No caching detected'}")
    logger.info("=" * 60)
    
    # Cost calculation
    # gpt-4o pricing: Input $2.50/M, Cached input $1.25/M, Output $10/M
    cost_new = new_tokens * 2.50 / 1_000_000
    cost_cached = cached_tokens * 1.25 / 1_000_000
    cost_output = completion_tokens * 10 / 1_000_000
    total_cost = cost_new + cost_cached + cost_output
    
    # What it would have cost WITHOUT caching
    cost_without_cache = prompt_tokens * 2.50 / 1_000_000 + cost_output
    savings = cost_without_cache - total_cost
    
    logger.info(f"💰 Cost for this request:")
    logger.info(f"  ├─ New tokens: ${cost_new:.6f}")
    logger.info(f"  ├─ Cached tokens: ${cost_cached:.6f}")
    logger.info(f"  ├─ Output tokens: ${cost_output:.6f}")
    logger.info(f"  └─ Total: ${total_cost:.6f}")
    logger.info(f"")
    logger.info(f"💸 Savings from caching:")
    logger.info(f"  ├─ Without cache: ${cost_without_cache:.6f}")
    logger.info(f"  ├─ With cache: ${total_cost:.6f}")
    logger.info(f"  └─ Saved: ${savings:.6f} ({savings/cost_without_cache*100:.1f}%)")
    logger.info("=" * 60)
    
    return {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "new_tokens": new_tokens,
        "cache_hit_rate": f"{cache_hit_rate:.1f}%",
        "cost": {
            "total": f"${total_cost:.6f}",
            "saved": f"${savings:.6f}",
            "savings_percent": f"{savings/cost_without_cache*100:.1f}%"
        }
    }

async def ask_doc_bot(
    question: str, 
    organization: str, 
    auth_token: str
) -> JSONResponse:
    """
    Main chatbot function with all optimizations
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # ================= VALIDATION =================
        # if not question or len(question.strip()) < 3:
        #     return JSONResponse(
        #         status_code=400,
        #         content={
        #             "status": "error",
        #             "message": "Question too short (minimum 3 characters)"
        #         }
        #     )
        
        # if len(question) > 1000:
        #     return JSONResponse(
        #         status_code=400,
        #         content={
        #             "status": "error",
        #             "message": "Question too long (maximum 1000 characters)"
        #         }
        #     )
        
        if not organization or not auth_token:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing required fields"
                }
            )
        
        # ================= EARLY AUTH & TOKEN CHECK =================
        from app.services.input_validator import validate_input
        validation_response = await validate_input(question, organization, auth_token)
        if validation_response is not True:
            return validation_response
        
        # ================= CLASSIFY QUESTION =================
        question_type = classify_question_type(question)
        logger.info(f"📝 Question type: {question_type}")
        
        # ================= CONDITIONAL PARALLEL FETCH =================
        logger.info("⏱️ Starting parallel fetch...")
        fetch_start = asyncio.get_event_loop().time()
        
        try:
            history_task = fetch_history_async(auth_token, limit=10, offset=0)
            
            # Skip context fetching for CASUAL questions only
            if question_type == "CASUAL":
                logger.info("⏭️ Skipping context fetch for casual question")
                history_result = await history_task
                context_result = {"org_context": [], "law_context": []}
            else:
                # Fetch context for EMERGENCY, INFORMATIVE, LAW, POLICY questions
                context_task = build_context_from_weaviate_results(
                    organization=organization,
                    query_text=question,
                    question_type=question_type
                )
                
                history_result, context_result = await asyncio.gather(
                    history_task, 
                    context_task,
                    return_exceptions=True
                )
                
                if isinstance(context_result, Exception):
                    logger.error(f"Context fetch failed: {context_result}")
                    context_result = {"org_context": [], "law_context": []}
            
            if isinstance(history_result, Exception):
                raise history_result
            
            org_context = context_result.get("org_context", [])
            law_context = context_result.get("law_context", [])
            
        except Exception as e:
            logger.error(f"❌ Parallel fetch failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to retrieve necessary data",
                    "error": str(e)
                }
            )
        
        fetch_time = asyncio.get_event_loop().time() - fetch_start
        logger.info(f"✅ Fetch done: {fetch_time:.2f}s | Org: {len(org_context)}, Law: {len(law_context)}")

        # Check token limit EARLY
        remaining_tokens = history_result.get('remaining_tokens')
        print(f"Remaining tokens: {remaining_tokens}")
        # if remaining_tokens is not None and remaining_tokens < RAG_CONFIG["min_tokens_required"]:
        #     logger.warning(f"❌ Insufficient tokens: {remaining_tokens}")
        #     return JSONResponse(
        #         status_code=400,
        #         content={
        #             "status": "error",
        #             "message": "Insufficient tokens to continue the conversation.",
        #             "remaining_tokens": remaining_tokens
        #         }
        #     )
        
        # Build chat history

        # chat_history = []
        # for h in history_result.get('histories', []):
        #     chat_history.append({"role": "user", "content": h['prompt']})
        #     chat_history.append({"role": "assistant", "content": h['response']})
        chat_history = build_minimal_history(
            history_result.get('histories', []),
            max_pairs=2  # Only last 2 exchanges
        )
        
        # ================= BUILD PROMPT =================
        system_prompt = build_system_prompt() # For caching, static system prompt
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        
        formatted_content_str = await formatted_content(question_type, org_context, law_context)
        
        # user_content = f"{formatted_content_str}QUESTION: {question}"

        # ✅ Extract previous assistant responses
        previous_responses = []
        for h in history_result.get('histories', [])[-3:]:  # শেষ 3টা
            if 'response' in h:
                previous_responses.append(h['response'])

        # User message build করার সময়
        user_content = await build_user_message_with_context(
            question=question,
            question_type=question_type,
            org_context=org_context,
            law_context=law_context,
            formatted_content_str=formatted_content_str,
            previous_responses=previous_responses  # ← পাঠানো
        )

        messages.append({"role": "user", "content": user_content})
        
        # ================= LLM CALL WITH FORCED JSON =================
        logger.info("🤖 Calling LLM...")
        llm_start = asyncio.get_event_loop().time()

        try:
            async with AsyncOpenAI(api_key=OPENAI_API_KEY) as openai_client:
                response = await openai_client.chat.completions.create(
                    model="gpt-4o",  # ✅ Cache-supported model
                    messages=messages,
                    temperature=0.95,
                    top_p=0.95,               # NEW
                    frequency_penalty=0.5,    # NEW - reduces repetition
                    presence_penalty=0.3,     # NEW - encourages variety
                    max_completion_tokens=3000,
                    response_format={"type": "json_object"}
                )
        except openai.APIError as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "AI service temporarily unavailable"
                }
            )
        except openai.RateLimitError as e:
            logger.error(f"❌ Rate limit: {e}")
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Too many requests. Please try again later."
                }
            )
        except Exception as e:
            logger.error(f"❌ LLM failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to generate response"
                }
            )
        
        llm_time = asyncio.get_event_loop().time() - llm_start

        cache_stats = log_cache_stats(response)
        print("cache statistics-----------------\n", cache_stats)
        used_tokens = response.usage.total_tokens
        
        logger.info(f"✅ LLM done: {llm_time:.2f}s | Tokens: {used_tokens}")
        
        # Parse JSON response
        answer = response.choices[0].message.content.strip()
        
        try:
            json_answer = json.loads(answer)
        except json.JSONDecodeError as e:
            logger.error(f"⚠️ JSON parse failed: {e}")
            json_answer = {
                "answer": answer,
                "used_document": question_type == "POLICY" and bool(org_context),
                "sources": []
            }
        
        # Validate used_document flag for MIXED questions
        if question_type == "LAW" and json_answer.get('used_document', False):
            logger.warning("⚠️ Correcting used_document flag for law question")
            json_answer['used_document'] = False
        
        if question_type == "MIXED":
            # For MIXED: only true if org context was actually used
            if not org_context:
                json_answer['used_document'] = False
                logger.info("ℹ️ MIXED question with no org context: used_document=False")
        
        # ============ PARALLEL SAVE (Background) ============
        logger.info("💾 Starting background save...")
        save_start = asyncio.get_event_loop().time()
        
        readcount_data = {}

        if json_answer.get('used_document', False) and org_context:
            for c in org_context:
                doc_id = c.properties.get("document_id", "")
                if doc_id:
                    readcount_data[doc_id] = 1
                    
        
        history_data = {
            "prompt": question,
            "response": json_answer['answer'],
            "used_tokens": used_tokens
        }
        
        token_data = {"used_tokens": used_tokens}
        
        try:
            save_results = await save_data_parallel(
                history_data, readcount_data, token_data, auth_token
            )
            
            save_time = asyncio.get_event_loop().time() - save_start
            logger.info(f"✅ Save done: {save_time:.2f}s")
            
            for result in save_results:
                if isinstance(result, dict) and not result.get('success', False) and not result.get('skipped', False):
                    logger.warning(f"⚠️ {result['type']} save failed")
        
        except Exception as e:
            logger.warning(f"⚠️ Background save failed: {e}")
            save_time = asyncio.get_event_loop().time() - save_start
        
        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"🎯 Total: {total_time:.2f}s")
        
        # ============ BUILD SOURCE DOCUMENTS LIST ============
        source_documents = []
        used_document = json_answer.get('used_document', False)

        # Only include source documents if the bot actually used them
        if used_document:
            # Add organization context sources (for POLICY questions)
            if org_context and question_type in ["POLICY", "MIXED"]:
                for i, doc in enumerate(org_context, 1):
                    source_documents.append({
                        "document_id": doc.properties.get("document_id", ""),
                        # "chunk_index": i
                    })
            
            # Add law context sources (for LAW or MIXED questions)
            if law_context and question_type in ["LAW", "MIXED"]:
                for i, doc in enumerate(law_context, 1):
                    source_documents.append({
                        "document_id": doc.properties.get("document_id", ""),
                        # "chunk_index": i
                    })

        # ============ BUILD RESPONSE (CONDITIONAL SOURCE DOCS) ============
        print("llm answer:-----------------\n", json_answer['answer'])
        response_content = {
            "status": "success",
            "question": question,
            "answer": json_answer['answer'],
            "used_tokens": used_tokens
        }

        # Only add source_documents if bot used documents
        if used_document and source_documents:
            response_content["source_documents"] = source_documents

        return JSONResponse(status_code=200, content=response_content)
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred",
                "error": str(e)
            }
        )












# import asyncio
# import aiohttp
# import openai
# from fastapi.responses import JSONResponse
# from app.config import OPENAI_API_KEY, BACKEND_HISTORY_URL, BACKEND_DOC_READ_COUNT_URL
# from openai import AsyncOpenAI
# from app.services.llm_response_correction import extract_json_from_llm
# from app.services.store_used_token import used_token_store
# import json
# from app.services.fetch_history import fetch_history_async
# from app.services.build_context import build_context_from_weaviate_results

# async def save_data_parallel(history_data: dict, readcount_data: dict, 
#                             token_data: dict, auth_token: str):
#     """Save history, read count, and token data in parallel with dynamic error handling"""
#     header = {"Authorization": f"Bearer {auth_token}"}
    
#     async def post_with_error_handling(url: str, data: dict, request_type: str):
#         """Generic POST handler with single exception block"""
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(url, json=data, headers=header, 
#                                        timeout=aiohttp.ClientTimeout(total=10)) as response:
#                     status = response.status
                    
#                     if status in [200, 201]:
#                         return {"type": request_type, "status": status, "success": True}
#                     else:
#                         # Extract error message from response
#                         error_msg = f"Failed with status {status}"
#                         try:
#                             error_data = await response.json()
#                             error_msg = error_data.get('message', error_data.get('detail', error_msg))
#                         except:
#                             pass
                        
#                         print(f"❌ {request_type} save failed [{status}]: {error_msg}")
#                         return {
#                             "type": request_type,
#                             "status": status,
#                             "success": False,
#                             "error": error_msg
#                         }
#         except Exception as e:
#             # All errors handled in one block
#             print(f"❌ {request_type} save error: {type(e).__name__} - {str(e)}")
            
#             status_code = 503
#             if isinstance(e, asyncio.TimeoutError):
#                 status_code = 504
            
#             return {
#                 "type": request_type,
#                 "status": status_code,
#                 "success": False,
#                 "error": str(e)
#             }
    
#     async def post_history():
#         return await post_with_error_handling(BACKEND_HISTORY_URL, history_data, "history")
    
#     async def post_readcount():
#         if not readcount_data:
#             return {"type": "readcount", "status": 200, "skipped": True, "success": True}
#         return await post_with_error_handling(BACKEND_DOC_READ_COUNT_URL, readcount_data, "readcount")
    
#     async def post_token():
#         try:
#             loop = asyncio.get_event_loop()
#             resp = await loop.run_in_executor(
#                 None, 
#                 lambda: used_token_store(
#                     type='chatbot', 
#                     used_tokens=token_data['used_tokens'], 
#                     auth_token=auth_token
#                 )
#             )
            
#             status = resp.status_code if hasattr(resp, 'status_code') else 200
#             if status in [200, 201]:
#                 return {"type": "token", "status": status, "success": True}
#             else:
#                 error_msg = f"Failed with status {status}"
#                 try:
#                     error_msg = resp.json().get('message', error_msg) if hasattr(resp, 'json') else error_msg
#                 except:
#                     pass
                
#                 print(f"❌ Token save failed [{status}]: {error_msg}")
#                 return {
#                     "type": "token",
#                     "status": status,
#                     "success": False,
#                     "error": error_msg
#                 }
#         except Exception as e:
#             print(f"❌ Token save error: {type(e).__name__} - {str(e)}")
#             return {
#                 "type": "token",
#                 "status": 500,
#                 "success": False,
#                 "error": str(e)
#             }
    
#     results = await asyncio.gather(
#         post_history(),
#         post_readcount(),
#         post_token(),
#         return_exceptions=True
#     )
    
#     return results

# async def ask_doc_bot(question: str, organization: str, auth_token: str):
#     try:
#         # ================= STEP 0: Check Auth Token =================
#         history_result = await fetch_history_async(auth_token)
#         if not history_result.get("success", True):
#             if history_result.get("error") == "Unauthorized":
#                 return JSONResponse(
#                     status_code=401,
#                     content={
#                         "success": False,
#                         "error": "Unauthorized",
#                         "message": "You are unauthorized to access this resource."
#                     }
#                 )

#         # ================= STEP 1: PARALLEL FETCH (History + Context) =================
#         print("⏱️ Starting parallel fetch...")
#         start_time = asyncio.get_event_loop().time()
        
#         try:
#             history_task = fetch_history_async(auth_token)
#             context_task = build_context_from_weaviate_results(
#                 organization=organization,
#                 query_text=question,
#                 initial_limit=20,
#                 final_limit=5,
#                 include_law=True  # Always include all law collections
#             )
            
#             history_result, context_result = await asyncio.gather(history_task, context_task)
            
#             # Extract separate contexts
#             org_context = context_result.get("org_context", []) if isinstance(context_result, dict) else []
#             law_context = context_result.get("law_context", []) if isinstance(context_result, dict) else []
            
#         except Exception as e:
#             print(f"❌ Parallel fetch failed: {e}")
#             return JSONResponse(status_code=500, content={
#                 "status": "error",
#                 "message": "Failed to retrieve necessary data",
#                 "error": str(e)
#             })
        
#         fetch_time = asyncio.get_event_loop().time() - start_time
#         print(f"✅ Parallel fetch completed in {fetch_time:.2f}s")
        
#         # ============ CHECK HISTORY FETCH ERRORS ============
#         if not history_result['success']:
#             error_code = history_result.get('status_code', 500)
#             error_message = history_result.get('message', 'Failed to fetch chat history')
            
#             print(f"❌ History fetch failed: {error_message}")
#             return JSONResponse(status_code=error_code, content={
#                 "status": "error",
#                 "message": error_message,
#                 "error_type": history_result.get('error', 'unknown_error')
#             })
        
#         # ============ CHECK TOKEN LIMIT ============
#         if history_result['remaining_tokens'] is not None and history_result['remaining_tokens'] < 1000:
#             print("❌ Insufficient tokens")
#             return JSONResponse(status_code=400, content={
#                 "status": "error",
#                 "message": "Insufficient tokens to continue the conversation.",
#                 "remaining_tokens": history_result['remaining_tokens']
#             })
        
#         # Build chat history
#         chat_history = []
#         for h in history_result.get('histories', []):
#             chat_history.append({"role": "user", "content": h['prompt']})
#             chat_history.append({"role": "assistant", "content": h['response']})
        
#         # ================= DEBUG CONTEXT =================
#         print(f"📄 Org Context: {len(org_context)} docs, Law Context: {len(law_context)} docs")
#         if org_context:
#             for i, doc in enumerate(org_context[:2]):
#                 title = doc.properties.get('title', 'No title')[:30]
#                 print(f"   Org {i+1}: {title}")
#         if law_context:
#             for i, doc in enumerate(law_context[:2]):
#                 title = doc.properties.get('title', 'No title')[:30]
#                 print(f"   Law {i+1}: {title}")
        
#         # ================= STEP 2: LLM CALL ================="
               
#         system_prompt = (
#             "You are Nestor AI, a friendly and knowledgeable assistant specializing in aged care and Australian law. "
#             "You communicate with warmth, empathy, and genuine care for helping people understand complex topics.\n\n"

#             "🎯 **PERSONALITY & STYLE:**\n"
#             "- Be conversational, warm, and approachable like a real chatbot\n"
#             "- Start with natural responses like 'Sure!', 'Of course!', 'Absolutely!', 'Good question!'\n"
#             "- End with encouraging closings like 'Hope this helps!', 'Let me know if you need more info!', 'Happy to help!'\n"
#             "- Use natural emojis occasionally to add warmth (🏛️, 📋, ⚖️, 👥, 💡, 🤔)\n"
#             "- Break down complex information into digestible, conversational chunks\n"
#             "- Use phrases like 'Here's what I found', 'Based on my knowledge', 'Let me explain this for you'\n"
#             "- Show genuine interest with phrases like 'That's a great question!', 'I understand your concern about...'\n\n"

#             "💬 **CONVERSATIONAL FLOW:**\n"
#             "1. **Opening:** Start with a warm greeting that acknowledges the question\n"
#             "2. **Body:** Provide the main information in clear, organized but natural sections\n"
#             "3. **Closing:** End with an encouraging note and invitation for follow-up\n"
#             "4. **Tone:** Maintain helpful, patient, and supportive tone throughout\n\n"

#             "🔧 **COMPLEX SCENARIOS (NEW):**\n"
#             "- When a user asks a scenario-based or multi-step question, follow this pattern automatically:\n"
#             "  1) Summarize the user's scenario in 1-2 sentences.\n"
#             "  2) List explicit assumptions you make (numbered).\n"
#             "  3) Provide an analysis that covers key edge cases and recommended options.\n"
#             "  4) If legal/policy obligations apply, mark them clearly and cite exact lines where possible.\n\n"

#             "📝 **STEP-BY-STEP ACTIONS / SOP (NEW):**\n"
#             "- For any 'how-to' or operational question, always include a numbered SOP with:\n"
#             "  • Preconditions / prerequisites (short checklist).\n"
#             "  • Clear numbered steps (each one short and actionable).\n"
#             "  • Roles or inputs required per step (if applicable).\n"
#             "  • Expected outcome for each major step.\n"
#             "- If full automation isn't possible, include a concise 'What to do if X fails' fallback.\n\n"

#             "📚 **CITATIONS & EXACT POLICY LINES (NEW):**\n"
#             "- When referring to law or policy, do the following:\n"
#             "  1) If the exact text is available, quote up to 25 words of the exact line in quotation marks.\n"
#             "  2) Immediately follow with a citation in this precise format: (Document Title, Section/Clause X; YYYY).\n"
#             "  3) If a reliable URL is available, include it in the source metadata only (do not place raw URLs in user-facing prose unless asked).\n"
#             "  4) Set the boolean `used_document` according to whether the user's organization document was used (true) or an external law/source was used (false).\n"
#             "  5) If the exact text cannot be found, state: 'Exact clause text not provided / not found.' and offer to retrieve and verify sources.\n\n"

#             "🔍 **CONTEXT USAGE RULES (UPDATED):**\n"
#             "- LAW QUESTIONS: Use Australian Law Context, set used_document=false unless the user supplied a legal document.\n"
#             "- POLICY QUESTIONS: Use organization context, set used_document=true when using the organization doc.\n"
#             "- NEVER mix organization context with law questions unless explicitly instructed and cite both clearly.\n\n"

#             "🌍 **MULTILINGUAL SUPPORT:**\n"
#             "- ALWAYS respond in the SAME language the user asks in\n"
#             "... (keep the original multilingual lines from previous prompt) ...\n\n"

#             "📋 **RESPONSE FORMAT (STRICT JSON):**\n"
#             "You MUST return ONLY this JSON format - no other text, no markdown, no code blocks:\n"
#             "{\n"
#             '  \"answer\": \"Your natural, conversational response here with proper opening and closing\", \n'
#             '  \"used_document\": true_or_false, \n'
#             '  \"sources\": [\n'
#             '      { \"title\": \"Document Title or Law\", \"section\": \"Section X\", \"quote\": \"Up to 25-word quote (if used)\", \"meta\": \"(year or other metadata)\" }\n'
#             '  ]\n'
#             "}\n\n"

#             "🚫 **FORBIDDEN:**\n"
#             "- No markdown formatting (**, ##, etc.)\n"
#             "- No code blocks or triple backticks in user-facing output\n"
#             "- No nested JSON in answer field\n"
#             "- No technical formatting - just natural conversation\n"
#             "- No 'As an AI assistant' disclaimers\n\n"

#             "✅ **GOOD RESPONSE EXAMPLE (UPDATED):**\n"
#             "{\n"
#             "  \"answer\": \"Sure! That\\'s a great question about aged care regulations. Let me explain how this works in Australia...\\n\\nBased on the Aged Care Act 1997, here are the key requirements...\\n\\nStep-by-step SOP: 1) Check eligibility; 2) Gather documents; 3) Submit application...\\n\\nHope this helps! Let me know if you want the exact clause text or links.\",\n"
#             "  \"used_document\": false,\n"
#             "  \"sources\": [ { \"title\": \"Aged Care Act 1997\", \"section\": \"S.12A\", \"quote\": \"[exact 25-word quote if used]\", \"meta\": \"1997\" } ]\n"
#             "}\n"
#         )



        
#         messages = [{"role": "system", "content": system_prompt}]
#         messages.extend(chat_history)
#         formatted_content = ""
        
#         # Add organization context if available
#         if org_context:
#             formatted_content += "ORGANIZATION CONTEXT:\n"
#             for i, doc in enumerate(org_context, 1):
#                 title = doc.properties.get('title', 'Unknown Title')
#                 version_number = doc.properties.get('version_number', 1)
#                 document_id = doc.properties.get('document_id', 'Unknown ID')
#                 data = doc.properties.get('data', '')
                
#                 formatted_content += f"[Org-{i}] {title} v{version_number} [{document_id}]\n"
#                 formatted_content += f"{data}\n\n"
        
#         # Add law context if available
#         if law_context:
#             formatted_content += "AUSTRALIAN LAW CONTEXT:\n"
#             for i, doc in enumerate(law_context, 1):
#                 title = doc.properties.get('title', 'Unknown Title')
#                 version_number = doc.properties.get('version_number', 1)
#                 document_id = doc.properties.get('document_id', 'Unknown ID')
#                 data = doc.properties.get('data', '')
                
#                 formatted_content += f"[Law-{i}] {title} v{version_number} [{document_id}]\n"
#                 formatted_content += f"{data}\n\n"
        
#         # Determine context status for clear instruction
#         context_status = ""
#         if org_context and law_context:
#             context_status = "Both organization and law context available."
#         elif org_context:
#             context_status = "Only organization context available."
#         elif law_context:
#             context_status = "Only Australian law context available."
#         else:
#             context_status = "No specific context available - use general knowledge with disclaimer."
        
#         # Determine if this is a law question
#         law_keywords = ['law', 'act', 'legislation', 'legal', 'aged care act', 'ndis act', 'regulation']
#         is_law_question = any(keyword in question.lower() for keyword in law_keywords)
        
#         question_type = "LAW QUESTION" if is_law_question else "POLICY QUESTION"
        
#         user_content = f"{formatted_content}QUESTION: {question}\n\nQUESTION TYPE: {question_type}\n\nCONTEXT STATUS: {context_status}\n\nCRITICAL INSTRUCTIONS:\n- This is a {question_type}\n- If LAW QUESTION: Use ONLY Australian Law Context, IGNORE organization context, set used_document=false\n- If POLICY QUESTION: Use organization context, set used_document=true\n- NEVER mix organization context with law questions"
#         messages.append({"role": "user", "content": user_content})
        
#         print("⏱️ Starting LLM call...")
#         llm_start = asyncio.get_event_loop().time()
        
#         try:
#             async with AsyncOpenAI(api_key=OPENAI_API_KEY) as openai_client:
#                 response = await openai_client.chat.completions.create(
#                     model="gpt-4",
#                     messages=messages,
#                     temperature=0.3
#                 )
#         except openai.APIError as e:
#             print(f"❌ OpenAI API error: {e}")
#             return JSONResponse(status_code=503, content={
#                 "status": "error",
#                 "message": "AI service temporarily unavailable",
#                 "error": str(e)
#             })
#         except openai.RateLimitError as e:
#             print(f"❌ OpenAI rate limit exceeded: {e}")
#             return JSONResponse(status_code=429, content={
#                 "status": "error",
#                 "message": "Too many requests. Please try again later.",
#                 "error": str(e)
#             })
#         except Exception as e:
#             print(f"❌ LLM call failed: {e}")
#             return JSONResponse(status_code=500, content={
#                 "status": "error",
#                 "message": "Failed to generate response",
#                 "error": str(e)
#             })
        
#         llm_time = asyncio.get_event_loop().time() - llm_start
#         print(f"✅ LLM call completed in {llm_time:.2f}s")
        
#         used_tokens = response.usage.total_tokens
#         answer = response.choices[0].message.content.strip()
#         print("🤖 Raw LLM answer:", answer)
        
#         try:
#             # Decode HTML entities and try direct JSON parsing
#             import html
#             decoded_answer = html.unescape(answer)
#             json_answer = json.loads(decoded_answer)
#         except json.JSONDecodeError:
#             try:
#                 # Try extracting JSON from LLM response
#                 json_answer = extract_json_from_llm(answer)
#                 if isinstance(json_answer, str):
#                     json_answer = json.loads(html.unescape(json_answer))
#             except Exception as e:
#                 print(f"⚠️ JSON parsing failed, using fallback: {e}")
#                 # Determine used_document based on question type
#                 law_keywords = ['law', 'act', 'legislation', 'legal', 'aged care act', 'ndis act', 'regulation']
#                 is_law_question = any(keyword in question.lower() for keyword in law_keywords)
#                 json_answer = {
#                     "answer": answer,
#                     "used_document": not is_law_question and bool(org_context)
#                 }
        
#         print('✅ Bot answer after LLM parser:', json_answer)
        
#         # ============ STEP 3: PARALLEL SAVE (Non-blocking) ============
#         print("⏱️ Starting parallel save...")
#         save_start = asyncio.get_event_loop().time()
        
#         readcount_data = {}
#         if json_answer.get('used_document', False) and org_context:
#             for c in org_context:
#                 doc_id = c.properties.get("document_id", "")
#                 if doc_id:
#                     readcount_data[doc_id] = 1
#             print("📄 Organization document IDs to update:", readcount_data)
        
#         # Debug: Check if used_document is incorrectly set for law questions
#         law_keywords = ['law', 'act', 'legislation', 'legal', 'aged care act', 'ndis act', 'regulation']
#         is_law_question = any(keyword in question.lower() for keyword in law_keywords)
#         if is_law_question and json_answer.get('used_document', False):
#             print("⚠️ WARNING: Law question incorrectly set used_document=true")
        
#         history_data = {
#             "prompt": question,
#             "response": json_answer['answer'],
#             "used_tokens": used_tokens
#         }
        
#         token_data = {"used_tokens": used_tokens}
        
#         # Save in background (don't block response)
#         try:
#             save_results = await save_data_parallel(
#                 history_data, readcount_data, token_data, auth_token
#             )
            
#             save_time = asyncio.get_event_loop().time() - save_start
#             print(f"✅ Parallel save completed in {save_time:.2f}s")
            
#             # Log warnings but don't fail the request
#             for result in save_results:
#                 if isinstance(result, dict) and not result.get('success', False) and not result.get('skipped', False):
#                     print(f"⚠️ Warning: {result['type']} save failed with status {result.get('status', 'unknown')}")
        
#         except Exception as e:
#             # Log but don't fail the request
#             print(f"⚠️ Background save failed: {e}")
#             save_time = asyncio.get_event_loop().time() - save_start
        
#         total_time = asyncio.get_event_loop().time() - start_time
#         print(f"🎯 Total request time: {total_time:.2f}s")
        
#         return JSONResponse(status_code=200, content={
#             "status": "success",
#             "question": question,
#             "answer": json_answer['answer'],
#             "used_tokens": used_tokens,
#             "performance": {
#                 "fetch_time": f"{fetch_time:.2f}s",
#                 "llm_time": f"{llm_time:.2f}s",
#                 "save_time": f"{save_time:.2f}s",
#                 "total_time": f"{total_time:.2f}s"
#             }
#         })
    
#     except Exception as e:
#         print(f"❌ Unexpected error in ask_doc_bot: {str(e)}")
#         return JSONResponse(status_code=500, content={
#             "status": "error",
#             "message": "An unexpected error occurred",
#             "error": str(e)
#         })

