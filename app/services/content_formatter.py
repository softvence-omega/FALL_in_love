


"""
MINIMAL Content Formatter - Maximum Token Savings
Your system prompt already has ALL instructions, so user message just needs:
1. Type flag
2. Context
3. Question
That's it!
"""

async def formatted_content(question_type: str, org_context: list, law_context: list) -> str:
    """
    Ultra-minimal formatting - just essential content
    System prompt has all the instructions already
    """
    parts = []
    
    # Organization docs (if relevant)
    if question_type in ["POLICY", "MIXED"] and org_context:
        parts.append("ORG:")
        for i, doc in enumerate(org_context, 1):
            data = doc.properties.get('data', '').strip()
            parts.append(f"{i}. {data}")
    
    # Law docs (if relevant)
    if question_type in ["LAW", "MIXED"] and law_context:
        parts.append("LAW:")
        for i, doc in enumerate(law_context, 1):
            data = doc.properties.get('data', '').strip()
            parts.append(f"{i}. {data}")
    
    return "\n\n".join(parts) if parts else ""


async def build_user_message_with_context(
    question: str,
    question_type: str,
    org_context: list,
    law_context: list,
    formatted_content_str: str,
    previous_responses: list = None  # ← নতুন parameter
) -> str:
    """Add anti-repetition instruction dynamically"""
    
    org_flag = "Y" if org_context else "N"
    law_flag = "Y" if law_context else "N"
    header = f"[{question_type}|Org:{org_flag}|Law:{law_flag}]"
    
    parts = [header]
    
    # ✅ ADD THIS: যদি recent responses থাকে
    if previous_responses and len(previous_responses) >= 2:
        # শেষ 2টা response এর শুরুর line extract করা
        recent_openings = []
        for resp in previous_responses[-2:]:
            first_line = resp.split('\n')[0][:50]  # প্রথম 50 chars
            recent_openings.append(first_line)
        
        # LLM কে instruction দেওয়া
        anti_repeat = f"[Vary: Don't start with: {', '.join(recent_openings)}]"
        parts.append(anti_repeat)
    
    if formatted_content_str:
        parts.append(formatted_content_str)
    
    parts.append(f"Q: {question}")
    
    return "\n\n".join(parts)

# Build minimal history to reduce token cost
def build_minimal_history(histories: list, max_pairs: int = 2) -> list:
    """
    AGGRESSIVE: Only keep last 2 exchanges + topic summary
    
    Best for:
    - Maximum cache hit rate
    - Cost-sensitive applications
    - When conversation context is less critical
    """
    chat_history = []
    
    if not histories:
        return chat_history
    
    # Keep only last N exchanges
    recent_histories = histories[-max_pairs:] if len(histories) > max_pairs else histories
    
    # If there are older messages, add minimal summary
    if len(histories) > max_pairs:
        older_count = len(histories) - max_pairs
        chat_history.append({
            "role": "user",
            "content": f"[Previous: {older_count} earlier exchanges about aged care topics]"
        })
        chat_history.append({
            "role": "assistant",
            "content": "Understood."
        })
    
    # Add recent exchanges
    for h in recent_histories:
        chat_history.append({"role": "user", "content": h['prompt']})
        chat_history.append({"role": "assistant", "content": h['response']})
    
    return chat_history

# async def formatted_content(question_type, org_context, law_context) -> str:
#     """Format context with clear status indicators for document/law availability"""
#     formatted_content = ""
    
#     # Document status indicators
#     has_org_docs = bool(org_context)
#     has_law_docs = bool(law_context)
    
#     if question_type == "POLICY":
#         if has_org_docs:
#             formatted_content += "ORGANIZATION CONTEXT FOUND:\n"
#             for i, doc in enumerate(org_context, 1):
#                 title = doc.properties.get('title', 'Unknown')
#                 data = doc.properties.get('data', '')
#                 formatted_content += f"[Org-{i}] {title}\n{data}\n\n"
#         else:
#             formatted_content += "NO ORGANIZATION CONTEXT FOUND\n"
#             formatted_content += "INSTRUCTION: You MUST disclose this in your response and provide general Australian aged care best practices instead.\n\n"
    
#     elif question_type == "LAW":
#         if has_law_docs:
#             formatted_content += "AUSTRALIAN LAW CONTEXT FOUND:\n"
#             for i, doc in enumerate(law_context, 1):
#                 title = doc.properties.get('title', 'Unknown')
#                 data = doc.properties.get('data', '')
#                 formatted_content += f"[Law-{i}] {title}\n{data}\n\n"
#         else:
#             formatted_content += "NO SPECIFIC LAW CONTEXT FOUND\n"
#             formatted_content += "INSTRUCTION: You MUST disclose this in your response and provide general Australian aged care regulatory framework guidance.\n\n"
    
#     elif question_type == "MIXED":
#         # Organization context
#         if has_org_docs:
#             formatted_content += "=== ORGANIZATION CONTEXT FOUND ===\n"
#             for i, doc in enumerate(org_context, 1):
#                 title = doc.properties.get('title', 'Unknown')
#                 data = doc.properties.get('data', '')
#                 formatted_content += f"[Org-{i}] {title}\n{data}\n\n"
#         else:
#             formatted_content += "=== NO ORGANIZATION CONTEXT FOUND ===\n"
#             formatted_content += "INSTRUCTION: Disclose missing org documents and use general best practices based on australian context.\n\n"
        
#         # Law context
#         if has_law_docs:
#             formatted_content += "=== AUSTRALIAN LAW CONTEXT FOUND ===\n"
#             for i, doc in enumerate(law_context, 1):
#                 title = doc.properties.get('title', 'Unknown')
#                 data = doc.properties.get('data', '')
#                 formatted_content += f"[Law-{i}] {title}\n{data}\n\n"
#         else:
#             formatted_content += "=== NO SPECIFIC LAW CONTEXT FOUND ===\n"
#             formatted_content += "INSTRUCTION: Disclose missing specific legislation and use general regulatory framework based on australian context.\n\n"
    
#     # Add status summary for LLM to handle dynamically
#     formatted_content += f"STATUS SUMMARY:\n"
#     formatted_content += f"- Organization documents: {'FOUND' if has_org_docs else 'NOT FOUND'}\n"
#     formatted_content += f"- Specific laws/acts: {'FOUND' if has_law_docs else 'NOT FOUND'}\n"
#     formatted_content += f"- Question type: {question_type}\n\n"
    
#     return formatted_content
