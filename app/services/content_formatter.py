

async def formatted_content(question_type, org_context, law_context) -> str:
    """Format context based on question type with proper citation (title + version)"""
    print("law_context====================================\n", law_context)
    print("org_context====================================\n", org_context)

    formatted_content = ""
    
    def format_doc(doc, prefix: str) -> str:
        """Return string with proper citation [Ref: title - version]"""
        title = doc.properties.get('title', 'Unknown')
        version = doc.properties.get('version_number', '')
        data = doc.properties.get('data', '')
        citation = f"[Ref: {title} - {version}]" if title and version else ""
        print("citation====================================\n", citation)
        return f"{prefix} {title} {citation}\n{data}\n\n"

    if question_type == "POLICY":
        if org_context:
            formatted_content += "ORGANIZATION CONTEXT:\n"
            for i, doc in enumerate(org_context, 1):
                formatted_content += format_doc(doc, prefix="•")
        else:
            formatted_content += "NO ORGANIZATION CONTEXT AVAILABLE\n"
            formatted_content += "NOTE: No organizational documents have been uploaded yet. Provide general guidance with a disclaimer.\n\n"
    
    elif question_type == "LAW":
        if law_context:
            formatted_content += "AUSTRALIAN LAW CONTEXT:\n"
            for i, doc in enumerate(law_context, 1):
                formatted_content += format_doc(doc, prefix="•")
    
    elif question_type == "MIXED":
        if org_context:
            formatted_content += "=== ORGANIZATION CONTEXT ===\n"
            for i, doc in enumerate(org_context, 1):
                formatted_content += format_doc(doc, prefix="•")
        else:
            formatted_content += "=== NO ORGANIZATION CONTEXT ===\n"
            formatted_content += "NOTE: No organizational policies uploaded yet for this scenario.\n\n"
        
        if law_context:
            formatted_content += "=== AUSTRALIAN LAW CONTEXT ===\n"
            for i, doc in enumerate(law_context, 1):
                formatted_content += format_doc(doc, prefix="•")
        else:
            formatted_content += "=== NO LAW CONTEXT ===\n"
            formatted_content += "NOTE: No specific legislation found. Use general legal knowledge.\n\n"
    print("formatted_content====================================\n", formatted_content)
    return formatted_content



async def build_user_message_with_context(
    question: str,
    question_type: str,
    org_context: list,
    law_context: list,
    formatted_content_str: str,
    previous_responses: list = None,
    # response_instructions: str = None
) -> str:
    """Build user message with context, anti-repetition, and response instructions"""
    
    org_flag = "Y" if org_context else "N"
    law_flag = "Y" if law_context else "N"
    header = f"[{question_type}|Org:{org_flag}|Law:{law_flag}]"
    
    parts = [header]
    
    # Enhanced anti-repetition logic
    if previous_responses and len(previous_responses) >= 1:
        # Extract opening patterns from recent responses
        recent_patterns = []
        for resp in previous_responses[-3:]:  # Last 3 responses
            # Get first sentence or first 40 characters
            first_sentence = resp.split('.')[0][:40] if '.' in resp else resp.split('\n')[0][:40]
            if first_sentence.strip():
                recent_patterns.append(first_sentence.strip())
        
        if recent_patterns:
            # Create variation instruction
            anti_repeat = f"[VARY OPENING: Avoid starting with patterns like: {' | '.join(recent_patterns)}]"
            parts.append(anti_repeat)
    
    if formatted_content_str:
        parts.append(formatted_content_str)
    
    parts.append(f"Q: {question}")
    
    return "\n\n".join(parts)