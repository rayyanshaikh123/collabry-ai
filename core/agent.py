# core/agent.py
"""
Collabry Study Copilot Agent - Pedagogical AI learning assistant.

This agent is optimized for helping students learn through:
- Step-by-step explanations with examples and analogies
- Concept extraction and structured learning
- Follow-up question generation for active learning
- Clarifying questions when context is unclear
- Source citation (no hallucination)

Flow:
- Run local NLP pipeline (spell-correct, intent, NER) to enrich prompt.
- Check if clarification needed (Study Copilot enhancement)
- Ask LLM to decide: either call a tool or answer directly.
- LLM returns JSON: {"tool": "<name>", "args": {...}} OR {"tool": null, "answer": "...", "follow_up_questions": [...]}
- If tool selected: execute tool, synthesize pedagogical answer
- Add follow-up questions to encourage deeper learning
- Streaming: prints natural-language tokens as they arrive
"""

import json
import logging
import inspect
import time
from typing import Any, Dict, Optional, List

from tools import load_tools
from core.local_llm import LocalLLM, create_llm
from core.nlp import analyze
from core.memory import MemoryManager
from core.rag_retriever import RAGRetriever
from core.study_copilot import StudyCopilot
from core.prompt_templates import SYSTEM_PROMPT, USER_INSTRUCTION
from config import CONFIG
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# -------------------------
# Helpers
# -------------------------
def extract_json(text: str) -> Optional[dict]:
    """Extract first valid JSON object from text, return parsed dict or None."""
    if not text:
        return None
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to locate first {...} block
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        sub = text[s : e + 1]
        try:
            return json.loads(sub)
        except Exception:
            # try to remove trailing non-json content and reparse
            # as a last-ditch attempt, try to replace newlines and stray quotes
            try:
                return json.loads(sub.strip())
            except Exception:
                return None
    return None


def clean_answer(ans: Any) -> str:
    """Return a human-friendly string for the answer.
    If ans is dict -> pretty JSON string; if it's an escaped JSON string -> unescape.
    Otherwise return str(ans).
    """
    if ans is None:
        return ""
    if isinstance(ans, dict):
        try:
            return json.dumps(ans, indent=2, ensure_ascii=False)
        except Exception:
            return str(ans)
    if isinstance(ans, str):
        s = ans.strip()
        # If ans is an escaped JSON string like "{\"price\": 123}"
        if s.startswith("{") and s.endswith("}"):
            try:
                inner = json.loads(s)
                return json.dumps(inner, indent=2, ensure_ascii=False)
            except Exception:
                return s
        return s
    return str(ans)


def _typing_print(text: str, speed_ms: int = 30):
    """Typing-style print with per-character delay."""
    try:
        for ch in text:
            print(ch, end="", flush=True)
            time.sleep(speed_ms / 1000.0)
        print()
    except Exception:
        # fallback to simple print on any problem
        print(text)

# -------------------------
# Safe tool execution
# -------------------------
def safe_execute_tool(tool_entry, args: Optional[Dict[str, Any]], llm: LocalLLM):
    """
    Execute a tool safely.
    Supports:
      - tool_entry is a callable
      - tool_entry is a dict { "name":..., "func": callable, ... }
    Only passes parameters that the tool function accepts.
    If the tool supports an 'llm' parameter, we inject it.
    Returns tool result (any python object) or {"error": "..."} dict.
    """
    if args is None:
        args = {}

    target = tool_entry
    if isinstance(tool_entry, dict):
        target = tool_entry.get("func") or tool_entry.get("callable")
        if target is None:
            return {"error": "Invalid tool entry: missing 'func' key."}

    if not callable(target):
        return {"error": "Tool target is not callable."}

    sig = inspect.signature(target)
    accepted = {}

    # Basic argument adaptation layer for common mistakes (LLM robustness)
    # Map single 'content' or 'text' field into 'contents' if function expects 'contents'.
    if "contents" in sig.parameters and "contents" not in args:
        if "content" in args:
            args["contents"] = args.pop("content")
        elif "text" in args:
            args["contents"] = args.pop("text")
        elif "data" in args:
            args["contents"] = args.pop("data")

    # Provide a default path if write_file-like tool missing 'path'
    if "path" in sig.parameters and "path" not in args:
        # Use a safe default inside workspace
        args["path"] = "output.txt"

    for p in sig.parameters.values():
        if p.name in args:
            accepted[p.name] = args[p.name]
    if "llm" in sig.parameters:
        accepted["llm"] = llm

    try:
        return target(**accepted)
    except Exception as e:
        logger.exception("Tool execution failed")
        return {"error": str(e)}

# -------------------------
# Collabry Study Copilot Agent
# -------------------------
class COLLABRYAgent:
    """
    Collabry Study Copilot Agent.
    
    Pedagogical AI assistant that helps students learn through:
    - Step-by-step explanations
    - Examples and analogies
    - Concept extraction
    - Follow-up question generation
    - Clarifying questions
    """
    
    def __init__(
        self,
        llm: LocalLLM,
        tools: Dict[str, Any],
        memory: MemoryManager,
        rag_retriever: RAGRetriever,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize Study Copilot Agent.
        
        Args:
            llm: Language model for generation
            tools: Available tools for actions
            memory: Conversation memory manager
            rag_retriever: RAG retriever for user documents
            user_id: User identifier for isolation
            session_id: Session/notebook identifier for isolation
        """
        self.llm = llm
        self.tools = tools or {}
        self.memory = memory
        self.rag_retriever = rag_retriever
        self.study_copilot = StudyCopilot(llm)
        self.last_tool_called: Optional[str] = None
        self.user_id = user_id
        self.session_id = session_id

    def _build_instruction(
        self,
        user: str,
        chat_history: str,
        retrieved_docs: List[Document],
        entities: List[tuple],
        intent: Optional[str] = None
    ) -> str:
        """
        Construct pedagogical instruction prompt for the LLM.
        
        Args:
            user: User input
            chat_history: Previous conversation
            retrieved_docs: Retrieved documents from RAG
            entities: Named entities detected
            intent: Detected intent (optional)
            
        Returns:
            Complete prompt for LLM
        """
        # Build tool list with descriptions
        tool_lines = []
        for name, t in self.tools.items():
            func = None
            if isinstance(t, dict):
                func = t.get("func") or t.get("callable")
                desc = t.get("description", "")
            else:
                func = t
                try:
                    desc = ((t.__doc__ or "").strip().split("\n")[0])
                except Exception:
                    desc = ""

            sig_str = ""
            if callable(func):
                try:
                    params = []
                    for p in inspect.signature(func).parameters.values():
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                            params.append(p.name)
                    if params:
                        sig_str = f"({', '.join(params)})"
                except Exception:
                    sig_str = ""

            tool_lines.append(f"- {name}{sig_str}: {desc}")
        tool_list_txt = "\n".join(tool_lines) if tool_lines else "(no tools available)"

        # Format retrieved documents with source citation
        retrieved_context = ""
        if retrieved_docs:
            doc_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get('source', 'N/A')
                content = doc.page_content[:500]  # Limit length
                doc_parts.append(f"[Source {i}: {source}]\n{content}")
            retrieved_context = "\n\n".join(doc_parts)

        # Use Study Copilot system prompt
        system = SYSTEM_PROMPT

        # Add tool list to user instruction
        user_instr = USER_INSTRUCTION.replace("{{tool_list}}", tool_list_txt)

        # Build entity context
        entity_block = ""
        if entities:
            ent_lines = [f"- {text} ({label})" for text, label in entities]
            entity_block = "\n".join(ent_lines)

        # Construct complete prompt
        prompt_parts = [system, user_instr]
        
        if entity_block:
            prompt_parts.append(f"\nNamed Entities Detected:\n{entity_block}")
        
        if intent and self.study_copilot.is_study_intent(intent):
            prompt_parts.append(f"\n🎓 Study Intent Detected: {intent}")
        
        if retrieved_context:
            prompt_parts.append(
                f"\n{'='*70}\n"
                f"📚 RETRIEVED CONTEXT FROM USER'S DOCUMENTS\n"
                f"{'='*70}\n"
                f"{retrieved_context}\n"
                f"{'='*70}\n\n"
                f"🚨 CRITICAL INSTRUCTIONS - READ CAREFULLY:\n"
                f"1. The context above is from the user's actual documents\n"
                f"2. You MUST answer using ONLY the information from this context\n"
                f"3. DO NOT use ANY information from your training data or general knowledge\n"
                f"4. If the question cannot be answered from the context above, say: 'This information is not found in your documents.'\n"
                f"5. Always cite which source you're using: 'According to Source 1...' or 'Based on Source 2...'\n"
                f"6. If you find yourself about to mention COLLABRY, Rayyan, or general AI concepts NOT in the context above, STOP and say the info is not available\n"
                f"{'='*70}\n"
            )
        
        if chat_history:
            prompt_parts.append(f"\n💬 Chat History:\n{chat_history}")
        
        prompt_parts.append(f"\n👤 Student: {user}")
        prompt_parts.append("\n🤖 Collabry Study Copilot:")
        
        return "\n".join(prompt_parts)

    def handle_user_input_stream(self, user_input: str, on_token, source_ids: Optional[List[str]] = None):
        """
        Main entry point for handling user input with Study Copilot enhancements.
        
        Args:
            user_input: Student's question or input
            on_token: Callback for streaming tokens
            source_ids: Optional list of source IDs to filter RAG retrieval
        """
        # 1. NLP analysis (spell correction, intent, entities)
        # Skip NLP for very long texts (quiz generation with large documents)
        try:
            if len(user_input) > 1000000:  # Skip NLP for texts > 1MB
                logger.info(f"Skipping NLP analysis for long text ({len(user_input)} chars)")
                analysis = {"corrected": user_input}
            else:
                analysis = analyze(user_input)
        except Exception as e:
            logger.exception("NLP analyze failed")
            analysis = {"corrected": user_input}
        
        corrected = analysis.get("corrected", user_input)
        intent = analysis.get("intent")
        entities = analysis.get("entities", []) or []
        
        # 2. Study Copilot: Check if clarification needed
        clarification = self.study_copilot.needs_clarification(corrected)
        if clarification:
            response = f"🤔 {clarification}\n\n(This will help me give you a better explanation!)"
            on_token(response)  # Use callback instead of print
            self.memory.save_context({"user_input": corrected}, {"output": response})
            return

        # 3. Retrieve relevant documents (RAG) - cite sources in answer
        # Get user's documents across all sessions for better context
        retrieved_docs = self.rag_retriever.get_relevant_documents(
            corrected, 
            user_id=self.user_id
            # source_ids=source_ids  # Temporarily disabled to allow access to all user documents
        )
        
        # 3.5. Safety check: If source_ids were requested but no docs found, warn user
        if source_ids and len(source_ids) > 0 and len(retrieved_docs) == 0:
            error_msg = (
                "⚠️ **No documents found for your selected sources.**\n\n"
                "This might happen if:\n"
                "- The sources were uploaded before the latest update\n"
                "- The documents haven't been processed yet\n\n"
                "**Solution:** Please delete and re-upload your sources to fix this issue.\n\n"
                "For now, I'll answer using your general knowledge since no specific documents were found."
            )
            on_token(error_msg)
            self.memory.save_context({"user_input": corrected}, {"output": error_msg})
            return

        # 4. Load conversational memory
        memory_vars = self.memory.load_memory_variables({"user_input": corrected})
        chat_history = self.memory.get_history_string()

        # 5. Build pedagogical prompt
        prompt = self._build_instruction(corrected, chat_history, retrieved_docs, entities, intent)

        # 6. Generate response from LLM
        raw_response = ""
        try:
            raw_response = self.llm.invoke(prompt)
        except Exception as e:
            logger.exception("LLM invocation failed")
            print(f"[LLM error: {e}]")
            return

        # 7. Parse JSON decision
        parsed_json = extract_json(raw_response)
        if not parsed_json:
            # Fallback for non-JSON response
            on_token(raw_response)  # Use callback instead of print
            self.memory.save_context({"user_input": corrected}, {"output": raw_response})
            return

        # 8. Handle list responses (e.g., quiz questions array)
        if isinstance(parsed_json, list):
            # For list responses, just stream the raw response
            on_token(raw_response)
            self.memory.save_context({"user_input": corrected}, {"output": raw_response})
            return

        # Special handling for COURSE_FINDER_REQUEST: force web_search tool call
        if "[COURSE_FINDER_REQUEST]" in corrected:
            parsed_json = {"tool": "web_search", "args": {"query": "best courses Arrays in Programming online course tutorial"}}

        # 9. Handle direct answer (no tool)
        if parsed_json.get("tool") is None:
            answer = clean_answer(parsed_json.get("answer", ""))
            
            # Special handling for mindmap generation requests
            # If the user input contains mindmap generation marker, output the answer directly (it should be JSON)
            if "[MINDMAP_GENERATION_REQUEST]" in corrected or "[MINDMAP_GENERATION_REQUEST]" in user_input:
                # For mindmap, the answer should be raw JSON - output it directly without formatting
                on_token(answer)  # Output the raw JSON answer
                self.memory.save_context({"user_input": corrected}, {"output": answer})
                return
            
            follow_ups = parsed_json.get("follow_up_questions", [])
            
            # If no follow-ups provided, generate them
            if not follow_ups:
                follow_ups = self.study_copilot.generate_follow_up_questions(
                    corrected,
                    answer,
                    count=3
                )
            
            # Format response with follow-up questions
            full_response = answer
            if follow_ups:
                full_response += "\n\n📝 **Follow-up questions to deepen your understanding:**"
                for i, q in enumerate(follow_ups, 1):
                    full_response += f"\n{i}. {q}"
            
            # Add learning tip
            full_response += f"\n\n{self.study_copilot._get_learning_tip()}"
            
            on_token(full_response)  # Use callback instead of print
            self.memory.save_context({"user_input": corrected}, {"output": full_response})
            return
            
        # 9. Handle tool call path
        tool_name = parsed_json.get("tool")
        args = parsed_json.get("args", {}) or {}

        synonyms = CONFIG.get("tool_synonyms", {})
        canonical = synonyms.get(tool_name, tool_name)
        
        if canonical not in self.tools:
            msg = f"❌ Tool '{tool_name}' is not available.\n\n"
            suggestions = self.study_copilot.suggest_study_tools(corrected)
            if suggestions:
                msg += "💡 **Suggested tools:**\n"
                for sug in suggestions:
                    msg += f"- {sug['tool']}: {sug['reason']}\n"
            on_token(msg)  # Use callback instead of print
            self.memory.save_context({"user_input": corrected}, {"output": msg})
            return

        # Execute tool
        tool_entry = self.tools[canonical]
        print(f"[Tool Invocation] {canonical} args={args}")
        result = safe_execute_tool(tool_entry, args, self.llm)
        self.last_tool_called = canonical

        # Special handling for COURSE_FINDER_REQUEST
        if "[COURSE_FINDER_REQUEST]" in corrected and canonical == "web_search":
            if isinstance(result, dict) and "results" in result:
                courses = []
                for r in result["results"][:8]:  # up to 8 courses
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "")
                    platform = "Unknown"
                    rating = ""
                    price = ""
                    url_lower = url.lower()
                    if "udemy" in url_lower:
                        platform = "Udemy"
                    elif "coursera" in url_lower:
                        platform = "Coursera"
                    elif "edx" in url_lower:
                        platform = "edX"
                    elif "codecademy" in url_lower:
                        platform = "Codecademy"
                    # Simple rating extraction
                    import re
                    match = re.search(r'(\d\.\d)/5', snippet)
                    if match:
                        rating = match.group(1) + "/5"
                    # Simple price extraction
                    if "$" in snippet:
                        price_match = re.search(r'\$[\d]+', snippet)
                        if price_match:
                            price = price_match.group(0)
                    elif "free" in snippet.lower():
                        price = "Free"
                    course_line = f"[{title}]({url}) - Platform: {platform} | Rating: {rating} | Price: {price}"
                    courses.append(course_line)
                course_list = "\n".join(courses)
                json_output = f'{{"tool": null, "answer": "{course_list}"}}'
                on_token(json_output)
                self.memory.save_context({"user_input": corrected}, {"output": json_output})
                return
            else:
                # Fallback to short_answer
                answer = clean_answer(result.get("short_answer", ""))
                json_output = f'{{"tool": null, "answer": "{answer.replace(chr(34), chr(92) + chr(34))}"}}'
                on_token(json_output)
                self.memory.save_context({"user_input": corrected}, {"output": json_output})
                return

        # 10. Synthesize pedagogical answer from tool result
        immediate = None
        full_content = None
        
        if isinstance(result, str) and not result.startswith('{'):
            # Simple action tool with direct message
            immediate = result
        elif isinstance(result, dict):
            # Extract relevant fields
            immediate = result.get("short_answer") or result.get("snippet") or result.get("summary")
            
            # Store full content for web_scrape
            if canonical == "web_scrape":
                full_content = result.get("full_text") or result.get("text")
        
        if immediate:
            # Synthesize pedagogical response
            if "[COURSE_FINDER_REQUEST]" in corrected and canonical == "web_search":
                # Special prompt for course finder to enforce exact JSON format
                follow_prompt = (
                    f"Information found:\n\n{clean_answer(immediate)}\n\n"
                    f"Question: {corrected}\n\n"
                    "Extract 5-8 online courses about arrays in programming from the information above.\n"
                    "For each course, provide:\n"
                    "- Title\n"
                    "- URL (direct course link)\n"
                    "- Platform (e.g., Udemy, Coursera)\n"
                    "- Rating (e.g., 4.5/5)\n"
                    "- Price (e.g., $10 or Free)\n\n"
                    "Format each as: [Title](URL) - Platform: P | Rating: R | Price: $P\n"
                    "List them one per line in <COURSE_LIST>.\n\n"
                    "Output only: {\"tool\": null, \"answer\": \"<COURSE_LIST>\"}"
                )
            else:
                follow_prompt = (
                    f"The {canonical} tool returned:\n\n{clean_answer(immediate)}\n\n"
                    f"Student's question: {corrected}\n\n"
                    "Create a pedagogical response that:\n"
                    "1. Answers the student's specific question\n"
                    "2. Explains step-by-step if applicable\n"
                    "3. Cites sources (mention the tool used)\n"
                    "4. Never hallucinate - only use information from the tool result\n\n"
                    "Return JSON with answer and optional follow_up_questions:\n"
                    '{"tool": null, "answer": "<pedagogical response>", "follow_up_questions": ["Q1", "Q2", "Q3"]}'
                )
            
            try:
                final_raw = self.llm.invoke(follow_prompt)
                final_json = extract_json(final_raw)
            except Exception:
                final_json = None
            
            if final_json and final_json.get("tool") is None:
                final_answer = clean_answer(final_json.get("answer", ""))
                follow_ups = final_json.get("follow_up_questions", [])
                
                # Generate follow-ups if not provided
                if not follow_ups:
                    follow_ups = self.study_copilot.generate_follow_up_questions(
                        corrected,
                        final_answer,
                        count=3
                    )
                
                # Format response
                full_response = final_answer
                if follow_ups:
                    full_response += "\n\n📝 **Follow-up questions:**"
                    for i, q in enumerate(follow_ups, 1):
                        full_response += f"\n{i}. {q}"
                
                on_token(full_response)  # Use callback instead of print
                
                # Store with tool tag
                if full_content:
                    tagged_answer = f"[tool:{canonical}] {full_response}\n[CONTENT]\n{full_content[:2000]}..."
                else:
                    tagged_answer = f"[tool:{canonical}] {full_response}"
                
                self.memory.save_context({"user_input": corrected}, {"output": tagged_answer})
                return
            else:
                # Fallback: print immediate result
                on_token(clean_answer(immediate))  # Use callback instead of print
                self.memory.save_context(
                    {"user_input": corrected},
                    {"output": f"[tool:{canonical}] {clean_answer(immediate)}"}
                )
                return
        
        # 11. Synthesize from full tool result if no immediate answer
        follow_prompt = (
            f"The {canonical} tool returned:\n\n{clean_answer(result)}\n\n"
            f"Student's question: {corrected}\n\n"
            "Create a helpful, educational response that:\n"
            "1. Extracts the specific information requested\n"
            "2. Explains in a way that helps learning\n"
            "3. Cites the tool as the source\n"
            "4. Only uses information from the tool result (no hallucination)\n\n"
            "Return JSON:\n"
            '{"tool": null, "answer": "<response>"}'
        )
        
        try:
            final_raw = self.llm.invoke(follow_prompt)
            final_json = extract_json(final_raw)
        except Exception as e:
            logger.exception("LLM synthesis failed")
            out = clean_answer(result)
            on_token(out)  # Use callback instead of print
            self.memory.save_context({"user_input": corrected}, {"output": f"[tool:{canonical}] {out}"})
            return

        if final_json and isinstance(final_json, dict):
            final_answer = clean_answer(final_json.get("answer", ""))
        else:
            final_answer = clean_answer(result)
        
        on_token(final_answer)  # Use callback instead of print
        
        # Store with full content if web_scrape
        if full_content:
            tagged_answer = f"[tool:{canonical}] {final_answer}\n[CONTENT]\n{full_content[:2000]}..."
        else:
            tagged_answer = f"[tool:{canonical}] {final_answer}"
        
        self.memory.save_context({"user_input": corrected}, {"output": tagged_answer})


# -------------------------
# Factory function
# -------------------------
def create_agent(
    user_id: str,
    session_id: str,
    config: Optional[dict] = None
):
    """
    Create a Study Copilot agent instance with multi-user isolation.
    
    The Study Copilot is optimized for helping students learn through:
    - Step-by-step explanations with examples
    - Concept extraction and structured learning
    - Follow-up question generation
    - Clarifying questions when context is unclear
    - Source citation (no hallucination)
    
    Args:
        user_id: User identifier (from JWT token)
        session_id: Session identifier for this chat
        config: Optional configuration override
        
    Returns:
        Tuple of (agent, llm, tools, memory) with user context initialized
    """
    cfg = config or CONFIG
    llm = create_llm(cfg)
    tools = load_tools()
    
    # User-isolated memory with thread_id format: "user_id:session_id"
    memory = MemoryManager(user_id=user_id, session_id=session_id, llm=llm)
    
    # User-isolated RAG retriever (filters documents by user_id)
    rag_retriever = RAGRetriever(cfg, user_id=user_id)
    
    # Create Study Copilot agent with user and session context
    agent = COLLABRYAgent(
        llm, 
        tools, 
        memory, 
        rag_retriever,
        user_id=user_id,
        session_id=session_id
    )
    
    logger.info(
        f"✓ Study Copilot agent created for user_id={user_id}, session_id={session_id}"
    )
    return agent, llm, tools, memory

