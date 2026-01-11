# main.py
"""
Legacy CLI for local testing of Collabry AI Core Engine.

NOTE: This CLI uses hardcoded user_id/session_id for testing.
In production, user_id MUST come from JWT token validation (not client input).

Usage:
  python legacy_tools/main_cli.py [--user USER_ID] [--session SESSION_ID]
"""
import sys
from pathlib import Path

# Add parent directory (ai-engine) to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import re
import argparse

# Buffer used to accumulate streaming chunks so we can strip JSON objects
# that may be split across multiple chunks.
_stream_buffer = ""
from core.agent import create_agent
from config import CONFIG


def stream_printer(chunk: str):
    """Print streaming tokens from LLM."""
    try:
        global _stream_buffer

        # Append incoming chunk to buffer
        _stream_buffer += chunk

        # Helper: remove complete {...} blocks that contain our keywords
        def _remove_json_blocks(s: str):
            out_parts = []
            i = 0
            last_written = 0
            n = len(s)
            while i < n:
                if s[i] != '{':
                    i += 1
                    continue
                # Found a '{', try to find the matching '}' for a complete block
                depth = 0
                j = i
                complete = False
                while j < n:
                    if s[j] == '{':
                        depth += 1
                    elif s[j] == '}':
                        depth -= 1
                        if depth == 0:
                            complete = True
                            break
                    j += 1

                if not complete:
                    # Incomplete block: stop scanning; leave remainder in buffer
                    break

                # We have a complete block from i..j inclusive
                block = s[i : j + 1]
                # If block contains any of the JSON decision keywords, skip it
                if '"tool"' in block or '"args"' in block or '"answer"' in block:
                    # write everything before this block
                    out_parts.append(s[last_written:i])
                    last_written = j + 1
                # else: keep the block as normal text (it will be included below)

                i = j + 1

            # After scanning, include any text up to the last complete-removed index
            out = ''.join(out_parts) + s[last_written:]

            # Determine remainder: if there's an unmatched '{' at the end, keep it
            # as remainder; otherwise remainder is empty.
            # Find last unmatched '{' from end
            rem_start = None
            depth = 0
            for idx, ch in enumerate(s):
                pass
            # If the buffer contains an opening brace without a matching closing,
            # we keep the trailing incomplete segment starting at that brace.
            # Find the earliest position of an incomplete block by scanning for
            # a '{' that doesn't have a matching '}' later.
            k = 0
            n = len(s)
            while k < n:
                if s[k] == '{':
                    # try to find matching '}'
                    d = 0
                    m = k
                    matched = False
                    while m < n:
                        if s[m] == '{':
                            d += 1
                        elif s[m] == '}':
                            d -= 1
                            if d == 0:
                                matched = True
                                break
                        m += 1
                    if not matched:
                        rem_start = k
                        break
                    else:
                        k = m + 1
                else:
                    k += 1

            if rem_start is not None:
                printable = out[:rem_start]
                remainder = s[rem_start:]
            else:
                printable = out
                remainder = ''

            return printable, remainder

        printable, remainder = _remove_json_blocks(_stream_buffer)

        # Print what can be safely printed now
        if printable:
            if printable.strip():
                print(printable, end="", flush=True)

        # Keep remainder in buffer for next chunks
        _stream_buffer = remainder
    except Exception:
        print(chunk)


def detect_and_remove_wake_word(text: str, wake_words: list) -> tuple:
    """Detect wake word and remove it from input.
    
    Returns:
        (has_wake_word: bool, cleaned_text: str)
    """
    if not wake_words:
        return True, text
    
    lowered = text.lower().strip()
    
    # Sort wake words by length (longest first) to match more specific phrases first
    sorted_wake_words = sorted(wake_words, key=len, reverse=True)
    
    for wake_word in sorted_wake_words:
        wake_lower = wake_word.lower()
        
        # Check if wake word appears at start, middle, or end
        if wake_lower in lowered:
            # Find position and remove it
            idx = lowered.find(wake_lower)
            
            # Remove wake word and clean up extra whitespace
            before = text[:idx]
            after = text[idx + len(wake_word):]
            cleaned = (before + " " + after).strip()
            
            # Remove extra spaces and commas
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.lstrip(',').strip()
            
            return True, cleaned
    
    return False, text


def check_wake_word_status(config: dict) -> str:
    """Return a status message about wake word configuration."""
    if not config.get("wake_word_enabled", True):
        return "Wake word detection: DISABLED"
    
    wake_words = config.get("wake_words", [])
    timeout = config.get("wake_session_timeout", 180)
    
    words_list = ", ".join(f"'{w}'" for w in wake_words)
    
    return f"Wake word detection: ENABLED (session: {timeout}s)\nWake words: {words_list}"


def main():
    """Main CLI loop with multi-user context support."""
    parser = argparse.ArgumentParser(
        description="Collabry AI Core Engine CLI (Testing)",
        epilog="NOTE: In production, user_id comes from JWT token validation"
    )
    parser.add_argument(
        "--user",
        default="test_user",
        help="User ID for testing (default: test_user)"
    )
    parser.add_argument(
        "--session",
        default="default_session",
        help="Session ID for testing (default: default_session)"
    )
    args = parser.parse_args()
    
    print("="*70)
    print(" COLLABRY AI CORE ENGINE - CLI Interface (Legacy)")
    print("="*70)
    print(f"User Context: {args.user}:{args.session}")
    print(f"MongoDB: {CONFIG['mongo_uri']}")
    print("="*70)
    
    wake_status = check_wake_word_status(CONFIG)
    print(f"{wake_status}\n")
    
    # Create agent with user context
    agent, llm, tools, memory = create_agent(
        user_id=args.user,
        session_id=args.session,
        config=CONFIG
    )
    
    print(f"✓ Agent initialized for user: {args.user}")
    print(f"✓ Active session: {args.session}")
    print(f"✓ Memory thread_id: {memory.thread_id}")
    print(f"✓ Tools loaded: {len(tools)}\n")
    
    # List existing sessions for this user
    sessions = memory.list_user_sessions()
    if len(sessions) > 1:
        print(f"You have {len(sessions)} existing sessions:")
        for sess in sessions:
            is_current = " (CURRENT)" if sess['session_id'] == args.session else ""
            print(f"  - {sess['session_id']}{is_current}")
        print()
    
    print("Commands: 'exit', 'sessions', 'new session', 'switch <session_id>'\n")
    print("Type your question to begin.\n")

    # Wake session tracking
    wake_session_active = False
    wake_session_start = None

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            user_lower = user_input.lower()
            if user_lower in ("exit", "quit", "bye"):
                print("Goodbye!")
                break
            
            # Special commands
            if user_lower == "sessions":
                sessions = memory.list_user_sessions()
                print(f"\nYour sessions ({len(sessions)} total):")
                for sess in sessions:
                    is_current = " (CURRENT)" if sess['session_id'] == memory.session_id else ""
                    print(f"  - {sess['session_id']}{is_current}")
                print()
                continue
            
            if user_lower.startswith("switch "):
                new_session = user_input[7:].strip()
                try:
                    memory.switch_session(new_session)
                    print(f"✓ Switched to session: {new_session}\n")
                except ValueError as e:
                    print(f"✗ Error: {e}\n")
                continue
            
            if user_lower == "new session":
                new_id = memory.create_session()
                memory.switch_session(new_id)
                print(f"✓ Created and switched to new session: {new_id}\n")
                continue
            
            # Wake word detection (if enabled)
            wake_enabled = CONFIG.get("wake_word_enabled", True)
            wake_words = CONFIG.get("wake_words", [])
            wake_timeout = CONFIG.get("wake_session_timeout", 150)
            
            if wake_enabled and wake_words:
                has_wake, cleaned_input = detect_and_remove_wake_word(user_input, wake_words)
                
                # Check if wake session is still active
                if wake_session_active and wake_session_start:
                    elapsed = time.time() - wake_session_start
                    if elapsed > wake_timeout:
                        wake_session_active = False
                        wake_session_start = None
                
                if has_wake:
                    # Wake word detected - start/refresh session
                    wake_session_active = True
                    wake_session_start = time.time()
                    user_input = cleaned_input if cleaned_input else user_input
                    print("[COLLABRY activated - listening for {}s]\n".format(wake_timeout))
                elif wake_session_active:
                    # No wake word but session is active
                    remaining = int(wake_timeout - (time.time() - wake_session_start))
                    if remaining > 0:
                        print(f"[Session active - {remaining}s remaining]\n")
                    # Continue with original input
                else:
                    # No wake word and session expired/inactive
                    print("(Please use a wake word like 'COLLABRY' to activate me.)\n")
                    continue
            
            # Handle command via streaming
            try:
                agent.handle_user_input_stream(user_input, stream_printer)
                print("\n")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"COLLABRY: Error while processing: {e}\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")


if __name__ == "__main__":
    main()
