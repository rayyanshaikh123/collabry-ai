"""
Model Evaluation Script for Collabry

Compares base model vs fine-tuned model performance on:
- Tool selection accuracy
- Hallucination rate  
- Multi-step reasoning
- Appropriate refusals
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter
import os
import sys
from dotenv import load_dotenv

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import run_agent
from core.llm import reset_clients


class EvaluationMetrics:
    """Track evaluation metrics."""
    
    def __init__(self):
        self.total = 0
        self.correct_tool = 0
        self.wrong_tool = 0
        self.hallucinated_tool = 0
        self.missing_tool = 0
        self.correct_no_tool = 0
        self.multi_tool_correct = 0
        self.appropriate_refusal = 0
        
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.correct_tool + self.correct_no_tool + self.multi_tool_correct + self.appropriate_refusal) / self.total
    
    def hallucination_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hallucinated_tool / self.total
    
    def report(self) -> str:
        acc = self.accuracy() * 100
        hall_rate = self.hallucination_rate() * 100
        
        return f"""
📊 Evaluation Results:
  Total test cases: {self.total}
  
  ✅ Correct tool selection: {self.correct_tool}
  ❌ Wrong tool selected: {self.wrong_tool}
  🚫 Hallucinated tool (tool doesn't exist): {self.hallucinated_tool}
  ⚠️  Missing tool (should have used tool): {self.missing_tool}
  ✓  Correct no-tool response: {self.correct_no_tool}
  ✓  Multi-tool sequence correct: {self.multi_tool_correct}
  ✓  Appropriate refusal: {self.appropriate_refusal}
  
  Overall Accuracy: {acc:.1f}%
  Hallucination Rate: {hall_rate:.1f}%
"""


async def evaluate_test_case(
    test_case: Dict,
    user_id: str = "eval_user",
    session_id: str = "eval_session"
) -> Dict:
    """
    Evaluate a single test case.
    
    Returns:
        Dict with evaluation result
    """
    messages = test_case["messages"]
    expected = test_case["expected_behavior"]
    description = test_case.get("description", "")
    
    # Extract user message
    user_message = None
    for msg in messages:
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    
    if not user_message:
        return {"error": "No user message found", "test_case": description}
    
    # Run agent
    tools_used = []
    response_text = ""
    error = None
    
    try:
        async for event in run_agent(user_id, session_id, user_message, stream=True):
            if event["type"] == "tool_start":
                tools_used.append(event["tool"])
            elif event["type"] == "token":
                response_text += event["content"]
            elif event["type"] == "error":
                error = event["message"]
    except Exception as e:
        error = str(e)
    
    # Evaluate based on expected behavior
    result = {
        "test_case": description,
        "user_message": user_message,
        "expected": expected,
        "tools_used": tools_used,
        "response_length": len(response_text),
        "error": error
    }
    
    # Check correctness
    if expected == "no_tool":
        result["correct"] = len(tools_used) == 0 and not error
        result["result_type"] = "correct_no_tool" if result["correct"] else "missing_tool"
    
    elif expected == "refuse":
        # Check if response indicates refusal
        refusal_phrases = ["can't", "cannot", "unable", "not able", "outside my", "don't have"]
        refused = any(phrase in response_text.lower() for phrase in refusal_phrases)
        result["correct"] = refused and len(tools_used) == 0
        result["result_type"] = "appropriate_refusal" if result["correct"] else "wrong_tool"
    
    elif expected == "multi_tool":
        result["correct"] = len(tools_used) >= 2
        result["result_type"] = "multi_tool_correct" if result["correct"] else "missing_tool"
    
    else:
        # Single tool expected
        if expected in tools_used:
            result["correct"] = True
            result["result_type"] = "correct_tool"
        elif len(tools_used) == 0:
            result["correct"] = False
            result["result_type"] = "missing_tool"
        else:
            # Check if hallucinated (used non-existent tool)
            valid_tools = {
                "search_sources", "generate_quiz", "generate_flashcards",
                "generate_mindmap", "generate_study_plan", "summarize_notes"
            }
            hallucinated = any(tool not in valid_tools for tool in tools_used)
            result["correct"] = False
            result["result_type"] = "hallucinated_tool" if hallucinated else "wrong_tool"
    
    return result


async def evaluate_model(
    test_file: str,
    model_name: str = "base"
) -> EvaluationMetrics:
    """
    Evaluate model on test cases.
    
    Args:
        test_file: Path to test cases JSONL
        model_name: Name for reporting
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} model")
    print(f"{'='*60}\n")
    
    metrics = EvaluationMetrics()
    
    # Load test cases
    test_cases = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))
    
    print(f"Loaded {len(test_cases)} test cases\n")
    
    # Evaluate each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {test_case['description'][:50]}...", end=" ")
        
        result = await evaluate_test_case(test_case)
        
        metrics.total += 1
        result_type = result["result_type"]
        
        if result_type == "correct_tool":
            metrics.correct_tool += 1
            print("✅")
        elif result_type == "wrong_tool":
            metrics.wrong_tool += 1
            print(f"❌ (used {result['tools_used']})")
        elif result_type == "hallucinated_tool":
            metrics.hallucinated_tool += 1
            print(f"🚫 (hallucinated {result['tools_used']})")
        elif result_type == "missing_tool":
            metrics.missing_tool += 1
            print("⚠️  (no tool used)")
        elif result_type == "correct_no_tool":
            metrics.correct_no_tool += 1
            print("✓")
        elif result_type == "multi_tool_correct":
            metrics.multi_tool_correct += 1
            print(f"✓ (used {len(result['tools_used'])} tools)")
        elif result_type == "appropriate_refusal":
            metrics.appropriate_refusal += 1
            print("✓ (refused)")
        
        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)
    
    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Collabry model performance")
    parser.add_argument("--test-file", default="data/eval/test_cases.jsonl",
                       help="Path to test cases JSONL")
    parser.add_argument("--model", choices=["base", "finetuned", "both"], default="both",
                       help="Which model to evaluate")
    
    args = parser.parse_args()
    
    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"❌ Test file not found: {args.test_file}")
        return 1
    
    results = {}
    
    # Evaluate base model
    if args.model in ["base", "both"]:
        # Temporarily disable fine-tuned model
        os.environ["USE_FINETUNED_MODEL"] = "false"
        reset_clients()
        
        metrics_base = await evaluate_model(args.test_file, "Base (gpt-4o-mini)")
        results["base"] = metrics_base
        print(metrics_base.report())
    
    # Evaluate fine-tuned model
    if args.model in ["finetuned", "both"]:
        finetuned_model = os.getenv("OPENAI_FINETUNED_MODEL")
        
        if not finetuned_model:
            print("\n⚠️  No fine-tuned model configured (OPENAI_FINETUNED_MODEL not set)")
            if args.model == "finetuned":
                return 1
        else:
            # Enable fine-tuned model
            os.environ["USE_FINETUNED_MODEL"] = "true"
            reset_clients()
            
            metrics_finetuned = await evaluate_model(args.test_file, f"Fine-tuned ({finetuned_model[:20]}...)")
            results["finetuned"] = metrics_finetuned
            print(metrics_finetuned.report())
    
    # Comparison
    if args.model == "both" and "base" in results and "finetuned" in results:
        print("\n" + "="*60)
        print("📈 Comparison")
        print("="*60)
        
        base_acc = results["base"].accuracy() * 100
        ft_acc = results["finetuned"].accuracy() * 100
        improvement = ft_acc - base_acc
        
        base_hall = results["base"].hallucination_rate() * 100
        ft_hall = results["finetuned"].hallucination_rate() * 100
        hall_reduction = base_hall - ft_hall
        
        print(f"\nAccuracy:")
        print(f"  Base model:       {base_acc:.1f}%")
        print(f"  Fine-tuned model: {ft_acc:.1f}%")
        print(f"  Improvement:      {improvement:+.1f}%")
        
        print(f"\nHallucination Rate:")
        print(f"  Base model:       {base_hall:.1f}%")
        print(f"  Fine-tuned model: {ft_hall:.1f}%")
        print(f"  Reduction:        {hall_reduction:+.1f}%")
        
        if ft_acc >= 95 and ft_hall <= 5:
            print("\n🎉 Fine-tuned model meets success criteria!")
            print("   ✓ Accuracy ≥ 95%")
            print("   ✓ Hallucination rate ≤ 5%")
        else:
            print("\n⚠️  Fine-tuned model needs improvement:")
            if ft_acc < 95:
                print(f"   - Accuracy is {ft_acc:.1f}% (target: ≥95%)")
            if ft_hall > 5:
                print(f"   - Hallucination rate is {ft_hall:.1f}% (target: ≤5%)")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
