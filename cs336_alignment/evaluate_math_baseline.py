import json
from pathlib import Path
from typing import List, Dict, Callable
from vllm import LLM, SamplingParams
import time

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from utils import get_model_path


def load_math_validation(file_path: str) -> List[Dict]:
    """Load MATH validation examples from JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_r1_zero_prompt(problem: str) -> str:
    """
    Format a math problem using the r1_zero prompt template.

    The r1_zero prompt typically asks the model to solve step-by-step
    and end with <answer>...</answer> tags.
    """
    # This is a typical r1_zero prompt format based on the assignment description
    prompt = f"""Solve the following math problem step by step.

Problem: {problem}

Please provide a detailed solution and end with your final answer wapped in <answer> tags.

Solution: Let me solve this step by step.
"""
    return prompt


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
    output_file: str = "math_evaluation_results.json"
) -> Dict:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics, 
    and serialize results to disk.

    Args:
        vllm_model: The vLLM model to evaluate
        reward_fn: Function that takes (model_output, ground_truth) and returns rewards
        prompts: List of formatted prompts
        eval_sampling_params: Sampling parameters for generation
        ground_truths: List of ground truth answers
        output_file: Where to save results

    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating {len(prompts)} examples...")

    # Generate outputs
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Evaluate each output
    results = []
    metrics = {
        "total": len(prompts),
        "correct_both": 0,  # format=1, answer=1
        "format_only": 0,   # format=1, answer=0
        "incorrect_both": 0, # format=0, answer=0
        "format_rewards": [],
        "answer_rewards": []
    }

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]

        # Get rewards from the reward function
        rewards = reward_fn(generated_text, ground_truth)

        # Categorize the result
        format_reward = rewards.get("format_reward", 0)
        answer_reward = rewards.get("answer_reward", 0)

        metrics["format_rewards"].append(format_reward)
        metrics["answer_rewards"].append(answer_reward)

        if format_reward == 1 and answer_reward == 1:
            metrics["correct_both"] += 1
        elif format_reward == 1 and answer_reward == 0:
            metrics["format_only"] += 1
        elif format_reward == 0 and answer_reward == 0:
            metrics["incorrect_both"] += 1

        # Store detailed result
        results.append({
            "prompt": prompts[i],
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "rewards": rewards
        })

    # Calculate overall metrics
    metrics["format_accuracy"] = sum(metrics["format_rewards"]) / len(prompts)
    metrics["answer_accuracy"] = sum(metrics["answer_rewards"]) / len(prompts)

    # Save results
    output_data = {
        "metrics": metrics,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")
    return output_data


def main():
    # Load MATH validation data
    validation_file = "/data/a5-alignment/MATH/validation.jsonl"

    # For local testing, check multiple paths
    local_paths = [
        Path(__file__).parent.parent / "data" / "MATH" / "validation.jsonl",
        Path(__file__).parent.parent / "data" / "MATH" / "validation_sample.jsonl",
    ]

    if not Path(validation_file).exists():
        for local_path in local_paths:
            if local_path.exists():
                validation_file = str(local_path)
                print(f"Using local validation file: {validation_file}")
                break
        else:
            # Create a minimal test set if nothing exists
            print("No validation file found. Creating a minimal test set...")
            create_test_validation_file()
            validation_file = str(local_paths[1])  # Use sample path

    examples = load_math_validation(validation_file)
    print(f"Loaded {len(examples)} validation examples")

    # Format prompts and extract ground truths
    prompts = []
    ground_truths = []

    for example in examples:
        # MATH dataset typically has 'problem' and 'solution' fields
        problem = example.get("problem", "")
        solution = example.get("solution", "")

        # Extract ground truth from solution (usually the final answer)
        # The exact format depends on your dataset
        ground_truth = example.get("answer", solution)  # Adjust based on actual format

        prompts.append(format_r1_zero_prompt(problem))
        ground_truths.append(ground_truth)

    # Initialize model
    model_path = get_model_path()
    llm = LLM(model=model_path)

    # Set up sampling parameters as specified
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Evaluate
    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths,
        output_file="math_baseline_results.json"
    )

    # Print summary
    metrics = results["metrics"]
    print("\n=== Evaluation Summary ===")
    print(f"Total examples: {metrics['total']}")
    print(f"Correct (format=1, answer=1): {metrics['correct_both']}")
    print(f"Format only (format=1, answer=0): {metrics['format_only']}")
    print(f"Incorrect both (format=0, answer=0): {metrics['incorrect_both']}")
    print(f"Format accuracy: {metrics['format_accuracy']:.2%}")
    print(f"Answer accuracy: {metrics['answer_accuracy']:.2%}")

    # Analyze failure cases for part (b)
    analyze_failure_cases(results["results"])

def analyze_failure_cases(results: List[Dict], num_examples: int = 10):
    """Analyze and print examples of different failure modes."""

    format_failures = []
    answer_failures = []

    for result in results:
        if result["format_reward"] == 0:
            format_failures.append(result)
        elif result["format_reward"] == 1 and result["answer_reward"] == 0:
            answer_failures.append(result)

    print(f"\n=== Format Failures (showing {min(num_examples, len(format_failures))}) ===")
    for i, failure in enumerate(format_failures[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"Generated: {failure['generated_text'][:200]}...")
        print(f"Ground truth: {failure['ground_truth']}")

    print(f"\n=== Answer Failures (showing {min(num_examples, len(answer_failures))}) ===")
    for i, failure in enumerate(answer_failures[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"Generated: {failure['generated_text'][:200]}...")
        print(f"Ground truth: {failure['ground_truth']}")

def create_test_validation_file():
    """Create a small test validation file for local development."""
    test_data = [
        {"problem": "What is 2 + 2?", "solution": "2 + 2 = 4", "answer": "4"},
        {"problem": "Solve for x: 2x + 4 = 10", "solution": "2x + 4 = 10\n2x = 6\nx = 3", "answer": "3"},
        {"problem": "What is the area of a square with side length 5?", "solution": "Area = side × side = 5 × 5 = 25", "answer": "25"},
        {"problem": "Simplify: (x + 2)(x - 2)", "solution": "(x + 2)(x - 2) = x² - 4", "answer": "x² - 4"},
        {"problem": "What is 15% of 80?", "solution": "15% of 80 = 0.15 × 80 = 12", "answer": "12"},
    ]

    output_dir = Path(__file__).parent.parent / "data" / "MATH"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "validation_sample.jsonl"

    with open(output_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    print(f"Created test validation file at {output_file}")



if __name__ == "__main__":
    main()
