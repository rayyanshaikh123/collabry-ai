"""
OpenAI Fine-Tuning Script for Collabry

This script handles:
1. Uploading training data to OpenAI
2. Creating fine-tuning job
3. Monitoring training progress
4. Retrieving fine-tuned model ID
"""

import os
import argparse
import time
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def upload_training_file(client: OpenAI, file_path: str) -> str:
    """
    Upload training file to OpenAI.
    
    Returns:
        File ID
    """
    print(f"📤 Uploading {file_path}...")
    
    with open(file_path, "rb") as f:
        file_obj = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    print(f"✅ File uploaded successfully!")
    print(f"   File ID: {file_obj.id}")
    print(f"   Filename: {file_obj.filename}")
    print(f"   Size: {file_obj.bytes} bytes")
    
    return file_obj.id


def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    validation_file_id: str = None,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: str = "collabry",
    n_epochs: int = 3
) -> str:
    """
    Create fine-tuning job.
    
    Returns:
        Job ID
    """
    print(f"\n🚀 Creating fine-tuning job...")
    print(f"   Base model: {model}")
    print(f"   Suffix: {suffix}")
    print(f"   Epochs: {n_epochs}")
    
    job_params = {
        "training_file": training_file_id,
        "model": model,
        "suffix": suffix,
        "hyperparameters": {
            "n_epochs": n_epochs
        }
    }
    
    if validation_file_id:
        job_params["validation_file"] = validation_file_id
        print(f"   Validation file: {validation_file_id}")
    
    job = client.fine_tuning.jobs.create(**job_params)
    
    print(f"✅ Fine-tuning job created!")
    print(f"   Job ID: {job.id}")
    print(f"   Status: {job.status}")
    
    return job.id


def monitor_fine_tuning_job(client: OpenAI, job_id: str, poll_interval: int = 60):
    """
    Monitor fine-tuning job progress.
    
    Args:
        client: OpenAI client
        job_id: Fine-tuning job ID
        poll_interval: Seconds between status checks
    """
    print(f"\n⏳ Monitoring fine-tuning job {job_id}...")
    print(f"   Polling every {poll_interval} seconds")
    print("=" * 60)
    
    start_time = time.time()
    last_status = None
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"[{elapsed/60:.1f}min] Status: {status}")
            
            if job.trained_tokens:
                print(f"         Trained tokens: {job.trained_tokens:,}")
            
            last_status = status
        
        # Check for completion
        if status == "succeeded":
            elapsed = time.time() - start_time
            print("\n" + "=" * 60)
            print(f"🎉 Fine-tuning completed successfully!")
            print(f"   Duration: {elapsed/60:.1f} minutes")
            print(f"   Model ID: {job.fine_tuned_model}")
            print(f"   Trained tokens: {job.trained_tokens:,}")
            
            # Save model ID to config
            save_model_id(job.fine_tuned_model)
            
            return job.fine_tuned_model
        
        elif status == "failed":
            print("\n❌ Fine-tuning job failed!")
            if job.error:
                print(f"   Error: {job.error}")
            return None
        
        elif status == "cancelled":
            print("\n⚠️  Fine-tuning job was cancelled")
            return None
        
        # Wait before next check
        time.sleep(poll_interval)


def save_model_id(model_id: str):
    """Save fine-tuned model ID to .env file."""
    env_path = Path("../../.env")
    
    # Read existing .env
    env_lines = []
    finetuned_line_found = False
    
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("OPENAI_FINETUNED_MODEL="):
                    env_lines.append(f"OPENAI_FINETUNED_MODEL={model_id}\n")
                    finetuned_line_found = True
                elif line.startswith("USE_FINETUNED_MODEL="):
                    env_lines.append("USE_FINETUNED_MODEL=true\n")
                else:
                    env_lines.append(line)
    
    # Add if not found
    if not finetuned_line_found:
        env_lines.append(f"\n# Fine-tuned model\n")
        env_lines.append(f"OPENAI_FINETUNED_MODEL={model_id}\n")
        env_lines.append("USE_FINETUNED_MODEL=true\n")
    
    # Write updated .env
    with open(env_path, "w") as f:
        f.writelines(env_lines)
    
    print(f"\n💾 Model ID saved to .env file")
    print(f"   Set OPENAI_FINETUNED_MODEL={model_id}")


def list_fine_tuning_jobs(client: OpenAI, limit: int = 10):
    """List recent fine-tuning jobs."""
    print(f"📋 Recent fine-tuning jobs (limit {limit}):")
    print("=" * 60)
    
    jobs = client.fine_tuning.jobs.list(limit=limit)
    
    for job in jobs.data:
        print(f"\nJob ID: {job.id}")
        print(f"  Status: {job.status}")
        print(f"  Model: {job.model}")
        print(f"  Fine-tuned model: {job.fine_tuned_model or 'N/A'}")
        print(f"  Created: {job.created_at}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Collabry model with OpenAI")
    parser.add_argument("--train-file", default="data/training/collabry_training.jsonl", 
                       help="Path to training JSONL file")
    parser.add_argument("--val-file", default="data/training/synthetic_validation.jsonl",
                       help="Path to validation JSONL file")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18",
                       help="Base model to fine-tune")
    parser.add_argument("--suffix", default="collabry",
                       help="Model name suffix")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--list", action="store_true",
                       help="List recent fine-tuning jobs and exit")
    parser.add_argument("--monitor", metavar="JOB_ID",
                       help="Monitor existing fine-tuning job")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Don't monitor after creating job")
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    client = OpenAI(api_key=api_key)
    
    # Handle list command
    if args.list:
        list_fine_tuning_jobs(client)
        return 0
    
    # Handle monitor command
    if args.monitor:
        monitor_fine_tuning_job(client, args.monitor)
        return 0
    
    # Validate files exist
    train_path = Path(args.train_file)
    if not train_path.exists():
        print(f"❌ Error: Training file not found: {args.train_file}")
        return 1
    
    val_path = Path(args.val_file) if args.val_file else None
    if val_path and not val_path.exists():
        print(f"⚠️  Warning: Validation file not found: {args.val_file}")
        val_path = None
    
    try:
        # Upload training file
        train_file_id = upload_training_file(client, str(train_path))
        
        # Upload validation file if provided
        val_file_id = None
        if val_path:
            val_file_id = upload_training_file(client, str(val_path))
        
        # Create fine-tuning job
        job_id = create_fine_tuning_job(
            client,
            training_file_id=train_file_id,
            validation_file_id=val_file_id,
            model=args.model,
            suffix=args.suffix,
            n_epochs=args.epochs
        )
        
        print(f"\n📝 Fine-tuning job created: {job_id}")
        print(f"   View in dashboard: https://platform.openai.com/finetune/{job_id}")
        
        # Monitor if not disabled
        if not args.no_monitor:
            model_id = monitor_fine_tuning_job(client, job_id)
            
            if model_id:
                print("\n" + "=" * 60)
                print("✅ Fine-tuning complete!")
                print(f"\nTo use the fine-tuned model, add to your .env:")
                print(f"  OPENAI_FINETUNED_MODEL={model_id}")
                print(f"  USE_FINETUNED_MODEL=true")
                return 0
            else:
                return 1
        else:
            print(f"\n⏳ Job started successfully. Monitor with:")
            print(f"   python scripts/finetune_model.py --monitor {job_id}")
            return 0
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
