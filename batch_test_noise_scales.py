import subprocess
import time
import json
import os
import argparse
from datetime import datetime

def run_command(cmd, description):
    """Run a command and track its execution time."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start_time = time.time()
    result = subprocess.run(cmd, check=True, capture_output=False)
    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    return elapsed

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch test different noise scales for CI calibration")
    parser.add_argument("--start-day", type=int, default=5,
                        help="Starting day for generation (default: 5)")
    parser.add_argument("--days-to-generate", type=int, default=5810,
                        help="Number of days to generate (default: 5810)")
    parser.add_argument("--eval-start-day", type=int, default=None,
                        help="Starting day for evaluation (default: None, uses last 1000 days)")
    parser.add_argument("--eval-end-day", type=int, default=None,
                        help="Ending day for evaluation (default: None, uses last 1000 days)")
    args = parser.parse_args()

    # Noise values to test
    noise_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    # Determine output filename suffix based on evaluation window
    if args.eval_start_day is not None and args.eval_end_day is not None:
        results_suffix = f"_days{args.eval_start_day}-{args.eval_end_day}"
        eval_description = f"days {args.eval_start_day}-{args.eval_end_day}"
    else:
        results_suffix = ""
        eval_description = "last 1000 days (default)"

    print("="*70)
    print("Batch Testing Noise Scales for CI Calibration")
    print("="*70)
    print(f"\nGeneration window: days {args.start_day}-{args.start_day+args.days_to_generate} ({args.days_to_generate} days)")
    print(f"Evaluation window: {eval_description}")
    print(f"\nNoise values to test: {noise_values}")
    print(f"Total runs: {len(noise_values)}")
    print(f"Estimated time: ~{len(noise_values) * 11} minutes (for OOD) or ~{len(noise_values) * 2} minutes (for small in-dist)")
    print(f"\nStarting batch test at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Track results
    results = {
        "noise_values": noise_values,
        "generation_times": {},
        "evaluation_times": {},
        "total_times": {},
        "start_time": datetime.now().isoformat(),
        "completed": []
    }

    overall_start = time.time()

    for i, noise in enumerate(noise_values, 1):
        print(f"\n{'#'*70}")
        print(f"# TESTING NOISE_SCALE = {noise} ({i}/{len(noise_values)})")
        print(f"{'#'*70}")

        iter_start = time.time()

        try:
            # Step 1: Generate surfaces
            gen_cmd = ["python", "generate_surfaces_empirical.py",
                       "--noise-scale", str(noise),
                       "--start-day", str(args.start_day),
                       "--days-to-generate", str(args.days_to_generate)]
            gen_time = run_command(
                gen_cmd,
                f"[{i}/{len(noise_values)}] Generating surfaces with noise={noise}"
            )

            # Step 2: Evaluate CI calibration
            eval_cmd = ["python", "evaluate_empirical_ci.py",
                        "--noise-scale", str(noise),
                        "--start-day", str(args.start_day),
                        "--days-to-generate", str(args.days_to_generate)]
            if args.eval_start_day is not None:
                eval_cmd.extend(["--eval-start-day", str(args.eval_start_day)])
            if args.eval_end_day is not None:
                eval_cmd.extend(["--eval-end-day", str(args.eval_end_day)])
            eval_time = run_command(
                eval_cmd,
                f"[{i}/{len(noise_values)}] Evaluating CI calibration for noise={noise}"
            )

            iter_elapsed = time.time() - iter_start

            # Store results
            results["generation_times"][str(noise)] = gen_time
            results["evaluation_times"][str(noise)] = eval_time
            results["total_times"][str(noise)] = iter_elapsed
            results["completed"].append(noise)

            print(f"\n{'='*70}")
            print(f"✓ Completed noise={noise}")
            print(f"  Generation: {gen_time/60:.1f} min")
            print(f"  Evaluation: {eval_time/60:.1f} min")
            print(f"  Total: {iter_elapsed/60:.1f} min")
            print(f"  Progress: {i}/{len(noise_values)} ({100*i/len(noise_values):.1f}%)")
            print(f"{'='*70}")

        except subprocess.CalledProcessError as e:
            print(f"\n✗ ERROR: Failed at noise={noise}")
            print(f"  Error: {e}")
            results["error"] = {"noise": noise, "message": str(e)}
            break

    overall_elapsed = time.time() - overall_start
    results["end_time"] = datetime.now().isoformat()
    results["total_elapsed_seconds"] = overall_elapsed

    # Save results to JSON
    results_file = f"batch_test_results{results_suffix}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("BATCH TEST COMPLETE!")
    print("="*70)
    print(f"\nCompleted {len(results['completed'])}/{len(noise_values)} noise values")
    print(f"Total time: {overall_elapsed/60:.1f} minutes ({overall_elapsed/3600:.2f} hours)")
    print(f"Average time per noise: {overall_elapsed/len(results['completed'])/60:.1f} minutes")
    print(f"\nResults saved to: {results_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print summary
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"{'Noise':<10} {'Generation':<15} {'Evaluation':<15} {'Total':<15}")
    print("-"*70)
    for noise in results["completed"]:
        gen = results["generation_times"][str(noise)]
        ev = results["evaluation_times"][str(noise)]
        tot = results["total_times"][str(noise)]
        print(f"{noise:<10} {gen/60:>6.1f} min      {ev/60:>6.1f} min      {tot/60:>6.1f} min")
    print("-"*70)
    print(f"{'TOTAL':<10} {sum(results['generation_times'].values())/60:>6.1f} min      "
          f"{sum(results['evaluation_times'].values())/60:>6.1f} min      "
          f"{overall_elapsed/60:>6.1f} min")

    print("\n✓ Next step: Run analyze_noise_sweep.py to analyze all results")

if __name__ == "__main__":
    main()
