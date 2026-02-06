import argparse
import os
from model import ModelConfig
from train import train_single_model_cifar100
from utils import plot_histories_cifar100
import random



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=110,
                       help="ResNet depth (20, 32, 44, 56, 110)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--pL", type=float, default=0.5,
                       help="Final survival probability for Stochastic Depth")
    parser.add_argument("--out", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing CIFAR-100 data")
    args = parser.parse_args()

    
    seeds = [random.randint(0, 10**6) for _ in range(3)]


    print(f"Seeds: {seeds}")

    for s in seeds:

        log_dfs = []
        training_times = []
        avg_batch_times = []
        labels = []

        # Experiment 1: Without Stochastic Depth
        config_no_sd = ModelConfig(
            seed=s,
            depth=args.depth,
            sd_on=False,
            final_survival_prob=1.0,
            epochs=args.epochs,
            batch_size=args.batch,
            output_dir=os.path.join(args.out, f"{s}/no_sd"),
        )

        print("\n" + "=" * 60)
        print("EXPERIMENT 1: WITHOUT Stochastic Depth")
        print("=" * 60)
        log_df, train_time, avg_batch_time = train_single_model_cifar100(config_no_sd, data_dir=args.data_dir)
        log_dfs.append(log_df)
        training_times.append(train_time)
        avg_batch_times.append(avg_batch_time)
        labels.append("No SD")

        # Experiment 2: With Stochastic Depth
        config_sd = ModelConfig(
            seed=s,
            depth=args.depth,
            sd_on=True,
            final_survival_prob=args.pL,
            epochs=args.epochs,
            batch_size=args.batch,
            output_dir=os.path.join(args.out, f"{s}/sd_true_per_batch"),
        )

        print("\n" + "=" * 60)
        print("EXPERIMENT 2: WITH Stochastic Depth")
        print("=" * 60)
        log_df, train_time, avg_batch_time = train_single_model_cifar100(config_sd, data_dir=args.data_dir)
        log_dfs.append(log_df)
        training_times.append(train_time)
        avg_batch_times.append(avg_batch_time)
        labels.append("With SD ")

        # Visual
        print("\n" + "=" * 60)
        print("VISUALIZATION OF RESULTS WITH TIMING COMPARISON")
        print("=" * 60)
        plot_histories_cifar100(log_dfs, labels, training_times, avg_batch_times, args.out + f"/{s}")

        # Timing
        print("\n" + "=" * 60)
        print("TRAINING TIME STATISTICS")
        print("=" * 60)
        print(f"Without Stochastic Depth: {training_times[0]:.2f} seconds")
        print(f"With Stochastic Depth:  {training_times[1]:.2f} seconds")
        if training_times[0] > training_times[1]:
            speedup = training_times[0]/training_times[1]
            time_savings = ((training_times[0]-training_times[1])/training_times[0]*100)
            print(f"Speedup: {speedup:.2f}x")
            print(f"Time Savings: {time_savings:.1f}%")
        else:
            print("Speedup: < 1.0x (possible overhead from conditional execution)")

        print(f"\nAverage batch time:")
        print(f"Without Stochastic Depth: {avg_batch_times[0]:.4f} seconds")
        print(f"With Stochastic Depth:  {avg_batch_times[1]:.4f} seconds")
        if avg_batch_times[0] > avg_batch_times[1]:
            batch_speedup = avg_batch_times[0]/avg_batch_times[1]
            print(f"Batch speedup: {batch_speedup:.2f}x")
        else:
            print("Batch speedup: < 1.0x")

        # Save result
        results_file = os.path.join(args.out, "cifar100_timing_results.txt")
        with open(results_file, 'w') as f:
            f.write("CIFAR-100 Stochastic Depth Timing Analysis Results\n")
            f.write("="*50 + "\n")
            f.write(f"Without Stochastic Depth: {training_times[0]:.2f} seconds\n")
            f.write(f"With Stochastic Depth:  {training_times[1]:.2f} seconds\n")
            if training_times[0] > training_times[1]:
                f.write(f"Speedup: {training_times[0]/training_times[1]:.2f}x\n")
                f.write(f"Time Savings: {((training_times[0]-training_times[1])/training_times[0]*100):.1f}%\n")
            else:
                f.write("Speedup: < 1.0x (possibly due to conditional execution overhead)\n")
            f.write(f"\nAvg batch time without SD: {avg_batch_times[0]:.4f} seconds\n")
            f.write(f"Avg batch time with SD:  {avg_batch_times[1]:.4f} seconds\n")
            if avg_batch_times[0] > avg_batch_times[1]:
                f.write(f"Batch speedup: {avg_batch_times[0]/avg_batch_times[1]:.2f}x\n")
            else:
                f.write("Batch speedup: < 1.0x\n")

        print(f"\nResults saved to: {results_file}")
        print("\nExperiment completed!")

