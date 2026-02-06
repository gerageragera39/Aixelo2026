from resnet_bottleneck_model import BottleneckModelConfig
from train_bottleneck import train_bottleneck_resnet
import argparse
import os
from data_loader_lmdb import load_dataset_lmdb, make_dataloaders_lmdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=50,
                       help="ResNet depth (50, 101)")
    parser.add_argument("--more_epochs", type=int, required=True,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--sd_on", type=int, default=0,
                       help="Enable Stochastic Depth")
    parser.add_argument("--pL", type=float, default=0.5,
                       help="Final survival probability for Stochastic Depth")
    parser.add_argument("--out", type=str, default="../places/checkpoints",
                       help="Output directory")
    parser.add_argument("--data-dir", type=str, default="/media/herman/Windows/Users/samys/PycharmProjects/GPU/Resnet/StochDepth/ResNet/SuperPC/StochasticDepthThesis/places/Places365_small_lmdb",
                       help="Directory containing data")
    parser.add_argument("--num_classes", type=int, default=365,
                       help="Number of classes in the LMDB dataset")
    parser.add_argument("--lr", type=float, default=0.1,
                       help="Learning rate")
    parser.add_argument("--cp", type=str, default=None,
                       help="Checkpoint path for resuming training")
    args = parser.parse_args()


    SD_ON = bool(args.sd_on)

    trainset, testset, num_classes = load_dataset_lmdb(data_dir=args.data_dir, num_classes=args.num_classes)
    trainloader, testloader = make_dataloaders_lmdb(trainset, testset, args.batch)


    print(f"\nDataset Classes: {num_classes}")
    print(f"Train: {len(trainset)}, Test: {len(testset)}")


    config = BottleneckModelConfig(
        depth=args.depth,
        sd_on=SD_ON,
        final_survival_prob=args.pL,
        epochs=args.more_epochs,
        batch_size=args.batch,
        output_dir=os.path.join(args.out, f"seed_42/sd_on_{SD_ON}"),
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT:")
    print("=" * 60)

    log_df, train_time, avg_batch_time = train_bottleneck_resnet(config, trainloader, testloader, num_classes, args.cp, args.more_epochs, args.lr)
    print("\nExperiment completed!")

