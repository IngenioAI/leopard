import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module


def main(args):

    if args.shadow_train:
        save_dir = f"{args.save_dir}/shadow"
    else:
        save_dir = f"{args.save_dir}/target"

    os.makedirs(save_dir, exist_ok = True)

    
    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)

        checkpoint = ModelCheckpoint(
                    monitor='val_acc',
                    mode="max", 
                    save_last=False,
                    dirpath=save_dir,
                    filename='{epoch:02d}-{val_acc:.2f}')

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            gpus=-1,
            deterministic=True,
            weights_summary=None,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            checkpoint_callback=[checkpoint],
            precision=args.precision,
            weights_save_path=save_dir,
        )

        model = CIFAR10Module(args)
        data = CIFAR10Data(args)


        if args.shadow_train:
            trainer.fit(model, data.test_dataloader(args.shadow_train))
            trainer.test(model, data.train_dataloader(args.shadow_train))
        else:
            if bool(args.pretrained):
                state_dict = os.path.join(
                    "cifar10_models", "state_dicts", args.classifier + ".pt"
                )
                model.model.load_state_dict(torch.load(state_dict))

            if bool(args.test_phase):
                trainer.test(model, data.test_dataloader())
            else:
                trainer.fit(model, data)
                trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/data/datasets/cifar10-data")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--shadow_train", action="store_true")
   
    parser.add_argument("--save_dir", type=str, default="/home/yujeong/project/iitp-privacy-2024/mu-evaluator/pytorch_cifar10/cifar10_models")

    args = parser.parse_args()
    main(args)
