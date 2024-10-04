import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')
import logging
from torchvision import transforms
from models import build_model
from processors import build_processor
from utils import set_seed
from runner.runner import Runner

logger = logging.getLogger(__name__)


def run(args, model, processor, optimizer, scheduler, transform):
    set_seed(args)

    logger.info("train dataloader generation")
    print("train dataloader generation")
    train_examples, train_features, train_dataloader, args.train_invalid_num = processor.generate_dataloader('train', transform)
    logger.info("dev dataloader generation")
    print("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev', transform)
    logger.info("test dataloader generation")
    print("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test', transform)

    runner = Runner(
        cfg=args,
        data_samples=[train_examples, dev_examples, test_examples],
        data_features=[train_features, dev_features, test_features],
        data_loaders=[train_dataloader, dev_dataloader, test_dataloader],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn_dict=None,
        processor=processor,
    )
    runner.run()


def main():
    from config_parser import get_args_parser
    args = get_args_parser()

    if not args.inference_only:
        print(f"Output full path {os.path.join(os.getcwd(), args.output_dir)}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"), \
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt='%m/%d/%Y %H:%M:%S', level = logging.INFO
            )
    else:
        logging.basicConfig(
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
            )
    set_seed(args)
    transform = transforms.Compose([  # 设置transformes
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    model, tokenizer, optimizer, scheduler = build_model(args) 
    model.to(args.device)

    processor = build_processor(args, tokenizer, transform)

    logger.info("Training/evaluation parameters %s", args)
    run(args, model, processor, optimizer, scheduler, transform)
            

if __name__ == "__main__":
    main()