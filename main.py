from args_parser import parse_args
from train import train_one_epoch, evaluate
from model import create_model

def main():
    opt = parse_args()
    model = create_model(opt)
    # ...set up data loaders...
    for epoch in range(1, opt.epochs + 1):
        train_one_epoch(model, trainingdata, optimizer, device, scaler, opt, epoch)
        evaluate(model, testingdata, device, scaler, opt, epoch)
        # ...save model...

if __name__ == "__main__":
    main()