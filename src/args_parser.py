import argparse
import configparser
import os

def parse_args(full_path, colab):
    conf_parser = argparse.ArgumentParser(
    description=__doc__, # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
    )

    if colab:
        training_data_path = "/content/drive/Othercomputers/Mi portátil/dataset"
        test_data_path = "/content/drive/Othercomputers/Mi portátil/test_Data"
    else:
        data_path = "/content/drive/Othercomputers/Mi portátil/dataset"
        data_test_path = "/content/drive/MyDrive/dope-training/test_Data"

    conf_parser.add_argument("-c", "--config",
                            help="Specify config file", metavar="FILE")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
        default = training_data_path,
        help='path to training data')

    parser.add_argument('--datatest',
        default= test_data_path,
        help='path to data testing set')

    parser.add_argument('--object',
        default="60",
        help='In the dataset which object of interest')

    parser.add_argument('--workers',
        type=int,
        default=1,
        help='number of data loading workers')

    parser.add_argument('--batchsize',
        type=int,
        default=128,
        help='input batch size')

    parser.add_argument('--subbatchsize',
        type=int,
        default=32,
        help='input batch size')

    parser.add_argument('--imagesize',
        type=int,
        default=400,
        help='the height / width of the input image to network')

    parser.add_argument('--lr',
        type=float,
        default=0.0001,
        help='learning rate, default=0.0001')

    parser.add_argument('--noise',
        type=float,
        default=0.7,
        help='gaussian noise added to the image')

    parser.add_argument('--net',
        default=os.path.join("/content/drive/Othercomputers/Mi portátil/weights/net.pth"),
        help="path to net (to continue training)")

    parser.add_argument('--namefile',
        default='weights_mustard',
        help="name to put on the file of the save weights")

    parser.add_argument('--manualseed',
        type=int,
        help='manual seed')

    parser.add_argument('--epochs',
        type=int,
        default=10,
        help="number of epochs to train")

    parser.add_argument('--loginterval',
        type=int,
        default=100)

    parser.add_argument('--gpuids',
        nargs='+',
        type=int,
        default=[0],
        help='GPUs to use')

    parser.add_argument('--outf',
        default=os.path.join(full_path, "out"),
        help='folder to output images and model checkpoints, it will \
        add a train_ in front of the name')

    parser.add_argument('--sigma',
        default=8,
        help='keypoint creation size for sigma')

    parser.add_argument('--save',
        default = True,
        help='save a visual batch and quit, this is for\
        debugging purposes')

    parser.add_argument("--pretrained",
        default=True,
        help='do you want to use vgg imagenet pretrained weights')

    parser.add_argument('--nbupdates',
        default=None,
        help='nb max update to network, overwrites the epoch number\
        otherwise uses the number of epochs')

    parser.add_argument('--datasize',
        default=None,
        help='randomly sample that number of entries in the dataset folder')

    # Read the config but do not overwrite the args written
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = { "option":"default" }

    if args.config:
        config = configparser.SafeConfigParser()
        config.read([args.config])
        defaults.update(dict(config.items("defaults")))

    parser.set_defaults(**defaults)
    parser.add_argument("--option")

    # Parse known arguments
    opt, unknown = parser.parse_known_args(remaining_argv)

    if opt.pretrained in ['false', 'False']:
        opt.pretrained = False

    return opt