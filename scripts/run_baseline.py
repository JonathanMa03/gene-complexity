"""Train the processed-expression logistic baseline (no dictionary or scGPT required)."""
from _common import train, train_parser
if __name__ == '__main__':
    train(train_parser(__doc__,baseline=True).parse_args())
