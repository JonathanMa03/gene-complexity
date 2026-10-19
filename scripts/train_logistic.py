"""Train one logistic classifier on train cells; select C using separate validation runs."""
from _common import train, train_parser
if __name__ == '__main__':
    train(train_parser(__doc__).parse_args())
