from clean_data import main as clean_data_main
from encode_data import main as encode_data_main
from feature_engineering import main as feature_engineering_main
from train_model import main as train_model_main
from predict_test import main as predict_test_main


def main():
    clean_data_main()
    encode_data_main()
    feature_engineering_main()
    train_model_main()
    predict_test_main()


if __name__ == "__main__":
    main()
