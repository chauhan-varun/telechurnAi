from clean_data import main as clean_data_main
from encode_data import main as encode_data_main
from feature_engineering import main as feature_engineering_main


def main():
    clean_data_main()
    encode_data_main()
    feature_engineering_main()


if __name__ == "__main__":
    main()
