# data_collection.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(delivery_path, match_path):
    """Loads the delivery and match data from the given file paths."""
    try:
        logging.info(f"Loading delivery data from: {delivery_path}")
        delivery = pd.read_csv(delivery_path)
        logging.info(f"Delivery data loaded successfully with shape: {delivery.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Delivery data file not found at {delivery_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading delivery data: {e}")
        raise

    try:
        logging.info(f"Loading match data from: {match_path}")
        match = pd.read_csv(match_path)
        logging.info(f"Match data loaded successfully with shape: {match.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Match data file not found at {match_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading match data: {e}")
        raise

    return delivery, match

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, 'data')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')
    matches_path = os.path.join(data_dir, 'matches.csv')
    # Load data using the paths constructed dynamically
    delivery_data, match_data = load_data(deliveries_path, matches_path)
    
    print("Delivery Data Head:")
    print(delivery_data.head(2))
    print("\nMatch Data Head:")
    print(match_data.head(2))