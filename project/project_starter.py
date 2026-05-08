import re
import json
import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from openai import OpenAI
from typing import Dict, List, Any



# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

dotenv.load_dotenv()


model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    api_base="https://openai.vocareum.com/v1",
)



# ----------------------------
# Helper structures / functions
# ----------------------------
CATALOG_DF = pd.DataFrame(paper_supplies)
CATALOG_LOOKUP = {
    row["item_name"].lower(): {
        "item_name": row["item_name"],
        "category": row["category"],
        "unit_price": float(row["unit_price"]),
    }
    for _, row in CATALOG_DF.iterrows()
}

KEYWORD_TO_ITEM = {
    "poster": "Large poster paper (24x36 inches)",
    "banner": "Rolls of banner paper (36-inch width)",
    "flyer": "Flyers",
    "invitation": "Invitation cards",
    "invite": "Invitation cards",
    "folder": "Presentation folders",
    "plate": "Paper plates",
    "plates": "Paper plates",
    "cup": "Paper cups",
    "cups": "Paper cups",
    "napkin": "Paper napkins",
    "napkins": "Paper napkins",
    "sticky": "Sticky notes",
    "notes": "Sticky notes",
    "notepad": "Notepads",
    "notepads": "Notepads",
    "envelope": "Envelopes",
    "envelopes": "Envelopes",
    "streamer": "Party streamers",
    "streamers": "Party streamers",
    "bag": "Paper party bags",
    "bags": "Paper party bags",
    "tag": "Name tags with lanyards",
    "tags": "Name tags with lanyards",
    "cardstock": "Cardstock",
    "glossy": "Glossy paper",
    "matte": "Matte paper",
    "photo paper": "Photo paper",
    "copy paper": "Standard copy paper",
    "a4": "A4 paper",
    "letterhead": "Letterhead paper",
    "legal": "Legal-size paper",
    "colored paper": "Colored paper",
    "construction paper": "Construction paper",
    "wrapping paper": "Wrapping paper",
}

UNIT_WORDS = r"(?:units?|sheets?|rolls?|plates?|cups?|napkins?|pads?|bags?|cards?|folders?|covers?|tags?|lanyards?|reams?|pieces?)"


def extract_request_text_and_date(full_request: str):
    """
    Extracts clean request text and request date from strings like:
    'Need 100 flyers (Date of request: 2025-01-03)'
    """
    match = re.search(r"\(Date of request:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\)\s*$", full_request)
    if match:
        request_date = match.group(1)
        request_text = re.sub(r"\(Date of request:\s*[0-9]{4}-[0-9]{2}-[0-9]{2}\)\s*$", "", full_request).strip()
        return request_text, request_date
    return full_request.strip(), datetime.now().strftime("%Y-%m-%d")


def infer_default_quantity(text: str) -> int:
    """
    Heuristic default quantity if no explicit quantity is found.
    """
    text_lower = text.lower()

    # Large order hints
    if any(word in text_lower for word in ["bulk", "large", "conference", "festival", "expo", "trade show"]):
        return 500
    if any(word in text_lower for word in ["medium", "office event", "seminar", "school event"]):
        return 250
    if any(word in text_lower for word in ["small", "team event", "birthday", "meeting"]):
        return 100

    # If any number exists in text, use the first reasonable one
    nums = re.findall(r"\b(\d+)\b", text_lower)
    if nums:
        value = int(nums[0])
        if value > 0:
            return value

    return 100


def find_quantity_for_label(text: str, label: str) -> int:
    """
    Attempts to find a quantity associated with a given item label/keyword.
    """
    label_escaped = re.escape(label.lower())

    patterns = [
        rf"(\d+)\s+(?:{UNIT_WORDS}\s+)?(?:of\s+)?{label_escaped}",
        rf"(\d+)\s+(?:{UNIT_WORDS}\s+)?{label_escaped}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))

    return infer_default_quantity(text)

def normalize_item_name(item_name: str) -> str | None:
    """
    Normalize natural-language item names to the closest supported catalog item.

    Args:
        item_name (str): Raw item name from the agent or user request.

    Returns:
        str | None: A supported catalog item name if matched, otherwise None.
    """
    if not item_name:
        return None

    name = item_name.strip().lower()

    alias_map = {
        "reams of paper": "Standard copy paper",
        "ream of paper": "Standard copy paper",
        "copy paper": "Standard copy paper",
        "printer paper": "Standard copy paper",
        "a4 sheets": "A4 paper",
        "a4 sheet": "A4 paper",
        "a4 glossy paper": "Glossy paper",
        "glossy a4 paper": "Glossy paper",
        "colored sheets": "Colored paper",
        "colored paper": "Colored paper",
        "heavy cardstock": "Cardstock",
        "white cardstock": "Cardstock",
        "poster sheets": "Large poster paper (24x36 inches)",
        "banner sheets": "Rolls of banner paper (36-inch width)",
    }

    # direct alias hit
    if name in alias_map:
        return alias_map[name]

    # exact catalog match
    if name in CATALOG_LOOKUP:
        return CATALOG_LOOKUP[name]["item_name"]

    # partial heuristic matches
    if "a4" in name:
        return "A4 paper"
    if "glossy" in name:
        return "Glossy paper"
    if "cardstock" in name:
        return "Cardstock"
    if "colored" in name:
        return "Colored paper"
    if "poster" in name:
        return "Large poster paper (24x36 inches)"
    if "banner" in name:
        return "Rolls of banner paper (36-inch width)"
    if "ream" in name or "paper" in name:
        return "Standard copy paper"

    # explicitly unsupported but recognized cases
    if "a3" in name:
        return None

    return None

def parse_requested_items(request_text: str) -> List[Dict]:
    """
    Parses request text into catalog items and quantities.
    Uses exact item matches first, then keyword-based fallback.
    """
    text_lower = request_text.lower()
    parsed_items = []
    seen = set()

    # 1) Exact item-name matches
    catalog_names_sorted = sorted(CATALOG_LOOKUP.keys(), key=len, reverse=True)
    for item_name_lower in catalog_names_sorted:
        if item_name_lower in text_lower:
            catalog_item = CATALOG_LOOKUP[item_name_lower]
            qty = find_quantity_for_label(request_text, item_name_lower)
            if catalog_item["item_name"] not in seen:
                parsed_items.append({
                    "item_name": catalog_item["item_name"],
                    "quantity": int(qty),
                    "unit_price": float(catalog_item["unit_price"]),
                    "category": catalog_item["category"],
                })
                seen.add(catalog_item["item_name"])

    # 2) Keyword fallback
    if not parsed_items:
        for keyword, item_name in KEYWORD_TO_ITEM.items():
            if keyword in text_lower and item_name not in seen:
                catalog_item = CATALOG_LOOKUP[item_name.lower()]
                qty = find_quantity_for_label(request_text, keyword)
                parsed_items.append({
                    "item_name": catalog_item["item_name"],
                    "quantity": int(qty),
                    "unit_price": float(catalog_item["unit_price"]),
                    "category": catalog_item["category"],
                })
                seen.add(item_name)

    # 3) Generic print fallback
    if not parsed_items:
        if any(word in text_lower for word in ["print", "copies", "copy", "documents", "handouts"]):
            fallback_item = CATALOG_LOOKUP["standard copy paper"]
            qty = infer_default_quantity(request_text)
            parsed_items.append({
                "item_name": fallback_item["item_name"],
                "quantity": int(qty),
                "unit_price": float(fallback_item["unit_price"]),
                "category": fallback_item["category"],
            })

    return parsed_items


def extract_search_terms(request_text: str, parsed_items: List[Dict]) -> List[str]:
    """
    Builds search terms for quote-history lookup.
    """
    terms = []

    for item in parsed_items:
        item_words = item["item_name"].lower().split()
        if item_words:
            terms.extend(item_words[:2])

    request_words = re.findall(r"[a-zA-Z]+", request_text.lower())
    request_words = [w for w in request_words if len(w) > 3]
    terms.extend(request_words[:5])

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for term in terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)

    return deduped[:6]


def get_catalog_item(item_name: str) -> Dict | None:
    """
    Return catalog metadata for a supported item, or None if unknown.
    """
    normalized_name = normalize_item_name(item_name)

    if normalized_name is None:
        return None

    return CATALOG_LOOKUP.get(normalized_name.lower())



"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent

@tool
def check_inventory_for_item(item_name: str, as_of_date: str) -> Dict:
    """
    Check current stock, threshold, and pricing details for one item.

    Args:
        item_name (str): Name of the requested or catalog item.
        as_of_date (str): Inventory date in YYYY-MM-DD format.

    Returns:
        Dict: Stock details or a structured message if the item is unsupported.
    """
    catalog_item = get_catalog_item(item_name)

    if catalog_item is None:
        return {
            "success": False,
            "item_name": item_name,
            "normalized_item_name": None,
            "as_of_date": as_of_date,
            "message": f"Unsupported or unknown item: {item_name}",
            "current_stock": 0,
            "min_stock_level": 0,
            "unit_price": None,
            "category": None,
            "below_minimum": False,
        }

    normalized_name = catalog_item["item_name"]

    stock_df = get_stock_level(normalized_name, as_of_date)
    current_stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inv_match = inventory_df[inventory_df["item_name"].str.lower() == normalized_name.lower()]

    if not inv_match.empty:
        min_stock_level = int(inv_match["min_stock_level"].iloc[0])
        unit_price = float(inv_match["unit_price"].iloc[0])
        category = str(inv_match["category"].iloc[0])
    else:
        min_stock_level = 0
        unit_price = float(catalog_item["unit_price"])
        category = str(catalog_item["category"])

    return {
        "success": True,
        "item_name": item_name,
        "normalized_item_name": normalized_name,
        "as_of_date": as_of_date,
        "current_stock": current_stock,
        "min_stock_level": min_stock_level,
        "unit_price": unit_price,
        "category": category,
        "below_minimum": current_stock < min_stock_level if min_stock_level > 0 else False,
        "message": f"Inventory checked for normalized item '{normalized_name}'.",
    }

@tool
def check_inventory_for_request(request_text: str, as_of_date: str) -> Dict:
    """
    Parse a request and check whether inventory can fulfill it.

    Args:
        request_text (str): Natural-language customer request.
        as_of_date (str): Inventory date in YYYY-MM-DD format.

    Returns:
        Dict: parsed items, shortages, and whether all requested items are available.
    """
    parsed_items = parse_requested_items(request_text)
    if not parsed_items:
        return {
            "parsed_items": [],
            "all_available": False,
            "shortages": [],
            "message": "Could not determine requested items from the request."
        }

    inventory_snapshot = get_all_inventory(as_of_date)
    shortages = []
    item_status = []

    for item in parsed_items:
        available = int(inventory_snapshot.get(item["item_name"], 0))
        shortage_qty = max(0, item["quantity"] - available)

        item_status.append({
            "item_name": item["item_name"],
            "requested_quantity": int(item["quantity"]),
            "available_quantity": available,
            "unit_price": float(item["unit_price"]),
        })

        if shortage_qty > 0:
            shortages.append({
                "item_name": item["item_name"],
                "required_quantity": int(item["quantity"]),
                "available_quantity": available,
                "shortage_quantity": shortage_qty,
                "unit_price": float(item["unit_price"]),
            })

    return {
        "parsed_items": parsed_items,
        "item_status": item_status,
        "all_available": len(shortages) == 0,
        "shortages": shortages,
    }


# Tools for quoting agent

@tool
def lookup_quote_history(request_text: str, limit: int = 3) -> List[Dict]:
    """
    Search historical quote examples relevant to the request.

    Args:
        request_text (str): Natural-language customer request.
        limit (int): Max number of history rows to return.

    Returns:
        List[Dict]: Matching historical quote examples.
    """
    parsed_items = parse_requested_items(request_text)
    search_terms = extract_search_terms(request_text, parsed_items)
    return search_quote_history(search_terms, limit=limit)


@tool
def generate_customer_quote(request_text: str, as_of_date: str, markup: float = 1.6) -> Dict:
    """
    Create a quote estimate from the parsed request and inventory state.

    Args:
        request_text (str): Natural-language customer request.
        as_of_date (str): Quote date in YYYY-MM-DD format.
        markup (float): Sales markup applied to unit prices.

    Returns:
        Dict: Quote details, availability, historical context, and pricing summary.
    """
    # Normalize markup safely
    if markup is None:
        markup = 1.6
    else:
        try:
            markup = float(markup)
        except (TypeError, ValueError):
            markup = 1.6

    parsed_items = parse_requested_items(request_text)

    if not parsed_items:
        return {
            "success": False,
            "message": "Unable to identify requested items for quoting.",
            "parsed_items": [],
            "line_items": [],
            "total_amount": 0.0,
            "can_fulfill_now": False,
            "shortages": [],
            "historical_examples": [],
            "quote_explanation": "No supported catalog items could be identified from the request.",
        }

    inventory_check = check_inventory_for_request(request_text, as_of_date)
    history = lookup_quote_history(request_text, limit=3)

    line_items = []
    total_amount = 0.0

    for item in parsed_items:
        quantity = int(item["quantity"])
        unit_price = float(item["unit_price"])

        quoted_unit_price = round(unit_price * markup, 2)
        line_total = round(quantity * quoted_unit_price, 2)
        total_amount += line_total

        line_items.append({
            "item_name": item["item_name"],
            "quantity": quantity,
            "unit_price": unit_price,
            "quoted_unit_price": quoted_unit_price,
            "line_total": line_total,
        })

    explanation_parts = [f"Quote prepared for {len(line_items)} item(s)."]

    if inventory_check.get("all_available"):
        explanation_parts.append("All requested items are currently available in inventory.")
    else:
        explanation_parts.append("Some requested items are not fully available and may require restocking.")

    if history:
        explanation_parts.append(f"Used {len(history)} historical quote example(s) for reference.")

    return {
        "success": True,
        "as_of_date": as_of_date,
        "parsed_items": parsed_items,
        "line_items": line_items,
        "total_amount": round(total_amount, 2),
        "can_fulfill_now": inventory_check.get("all_available", False),
        "shortages": inventory_check.get("shortages", []),
        "historical_examples": history,
        "quote_explanation": " ".join(explanation_parts),
    }
# Tools for ordering agent

@tool
def place_restock_order(request_text: str, as_of_date: str) -> Dict:
    """
    Place supplier stock orders for any missing inventory needed for the request.

    Args:
        request_text (str): Natural-language customer request.
        as_of_date (str): Request date in YYYY-MM-DD format.

    Returns:
        Dict: status of restock operation.
    """
    inventory_check = check_inventory_for_request(request_text, as_of_date)

    if not inventory_check["parsed_items"]:
        return {
            "success": False,
            "status": "failed",
            "message": "Could not parse requested items; no restock order placed.",
            "orders": [],
        }

    if inventory_check["all_available"]:
        return {
            "success": True,
            "status": "not_needed",
            "message": "No restock required; all items already available.",
            "orders": [],
        }

    shortages = inventory_check["shortages"]
    current_cash = get_cash_balance(as_of_date)

    total_purchase_cost = 0.0
    proposed_orders = []

    for shortage in shortages:
        item_name = shortage["item_name"]
        qty = int(shortage["shortage_quantity"])
        unit_price = float(shortage["unit_price"])
        total_cost = qty * unit_price
        delivery_date = get_supplier_delivery_date(as_of_date, qty)

        proposed_orders.append({
            "item_name": item_name,
            "quantity": qty,
            "unit_price": unit_price,
            "total_cost": round(total_cost, 2),
            "delivery_date": delivery_date,
        })
        total_purchase_cost += total_cost

    if total_purchase_cost > current_cash:
        return {
            "success": False,
            "status": "insufficient_cash",
            "message": f"Insufficient cash to restock. Needed ${total_purchase_cost:.2f}, available ${current_cash:.2f}.",
            "orders": proposed_orders,
            "required_cash": round(total_purchase_cost, 2),
            "available_cash": round(current_cash, 2),
        }

    created_transaction_ids = []
    for order in proposed_orders:
        txn_id = create_transaction(
            item_name=order["item_name"],
            transaction_type="stock_orders",
            quantity=int(order["quantity"]),
            price=float(order["total_cost"]),
            date=order["delivery_date"],
        )
        created_transaction_ids.append(txn_id)

    latest_delivery = max(order["delivery_date"] for order in proposed_orders)

    return {
        "success": True,
        "status": "ordered",
        "message": f"Restock order placed for {len(proposed_orders)} item(s).",
        "orders": proposed_orders,
        "transaction_ids": created_transaction_ids,
        "latest_delivery_date": latest_delivery,
        "total_purchase_cost": round(total_purchase_cost, 2),
    }


@tool
def fulfill_customer_order(request_text: str, as_of_date: str, markup: float = 1.6) -> Dict:
    """
    Fulfill a customer order from current inventory by creating sales transactions.

    Args:
        request_text (str): Natural-language customer request.
        as_of_date (str): Sale date in YYYY-MM-DD format.
        markup (float, optional): Sales markup applied to the base unit price.

    Returns:
        Dict: sale result and transaction details.
    """
    if markup is None:
        markup = 1.6
    else:
        markup = float(markup)

    quote = generate_customer_quote(request_text, as_of_date, markup=markup)
    if not quote["success"]:
        return {
            "success": False,
            "status": "failed",
            "message": "Unable to generate quote; cannot fulfill order.",
        }

    if not quote["can_fulfill_now"]:
        return {
            "success": False,
            "status": "awaiting_stock",
            "message": "Cannot fulfill order yet because inventory is insufficient.",
            "shortages": quote["shortages"],
        }

    sale_transaction_ids = []
    for line in quote["line_items"]:
        txn_id = create_transaction(
            item_name=line["item_name"],
            transaction_type="sales",
            quantity=int(line["quantity"]),
            price=float(line["line_total"]),
            date=as_of_date,
        )
        sale_transaction_ids.append(txn_id)

    return {
        "success": True,
        "status": "fulfilled",
        "message": "Customer order fulfilled successfully.",
        "transaction_ids": sale_transaction_ids,
        "line_items": quote["line_items"],
        "total_amount": quote["total_amount"],
    }


# Set up your agents and create an orchestration agent that will manage them.


# Set up your agents and create an orchestration agent that will manage them.

class InventoryAgent(ToolCallingAgent):
    def __init__(self):
        super().__init__(
            name="InventoryAgent",
            model=model,
            tools=[check_inventory_for_item, check_inventory_for_request],
        )
        
        self.description = (
            "Checks stock availability for requested paper items, identifies shortages, "
            "and determines whether restocking is needed."
            )


    def handle(self, request_text: str, as_of_date: str) -> str:
        prompt = f"""
        You are the Inventory Agent for Munder Difflin.

        Your responsibilities:
        - Review the customer request
        - Identify requested inventory items
        - Check whether requested quantities are currently available
        - Identify any shortages
        - Determine if restocking is needed

        Customer request:
        {request_text}

        Date:
        {as_of_date}

        Use the available inventory tools to inspect the request and determine stock status.

        Return a concise summary including:
        - requested items
        - whether all items are available
        - any shortages
        """
        return self.run(prompt)


class QuotingAgent(ToolCallingAgent):
    def __init__(self):
        super().__init__(
            name="QuotingAgent",
            model=model,
            tools=[lookup_quote_history, generate_customer_quote],
        )
        self.description = "Generates customer quotes and uses quote history when helpful."

    def handle(self, request_text: str, as_of_date: str) -> str:
        prompt = f"""
        You are the Quoting Agent for Munder Difflin.

        Your responsibilities:
        - Review the customer request
        - Generate a quote for the requested items
        - Use historical quote data when useful
        - Clearly summarize the total price and whether the request can be fulfilled immediately

        Customer request:
        {request_text}

        Date:
        {as_of_date}

        Use the available quote tools to prepare the quote.

        Return a concise quote summary.
        """
        return self.run(prompt)


class OrderingAgent(ToolCallingAgent):
    def __init__(self):
        super().__init__(
            name="OrderingAgent",
            model=model,
            tools=[place_restock_order, fulfill_customer_order],
        )
        self.description = "Fulfills orders or places restock orders depending on inventory and delivery timelines."

    def handle_fulfillment(self, request_text: str, as_of_date: str) -> str:
        prompt = f"""
        You are the Ordering Agent for Munder Difflin.

        Your responsibilities:
        - Fulfill the customer order if stock is available
        - If stock is insufficient, place a restock order
        - Consider delivery timelines
        - Return an operational summary

        Customer request:
        {request_text}

        Date:
        {as_of_date}

        Use the available ordering tools to either fulfill the order or place a restock order.

        Return a concise operational summary.
        """
        return self.run(prompt)

class MunderDifflinOrchestrator(ToolCallingAgent):
    def __init__(self):
        self.inventory_agent = InventoryAgent()
        self.quoting_agent = QuotingAgent()
        self.ordering_agent = OrderingAgent()

        super().__init__(
            name="MunderDifflinOrchestrator",
            model=model,
            tools=[],
            managed_agents=[
                self.inventory_agent,
                self.quoting_agent,
                self.ordering_agent,
            ],
        )
        
        self.description = (
            "Main orchestrator for customer requests. Delegates work to inventory, quoting, "
            "and ordering agents, and returns a final customer-facing summary."
        )


    def handle_request(self, full_request: str) -> str:
        request_text, request_date = extract_request_text_and_date(full_request)
        self.memory.steps = []

        prompt = f"""
        You are the Munder Difflin Orchestrator Agent.

        You must coordinate the other agents to process this customer request.

        Customer request: {request_text}
        Request date: {request_date}

        Available managed agents:
        - InventoryAgent
        - QuotingAgent
        - OrderingAgent

        Required workflow:
        1. Delegate to InventoryAgent to determine whether requested items are available and whether there are shortages.
        2. Delegate to QuotingAgent to generate the customer quote.
        3. Delegate to OrderingAgent to either:
        - fulfill the order if inventory is available, or
        - place a restock order if inventory is insufficient.
        4. Return a final concise response for the customer.

        Your final response must include:
        - a short inventory summary
        - quote amount
        - whether the order was fulfilled or restocked
        - estimated delivery timing if restocking was needed

        You must delegate the work to the managed agents instead of doing it yourself.
        """

        return self.run(prompt)

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    
    orchestrator = MunderDifflinOrchestrator()


    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = orchestrator.handle_request(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")
        
        response_lower = str(response).lower()

        if "fulfilled" in response_lower:
            request_status = "Fulfilled"
        elif "restock" in response_lower or "delivery" in response_lower or "shortage" in response_lower:
            request_status = "Unfulfilled / Restock Required"
        else:
            request_status = "Other / Review Needed"


        
        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "status": request_status,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    results_df = pd.DataFrame(results)
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    
    print("\n===== REQUEST FULFILLMENT SUMMARY =====")
    summary_table = results_df[["request_id", "request_date", "status"]]
    print(summary_table.to_string(index=False))

    print("\n===== STATUS COUNTS =====")
    print(results_df["status"].value_counts().to_string())

    return results


if __name__ == "__main__":
    results = run_test_scenarios()
