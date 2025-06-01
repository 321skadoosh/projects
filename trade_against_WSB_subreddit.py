import praw
from alpaca.trading.client import TradingClient
from textblob import TextBlob
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import csv
import re

# Reddit
CLIENT_ID = 'Enter ID'
CLIENT_SECRET = 'Enter secret ID'

# Alpaca
ALPACA_API_KEY = 'Enter ID'
ALPACA_SECRET_KEY = 'Enter secret ID'

reddit_api_client = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent="our trading bot | by /u/username"  # replace username
)

api = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
account = api.get_account()

def get_weekly_most_recent_link():
    # returns first 'What Are Your Moves Tomorrow' post
    user = reddit_api_client.redditor("wsbapp")
    for post in user.submissions.new(limit=None):
        if "What Are Your Moves Tomorrow" in post.title:
            return post

def extract_all_comments_from_post(post):
    # returns all commebts from given post
    post.comments.replace_more()
    return [comment.body for comment in post.comments]

STOCK_LIST = set()
with open("nasdaq_100.csv", "r") as f:
    reader = csv.reader(f)
    for val in list(reader)[1:]:
        if val:
            STOCK_LIST.add(val[1])

def extract_stock_from_comment(comment):
    # Convert the comment to uppercase to ensure case-insensitive matching
    comment_upper = comment.upper()

    # returns the comments that mention a relevant stock
    return set(re.findall(r"\w+", comment_upper)).intersection(STOCK_LIST)

def get_ticket_sentiment_mapping():
    # conducts sentiment analysis of a stock ticker and maps it to its average sentiment value from all its occurrences.
    mapping = {}

    post = get_weekly_most_recent_link()
    for comment in extract_all_comments_from_post(post):
        stocks_mentioned = extract_stock_from_comment(comment)
        for ticker in stocks_mentioned:
            if ticker not in mapping:
                mapping[ticker] = []
            mapping[ticker].append(TextBlob(comment).polarity)
    return {stock: sum(values) / len(values) for stock, values in mapping.items()}


def make_trade(ticker, short=False):
    order_side = OrderSide.SELL if short else OrderSide.BUY
    order_request = LimitOrderRequest(
        symbol=ticker,
        qty=1,
        side=order_side,
        limit_price=999.0, 
        time_in_force=TimeInForce.DAY  
    )

    try:
        api.submit_order(order_request)
        print(f"{'Shorted' if short else 'Purchased'} {ticker}")
    except Exception as e:
        print(f"Error submitting order for {ticker}: {e}")


for stock, sentiment in get_ticket_sentiment_mapping().items():
    # if overall sentiment is positive, stock is sold
    if sentiment > 0:
        make_trade(stock, short=True)
    # if overall sentiment is negative, stock is bought
    elif sentiment < 0:
        make_trade(stock)
